// ============================================================================
// file: dualtier_credit_router_oracle.hpp
// Independent CPU oracle (stateful) + output check helpers.
// ============================================================================

#ifndef DUALTIER_CREDIT_ROUTER_ORACLE_HPP_
#define DUALTIER_CREDIT_ROUTER_ORACLE_HPP_

#include "dualtier_credit_router_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <deque>
#include <sstream>
#include <string>
#include <vector>

struct DtrExpected {
    std::vector<int32_t> s2_logits;    // [N * E]
    std::vector<int32_t> route_nodes;  // [N]
    std::vector<int32_t> route_pe_be;  // [N]
    int32_t log_len;
    std::vector<uint64_t> event_log;   // [log_len]
    std::vector<int32_t> counts;       // [E]
    std::vector<int32_t> offsets;      // [E + 1]
    std::vector<int32_t> packed_gid;   // [offsets[E]]
    std::vector<std::vector<int32_t>> packed_out;  // [offsets[E]][D]
    std::vector<uint32_t> credit_out;  // [E]
    uint64_t state_checksum;
};

static inline uint64_t dtr_oracle_fnv_byte(uint64_t h, uint8_t b) {
    return (h ^ static_cast<uint64_t>(b)) * DTR_FNV_PRIME;
}

static inline uint64_t dtr_oracle_fnv_u32(uint64_t h, uint32_t v) {
    for (int m = 0; m < 4; ++m) {
        h = dtr_oracle_fnv_byte(h, (uint8_t)((v >> (8 * m)) & 0xFF));
    }
    return h;
}

static inline uint64_t dtr_oracle_fnv_u64(uint64_t h, uint64_t v) {
    for (int m = 0; m < 8; ++m) {
        h = dtr_oracle_fnv_byte(h, (uint8_t)((v >> (8 * m)) & 0xFF));
    }
    return h;
}

struct DtrBacklogEntry {
    uint32_t gid;
    std::vector<int8_t> x;  // [D]
};

// Stateful CPU oracle mirroring the persistent device state.
struct DtrOracle {
    int E = 0;
    int P = 0;
    int D = 0;
    int H = 0;
    int ccap = 0;
    int bq = 0;
    uint32_t token_counter = 0;
    std::vector<uint32_t> credit;                    // [E]
    std::vector<std::deque<DtrBacklogEntry>> backlog;  // [E]

    void reset(const DtrRunSpec& shape) {
        E = shape.E;
        P = shape.P;
        D = shape.D;
        H = shape.H;
        ccap = shape.ccap;
        bq = shape.bq;
        token_counter = 0;
        credit.assign((size_t)E, (uint32_t)ccap);
        backlog.assign((size_t)E, std::deque<DtrBacklogEntry>());
    }

    // FFN of expert e on feature row xv with THIS run's weights.
    void ffn(const DtrRunSpec& run, const int8_t* w1, const int8_t* w2,
             int e, const int8_t* xv, int32_t* out) const {
        std::vector<int32_t> h((size_t)H, 0);
        for (int j = 0; j < H; ++j) {
            int32_t acc = 0;
            const int8_t* w1row = w1 + ((size_t)e * H + j) * D;
            for (int d = 0; d < D; ++d) {
                acc += (int32_t)w1row[d] * (int32_t)xv[d];
            }
            acc = acc > 0 ? acc : 0;
            acc >>= run.qshift;
            h[(size_t)j] = acc < 127 ? acc : 127;
        }
        for (int d = 0; d < D; ++d) {
            int32_t acc = 0;
            const int8_t* w2row = w2 + ((size_t)e * D + d) * H;
            for (int j = 0; j < H; ++j) {
                acc += (int32_t)w2row[j] * h[(size_t)j];
            }
            out[d] = acc;
        }
    }

    void run_round(
        const DtrRunSpec& run,
        const int8_t* x,
        const int8_t* wnode,
        const int8_t* wexp,
        const int8_t* w1,
        const int8_t* w2,
        DtrExpected* expected) {
        const int N = run.N;
        const int S = E / P;
        const uint32_t base = token_counter;

        // Refill.
        std::vector<uint32_t> cred0((size_t)E);
        for (int e = 0; e < E; ++e) {
            uint32_t c = credit[(size_t)e] + (uint32_t)run.refill;
            cred0[(size_t)e] = c < (uint32_t)ccap ? c : (uint32_t)ccap;
        }

        // Routing.
        expected->s2_logits.assign((size_t)N * E, 0);
        expected->route_nodes.assign((size_t)N, 0);
        expected->route_pe_be.assign((size_t)N, 0);
        std::vector<int> pe((size_t)N), be((size_t)N);

        for (int t = 0; t < N; ++t) {
            const int8_t* xrow = x + (size_t)t * D;
            int pn = -1, bn = -1;
            int32_t pn_s = 0, bn_s = 0;
            for (int p = 0; p < P; ++p) {
                int32_t s = 0;
                const int8_t* wrow = wnode + (size_t)p * D;
                for (int d = 0; d < D; ++d) {
                    s += (int32_t)xrow[d] * (int32_t)wrow[d];
                }
                if (pn < 0 || s > pn_s) {
                    bn = pn;
                    bn_s = pn_s;
                    pn = p;
                    pn_s = s;
                } else if (bn < 0 || s > bn_s) {
                    bn = p;
                    bn_s = s;
                }
            }
            for (int e = 0; e < E; ++e) {
                int32_t s = 0;
                const int8_t* wrow = wexp + (size_t)e * D;
                for (int d = 0; d < D; ++d) {
                    s += (int32_t)xrow[d] * (int32_t)wrow[d];
                }
                expected->s2_logits[(size_t)t * E + e] = s;
            }
            const int32_t* s2row = expected->s2_logits.data() + (size_t)t * E;
            int pbest = pn * S, bbest = bn * S;
            for (int q = 1; q < S; ++q) {
                if (s2row[pn * S + q] > s2row[pbest]) pbest = pn * S + q;
                if (s2row[bn * S + q] > s2row[bbest]) bbest = bn * S + q;
            }
            pe[(size_t)t] = pbest;
            be[(size_t)t] = bbest;
            expected->route_nodes[(size_t)t] = (pn << 16) | bn;
            expected->route_pe_be[(size_t)t] = (pbest << 16) | bbest;
        }

        // Per-expert phase bookkeeping.
        std::vector<uint32_t> c1((size_t)E), c2((size_t)E), c3((size_t)E);
        std::vector<int> a0((size_t)E, 0), a1((size_t)E, 0), a2((size_t)E, 0);

        struct Delivery {
            int expert;
            int phase;      // 0, 1, 2
            int rank;
            uint32_t gid;
            const int8_t* xv;          // for phases 1/2
            std::vector<int8_t> xcopy; // for phase 0 (owned)
        };
        std::vector<Delivery> deliveries;
        struct Phase3Event {
            int expert;
            int rank;
            uint32_t gid;
            bool queued;
        };
        std::vector<Phase3Event> phase3;

        // Phase 0: backlog drain.
        std::vector<int> drained_now((size_t)E, 0);
        for (int e = 0; e < E; ++e) {
            const int a = std::min<int>((int)backlog[(size_t)e].size(),
                                        (int)cred0[(size_t)e]);
            a0[(size_t)e] = a;
            for (int r = 0; r < a; ++r) {
                Delivery dl;
                dl.expert = e;
                dl.phase = 0;
                dl.rank = r;
                dl.gid = backlog[(size_t)e].front().gid;
                dl.xv = nullptr;
                dl.xcopy = std::move(backlog[(size_t)e].front().x);
                backlog[(size_t)e].pop_front();
                deliveries.push_back(std::move(dl));
            }
            c1[(size_t)e] = cred0[(size_t)e] - (uint32_t)a;
        }

        // Ranked attempt helper: sort token indices by (key desc, gid asc).
        auto ranked = [&](std::vector<int>& toks, const int* target,
                          const int32_t* /*unused*/) {
            std::stable_sort(toks.begin(), toks.end(),
                [&](int ta, int tb) {
                    const int32_t ka = expected->s2_logits[
                        (size_t)ta * E + target[ta]];
                    const int32_t kb = expected->s2_logits[
                        (size_t)tb * E + target[tb]];
                    if (ka != kb) return ka > kb;
                    return ta < tb;
                });
        };

        // Phase 1.
        std::vector<std::vector<int>> att1((size_t)E);
        for (int t = 0; t < N; ++t) att1[(size_t)pe[(size_t)t]].push_back(t);
        std::vector<int> rej1;
        for (int e = 0; e < E; ++e) {
            std::vector<int>& toks = att1[(size_t)e];
            ranked(toks, pe.data(), nullptr);
            const int adm = std::min<int>((int)toks.size(),
                                          (int)c1[(size_t)e]);
            a1[(size_t)e] = adm;
            for (int r = 0; r < adm; ++r) {
                Delivery dl;
                dl.expert = e;
                dl.phase = 1;
                dl.rank = r;
                dl.gid = base + (uint32_t)toks[(size_t)r];
                dl.xv = x + (size_t)toks[(size_t)r] * D;
                deliveries.push_back(std::move(dl));
            }
            for (size_t r = (size_t)adm; r < toks.size(); ++r) {
                rej1.push_back(toks[r]);
            }
            c2[(size_t)e] = c1[(size_t)e] - (uint32_t)adm;
        }

        // Phase 2.
        std::vector<std::vector<int>> att2((size_t)E);
        for (int t : rej1) att2[(size_t)be[(size_t)t]].push_back(t);
        std::vector<int> rej2;
        for (int e = 0; e < E; ++e) {
            std::vector<int>& toks = att2[(size_t)e];
            ranked(toks, be.data(), nullptr);
            const int adm = std::min<int>((int)toks.size(),
                                          (int)c2[(size_t)e]);
            a2[(size_t)e] = adm;
            for (int r = 0; r < adm; ++r) {
                Delivery dl;
                dl.expert = e;
                dl.phase = 2;
                dl.rank = r;
                dl.gid = base + (uint32_t)toks[(size_t)r];
                dl.xv = x + (size_t)toks[(size_t)r] * D;
                deliveries.push_back(std::move(dl));
            }
            for (size_t r = (size_t)adm; r < toks.size(); ++r) {
                rej2.push_back(toks[r]);
            }
            c3[(size_t)e] = c2[(size_t)e] - (uint32_t)adm;
        }

        // Phase 3.
        std::vector<std::vector<int>> att3((size_t)E);
        for (int t : rej2) att3[(size_t)pe[(size_t)t]].push_back(t);
        for (int e = 0; e < E; ++e) {
            std::vector<int>& toks = att3[(size_t)e];
            ranked(toks, pe.data(), nullptr);
            const int already = (int)backlog[(size_t)e].size();
            const int free_slots = bq - already;
            for (int r = 0; r < (int)toks.size(); ++r) {
                Phase3Event ev;
                ev.expert = e;
                ev.rank = r;
                ev.gid = base + (uint32_t)toks[(size_t)r];
                ev.queued = r < free_slots;
                phase3.push_back(ev);
                if (ev.queued) {
                    DtrBacklogEntry entry;
                    entry.gid = ev.gid;
                    entry.x.assign(
                        x + (size_t)toks[(size_t)r] * D,
                        x + (size_t)(toks[(size_t)r] + 1) * D);
                    backlog[(size_t)e].push_back(std::move(entry));
                }
            }
        }

        // Commit credits + counter.
        for (int e = 0; e < E; ++e) credit[(size_t)e] = c3[(size_t)e];
        token_counter += (uint32_t)N;

        // counts / offsets.
        expected->counts.assign((size_t)E, 0);
        expected->offsets.assign((size_t)E + 1, 0);
        for (int e = 0; e < E; ++e) {
            expected->counts[(size_t)e] =
                a0[(size_t)e] + a1[(size_t)e] + a2[(size_t)e];
        }
        for (int e = 0; e < E; ++e) {
            expected->offsets[(size_t)e + 1] =
                expected->offsets[(size_t)e] + expected->counts[(size_t)e];
        }

        // Packed arrays + FFN.
        const int total = expected->offsets[(size_t)E];
        expected->packed_gid.assign((size_t)total, 0);
        expected->packed_out.assign(
            (size_t)total, std::vector<int32_t>((size_t)D, 0));
        for (const Delivery& dl : deliveries) {
            int pos = expected->offsets[(size_t)dl.expert];
            if (dl.phase >= 1) pos += a0[(size_t)dl.expert];
            if (dl.phase >= 2) pos += a1[(size_t)dl.expert];
            pos += dl.rank;
            expected->packed_gid[(size_t)pos] = (int32_t)dl.gid;
            const int8_t* xv = dl.phase == 0 ? dl.xcopy.data() : dl.xv;
            ffn(run, w1, w2, dl.expert,
                xv, expected->packed_out[(size_t)pos].data());
        }

        // Event log: ascending (phase, expert, rank).
        std::vector<uint64_t> log;
        for (const Delivery& dl : deliveries) {
            if (dl.phase != 0) continue;
            const uint32_t start = cred0[(size_t)dl.expert];
            log.push_back(dtr_make_log_word(
                dl.gid, dl.expert, DTR_ACT_DELIV_BACKLOG, 0,
                start - (uint32_t)dl.rank - 1u));
        }
        for (const Delivery& dl : deliveries) {
            if (dl.phase != 1) continue;
            const uint32_t start = c1[(size_t)dl.expert];
            log.push_back(dtr_make_log_word(
                dl.gid, dl.expert, DTR_ACT_DELIV_PRIMARY, 1,
                start - (uint32_t)dl.rank - 1u));
        }
        for (const Delivery& dl : deliveries) {
            if (dl.phase != 2) continue;
            const uint32_t start = c2[(size_t)dl.expert];
            log.push_back(dtr_make_log_word(
                dl.gid, dl.expert, DTR_ACT_DELIV_BACKUP, 2,
                start - (uint32_t)dl.rank - 1u));
        }
        for (const Phase3Event& ev : phase3) {
            log.push_back(dtr_make_log_word(
                ev.gid, ev.expert,
                ev.queued ? DTR_ACT_QUEUED : DTR_ACT_DROPPED, 3,
                c3[(size_t)ev.expert] & 0xFFFFu));
        }
        // Deliveries were appended expert-ascending within each phase and
        // rank-ascending within each expert, so `log` is already in the
        // normative (phase, expert, rank) order.
        expected->event_log = std::move(log);
        expected->log_len = (int32_t)expected->event_log.size();

        // credit_out + state checksum.
        expected->credit_out.assign((size_t)E, 0);
        for (int e = 0; e < E; ++e) {
            expected->credit_out[(size_t)e] = credit[(size_t)e];
        }
        uint64_t root = DTR_FNV_BASIS;
        for (int e = 0; e < E; ++e) {
            uint64_t ed = DTR_FNV_BASIS;
            ed = dtr_oracle_fnv_u32(ed, credit[(size_t)e]);
            ed = dtr_oracle_fnv_u32(
                ed, (uint32_t)backlog[(size_t)e].size());
            for (const DtrBacklogEntry& entry : backlog[(size_t)e]) {
                uint64_t qd = DTR_FNV_BASIS;
                qd = dtr_oracle_fnv_u32(qd, entry.gid);
                for (int d = 0; d < D; ++d) {
                    qd = dtr_oracle_fnv_byte(qd, (uint8_t)entry.x[(size_t)d]);
                }
                ed = dtr_oracle_fnv_u64(ed, qd);
            }
            root = dtr_oracle_fnv_u64(root, ed);
        }
        expected->state_checksum = root;
    }
};

struct DtrHostOutputsView {
    const int32_t* s2_logits;
    const int32_t* route_nodes;
    const int32_t* route_pe_be;
    const int32_t* log_len;
    const uint64_t* event_log;
    const int32_t* counts;
    const int32_t* offsets;
    const int32_t* packed_gid;
    const int32_t* packed_out;
    const uint32_t* credit_out;
    const uint64_t* state_checksum;
};

static inline bool dtr_check_outputs(
    const DtrRunSpec& run,
    const DtrExpected& expected,
    const DtrHostOutputsView& got,
    std::string* error) {
    const int N = run.N;
    const int E = run.E;
    const int D = run.D;

    for (size_t i = 0; i < (size_t)N * E; ++i) {
        if (got.s2_logits[i] != expected.s2_logits[i]) {
            if (error) {
                std::ostringstream oss;
                oss << "s2_logits mismatch at flat " << i << ": got "
                    << got.s2_logits[i] << ", expected "
                    << expected.s2_logits[i];
                *error = oss.str();
            }
            return false;
        }
    }
    for (int t = 0; t < N; ++t) {
        if (got.route_nodes[t] != expected.route_nodes[(size_t)t]) {
            if (error) {
                std::ostringstream oss;
                oss << "route_nodes mismatch at t=" << t << ": got "
                    << got.route_nodes[t] << ", expected "
                    << expected.route_nodes[(size_t)t];
                *error = oss.str();
            }
            return false;
        }
        if (got.route_pe_be[t] != expected.route_pe_be[(size_t)t]) {
            if (error) {
                std::ostringstream oss;
                oss << "route_pe_be mismatch at t=" << t << ": got "
                    << got.route_pe_be[t] << ", expected "
                    << expected.route_pe_be[(size_t)t];
                *error = oss.str();
            }
            return false;
        }
    }
    if (got.log_len[0] != expected.log_len) {
        if (error) {
            std::ostringstream oss;
            oss << "log_len mismatch: got " << got.log_len[0]
                << ", expected " << expected.log_len;
            *error = oss.str();
        }
        return false;
    }
    for (int i = 0; i < expected.log_len; ++i) {
        if (got.event_log[i] != expected.event_log[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "event_log mismatch at index " << i << ": got 0x"
                    << std::hex << got.event_log[i] << ", expected 0x"
                    << expected.event_log[(size_t)i];
                *error = oss.str();
            }
            return false;
        }
    }
    for (int e = 0; e < E; ++e) {
        if (got.counts[e] != expected.counts[(size_t)e]) {
            if (error) {
                std::ostringstream oss;
                oss << "counts mismatch at e=" << e << ": got "
                    << got.counts[e] << ", expected "
                    << expected.counts[(size_t)e];
                *error = oss.str();
            }
            return false;
        }
    }
    for (int e = 0; e <= E; ++e) {
        if (got.offsets[e] != expected.offsets[(size_t)e]) {
            if (error) {
                std::ostringstream oss;
                oss << "offsets mismatch at e=" << e << ": got "
                    << got.offsets[e] << ", expected "
                    << expected.offsets[(size_t)e];
                *error = oss.str();
            }
            return false;
        }
    }
    const int total = expected.offsets[(size_t)E];
    for (int pos = 0; pos < total; ++pos) {
        if (got.packed_gid[pos] != expected.packed_gid[(size_t)pos]) {
            if (error) {
                std::ostringstream oss;
                oss << "packed_gid mismatch at pos=" << pos << ": got "
                    << got.packed_gid[pos] << ", expected "
                    << expected.packed_gid[(size_t)pos];
                *error = oss.str();
            }
            return false;
        }
        for (int d = 0; d < D; ++d) {
            if (got.packed_out[(size_t)pos * D + d] !=
                expected.packed_out[(size_t)pos][(size_t)d]) {
                if (error) {
                    std::ostringstream oss;
                    oss << "packed_out mismatch at pos=" << pos << " d=" << d
                        << ": got " << got.packed_out[(size_t)pos * D + d]
                        << ", expected "
                        << expected.packed_out[(size_t)pos][(size_t)d];
                    *error = oss.str();
                }
                return false;
            }
        }
    }
    for (int e = 0; e < E; ++e) {
        if (got.credit_out[e] != expected.credit_out[(size_t)e]) {
            if (error) {
                std::ostringstream oss;
                oss << "credit_out mismatch at e=" << e << ": got "
                    << got.credit_out[e] << ", expected "
                    << expected.credit_out[(size_t)e];
                *error = oss.str();
            }
            return false;
        }
    }
    if (got.state_checksum[0] != expected.state_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "state_checksum mismatch: got " << got.state_checksum[0]
                << ", expected " << expected.state_checksum;
            *error = oss.str();
        }
        return false;
    }
    return true;
}

/*
INTERMEDIATE CHECKS REQUIRED BY GRADER

After every solution_run the grader should verify (all exact):
  1. s2_logits / route_nodes / route_pe_be  (routing GEMMs + selections).
  2. log_len + event_log prefix             (full admission bookkeeping:
                                             actions, ranks and per-event
                                             credit_after values).
  3. counts / offsets / packed_gid / packed_out (expert-major packing and
                                             the delivered-token FFN under
                                             the delivering run's weights).
  4. credit_out                             (persistent credit dynamics).
  5. state_checksum                          (three-level FNV over credits
                                             + backlog gids + STORED x
                                             vectors: any state error in
                                             any earlier run cascades).

The grader should additionally enforce:
  - Sentinels around every output allocation; unspecified regions
    (event_log beyond log_len, packed arrays beyond offsets[E]) are never
    compared but capacity bounds must hold.
  - Input immutability before/after solution_run.
  - Determinism replay: identical reset+run sequences twice must produce
    identical bytes.
  - Multi-run chains with weight/refill/qshift changes between runs and
    resets mid-case.
*/

#endif  // DUALTIER_CREDIT_ROUTER_ORACLE_HPP_
