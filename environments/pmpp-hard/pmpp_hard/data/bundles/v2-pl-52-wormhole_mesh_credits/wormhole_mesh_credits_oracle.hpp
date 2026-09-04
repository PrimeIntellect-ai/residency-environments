// file: wormhole_mesh_credits_oracle.hpp
//
// Host-side golden model for T52. This is an INDEPENDENT third implementation:
// a plain-C++ event-driven simulation with std::vector FIFOs. The reference.cu
// and naive.cu device implementations do not share any code with this file.

#ifndef WORMHOLE_MESH_CREDITS_ORACLE_HPP_
#define WORMHOLE_MESH_CREDITS_ORACLE_HPP_

#include "wormhole_mesh_credits_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <deque>
#include <sstream>
#include <string>
#include <vector>

struct WmcOracleResult {
    int64_t counters[WMC_COUNTER_COUNT] = {0};
    uint64_t fabric_event_hash = 0;
    uint64_t packet_hash = 0;
    uint64_t buffer_hash = 0;
    uint64_t credit_hash = 0;
    uint64_t credit_queue_hash = 0;
    uint64_t cycle_out = 0;
    uint64_t event_seq_out = 0;
};

static inline uint64_t wmc_o_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}
static inline void wmc_o_fnv(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = wmc_o_fnv_byte(v, b[i]);
    *h = v;
}

struct WmcOFlit {
    uint64_t packet_id;
    uint64_t flit_index;
    uint64_t flit_count;
    uint32_t src;
    uint32_t dst;
    uint8_t vc;
    uint8_t kind;
    uint64_t inject_seq;
    uint64_t entered_cycle;
    uint32_t credit_return_node;  // UINT32_MAX = none
    uint8_t credit_return_port;   // 255 = none
};

struct WmcOPacket {
    bool used = false;
    uint64_t packet_id = 0;
    uint32_t src = 0;
    uint32_t dst = 0;
    uint64_t flit_count = 0;
    uint8_t prefer_adaptive = 0;
    uint64_t injected_flits = 0;
    uint64_t delivered_flits = 0;
    uint64_t inject_seq = 0;
    uint8_t state = WMC_PS_QUEUED;
    uint32_t chosen_vc = UINT32_MAX;
};

struct WmcOCreditRec {
    uint64_t due_cycle;
    uint32_t node;
    uint8_t port;
    uint8_t vc;
    uint64_t amount;
    uint64_t seq_created;
};

struct WmcOracle {
    WmcProblemSpec spec{};
    int rows = 0, cols = 0, nodes = 0, vc = 0, cap = 0;
    int credit_latency = 0;
    int max_packets = 0, max_injq = 0, max_credit_events = 0;

    uint64_t cycle = 0;
    uint64_t event_seq = 0;
    uint64_t inject_seq_next = 1;

    std::vector<uint8_t> link_up;                      // [node*5+port]
    std::vector<std::deque<WmcOFlit>> buf;             // [(node*5+port)*vc+vc]
    std::vector<int64_t> credit;                       // [(node*5+port)*vc+vc]
    std::vector<WmcOPacket> ptable;                    // [max_packets]
    std::vector<std::vector<uint64_t>> injq;           // [node] -> packet ids
    std::vector<WmcOCreditRec> creditq;                // unsorted; sorted on read

    int64_t counters[WMC_COUNTER_COUNT];
    uint64_t run_event_hash;  // FNV of this run's events

    int bidx(int node, int port, int v) const { return (node * WMC_PORT_COUNT + port) * vc + v; }
    int lidx(int node, int port) const { return node * WMC_PORT_COUNT + port; }

    void init(const WmcProblemSpec& s) {
        spec = s;
        rows = s.rows; cols = s.cols; nodes = rows * cols;
        vc = s.vc_count; cap = s.buffer_cap_per_vc;
        credit_latency = s.credit_latency;
        max_packets = s.max_packets;
        max_injq = s.max_injection_queue_per_node;
        max_credit_events = s.max_credit_events;
        reset();
    }

    void reset() {
        cycle = 0; event_seq = 0; inject_seq_next = 1;
        link_up.assign((size_t)nodes * WMC_PORT_COUNT, 0);
        buf.assign((size_t)nodes * WMC_PORT_COUNT * vc, std::deque<WmcOFlit>());
        credit.assign((size_t)nodes * WMC_PORT_COUNT * vc, 0);
        ptable.assign((size_t)max_packets, WmcOPacket());
        injq.assign((size_t)nodes, std::vector<uint64_t>());
        creditq.clear();
        for (int n = 0; n < nodes; ++n) {
            for (int p = 0; p < WMC_PORT_COUNT; ++p) {
                bool valid = output_valid(n, p);
                link_up[(size_t)lidx(n, p)] = valid ? 1 : 0;
                for (int v = 0; v < vc; ++v) {
                    credit[(size_t)bidx(n, p, v)] = valid ? cap : 0;
                }
            }
        }
        for (int i = 0; i < WMC_COUNTER_COUNT; ++i) counters[i] = 0;
    }

    // geometry helpers
    int rowof(int n) const { return n / cols; }
    int colof(int n) const { return n % cols; }
    bool output_valid(int node, int port) const {
        if (port == WMC_PORT_LOCAL) return false;
        int r = rowof(node), c = colof(node);
        if (port == WMC_PORT_N) return r - 1 >= 0;
        if (port == WMC_PORT_S) return r + 1 < rows;
        if (port == WMC_PORT_E) return c + 1 < cols;
        if (port == WMC_PORT_W) return c - 1 >= 0;
        return false;
    }
    int neighbor(int node, int port) const {
        int r = rowof(node), c = colof(node);
        if (port == WMC_PORT_N) return (r - 1) * cols + c;
        if (port == WMC_PORT_S) return (r + 1) * cols + c;
        if (port == WMC_PORT_E) return r * cols + (c + 1);
        if (port == WMC_PORT_W) return r * cols + (c - 1);
        return -1;
    }
    static int opposite(int port) {
        switch (port) {
            case WMC_PORT_N: return WMC_PORT_S;
            case WMC_PORT_S: return WMC_PORT_N;
            case WMC_PORT_E: return WMC_PORT_W;
            case WMC_PORT_W: return WMC_PORT_E;
            default: return WMC_PORT_LOCAL;
        }
    }

    // ---- event emission ----
    void emit(uint8_t kind, uint32_t node_or, uint8_t port_or, uint8_t vc_or,
              uint64_t packet_or, uint64_t flit_index_or, uint64_t aux,
              uint32_t op_index) {
        uint64_t es = event_seq;
        uint64_t* h = &run_event_hash;
        wmc_o_fnv(h, &kind, 1);
        wmc_o_fnv(h, &es, 8);
        uint64_t cyc = cycle;
        wmc_o_fnv(h, &cyc, 8);
        wmc_o_fnv(h, &op_index, 4);
        wmc_o_fnv(h, &node_or, 4);
        wmc_o_fnv(h, &port_or, 1);
        wmc_o_fnv(h, &vc_or, 1);
        wmc_o_fnv(h, &packet_or, 8);
        wmc_o_fnv(h, &flit_index_or, 8);
        wmc_o_fnv(h, &aux, 8);
        event_seq = es + 1;
    }

    // ---- credit scheduling ----
    void schedule_credit_return(const WmcOFlit& f, uint32_t op_index) {
        (void)op_index;
        if (f.credit_return_node == UINT32_MAX) return;
        if ((int)creditq.size() >= max_credit_events) return;  // silent drop
        WmcOCreditRec rec;
        rec.due_cycle = cycle + (uint64_t)credit_latency;
        rec.node = f.credit_return_node;
        rec.port = f.credit_return_port;
        rec.vc = f.vc;
        rec.amount = 1;
        rec.seq_created = event_seq;  // creation-order tiebreak
        creditq.push_back(rec);
    }

    // canonical credit-queue ordering
    static bool creditq_less(const WmcOCreditRec& a, const WmcOCreditRec& b) {
        if (a.due_cycle != b.due_cycle) return a.due_cycle < b.due_cycle;
        if (a.node != b.node) return a.node < b.node;
        if (a.port != b.port) return a.port < b.port;
        if (a.vc != b.vc) return a.vc < b.vc;
        return a.seq_created < b.seq_created;
    }

    int find_packet(uint64_t id) const {
        for (int i = 0; i < max_packets; ++i)
            if (ptable[(size_t)i].used && ptable[(size_t)i].packet_id == id) return i;
        return -1;
    }

    // ---- operations ----
    void op_inject(const WmcRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        uint64_t pid = (uint64_t)(uint32_t)run.a0;
        int src = run.a1, dst = run.a2;
        int fc = run.a3;
        uint8_t pref = (uint8_t)(run.a4 != 0);

        bool invalid = false;
        if (src < 0 || src >= nodes || dst < 0 || dst >= nodes) invalid = true;
        if (fc <= 0 || fc > WMC_MAX_FLIT_COUNT) invalid = true;
        int existing = find_packet(pid);
        if (existing >= 0) {
            uint8_t st = ptable[(size_t)existing].state;
            if (st == WMC_PS_QUEUED || st == WMC_PS_ACTIVE) invalid = true;
        }
        if (invalid) {
            counters[18]++;  // invalid_count
            emit(WMC_EV_INVALID, UINT32_MAX, 255, 255, pid, UINT64_MAX, 0, op_index);
            return;
        }

        // find a slot: reuse existing terminal id slot, else a free slot.
        int slot = -1;
        if (existing >= 0) {
            slot = existing;  // terminal id re-used in place
        } else {
            for (int i = 0; i < max_packets; ++i) {
                WmcOPacket& q = ptable[(size_t)i];
                if (!q.used || q.state == WMC_PS_DONE || q.state == WMC_PS_DROPPED) {
                    slot = i; break;
                }
            }
        }
        bool table_full = (slot < 0);
        bool injq_full = ((int)injq[(size_t)src].size() >= max_injq);
        if (table_full || injq_full) {
            counters[1]++;  // inject_rejected
            emit(WMC_EV_INJECT_REJECT, (uint32_t)src, 255, 255, pid, UINT64_MAX, 0, op_index);
            return;
        }

        WmcOPacket& p = ptable[(size_t)slot];
        p = WmcOPacket();
        p.used = true;
        p.packet_id = pid;
        p.src = (uint32_t)src;
        p.dst = (uint32_t)dst;
        p.flit_count = (uint64_t)fc;
        p.prefer_adaptive = pref;
        p.injected_flits = 0;
        p.delivered_flits = 0;
        p.inject_seq = inject_seq_next++;
        p.state = WMC_PS_QUEUED;
        p.chosen_vc = UINT32_MAX;
        injq[(size_t)src].push_back(pid);
        counters[0]++;  // packets_accepted
        emit(WMC_EV_PACKET_ACCEPT, (uint32_t)src, 255, 255, pid, UINT64_MAX, (uint64_t)dst, op_index);
    }

    void op_set_link(const WmcRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        int node = run.a0, port = run.a1, up = run.a2;
        bool invalid = false;
        if (node < 0 || node >= nodes) invalid = true;
        else if (port == WMC_PORT_LOCAL || port < 0 || port >= WMC_PORT_COUNT) invalid = true;
        else if (!output_valid(node, port)) invalid = true;
        if (invalid) {
            counters[18]++;
            emit(WMC_EV_INVALID, (uint32_t)(node >= 0 ? node : -1), (uint8_t)port, 255,
                 UINT64_MAX, UINT64_MAX, 0, op_index);
            return;
        }
        link_up[(size_t)lidx(node, port)] = (uint8_t)(up != 0);
        counters[2]++;  // links_set
        emit(WMC_EV_LINK_SET, (uint32_t)node, (uint8_t)port, 255, UINT64_MAX,
             UINT64_MAX, (uint64_t)(up != 0), op_index);
    }

    void op_drop(const WmcRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        uint64_t pid = (uint64_t)(uint32_t)run.a0;
        int idx = find_packet(pid);
        if (idx < 0) {
            counters[18]++;
            emit(WMC_EV_INVALID, UINT32_MAX, 255, 255, pid, UINT64_MAX, 0, op_index);
            return;
        }
        uint8_t st = ptable[(size_t)idx].state;
        if (st == WMC_PS_DONE || st == WMC_PS_DROPPED) {
            counters[18]++;
            emit(WMC_EV_INVALID, UINT32_MAX, 255, 255, pid, UINT64_MAX, 0, op_index);
            return;
        }
        // remove not-yet-injected references from every injection queue.
        for (int n = 0; n < nodes; ++n) {
            std::vector<uint64_t>& q = injq[(size_t)n];
            std::vector<uint64_t> nq;
            for (uint64_t id : q) if (id != pid) nq.push_back(id);
            q.swap(nq);
        }
        ptable[(size_t)idx].state = WMC_PS_DROPPED;
        counters[3]++;  // packets_dropped
        emit(WMC_EV_PACKET_DROP, UINT32_MAX, 255, 255, pid, UINT64_MAX, 0, op_index);
    }

    // minimal output ports toward dst (reduce Manhattan distance), regardless
    // of link state. Used for adaptive routing candidate set.
    void minimal_ports(int node, int dst, int* out, int* count) const {
        int rc = rowof(node), cc = colof(node);
        int rd = rowof(dst), cd = colof(dst);
        int n = 0;
        // order by port ordinal for deterministic tie-break iteration: N,E,S,W
        if (rd < rc) out[n++] = WMC_PORT_N;
        if (cd > cc) out[n++] = WMC_PORT_E;
        if (rd > rc) out[n++] = WMC_PORT_S;
        if (cd < cc) out[n++] = WMC_PORT_W;
        *count = n;
    }

    int choose_vc_for_packet(const WmcOPacket& p) const {
        if (p.prefer_adaptive == 0) return 0;  // escape_vc
        int src = (int)p.src, dst = (int)p.dst;
        int ports[4]; int pc = 0;
        minimal_ports(src, dst, ports, &pc);
        int best_vc = -1; int64_t best_credit = -1;
        for (int v = 1; v < vc; ++v) {
            int64_t total = 0;
            for (int i = 0; i < pc; ++i) {
                int port = ports[i];
                if (link_up[(size_t)lidx(src, port)] == 0) continue;
                total += credit[(size_t)bidx(src, port, v)];
            }
            if (total > best_credit) { best_credit = total; best_vc = v; }
        }
        if (best_vc < 0 || best_credit <= 0) return 0;  // escape
        return best_vc;
    }

    void op_tick(const WmcRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        int cc = run.a0;
        if (cc <= 0) {
            counters[18]++;
            emit(WMC_EV_INVALID, UINT32_MAX, 255, 255, UINT64_MAX, UINT64_MAX, 0, op_index);
            return;
        }
        for (int step = 0; step < cc; ++step) {
            cycle_procedure(op_index);
        }
    }

    void cycle_procedure(uint32_t op_index) {
        // ---- Phase A: apply due credits ----
        std::sort(creditq.begin(), creditq.end(), creditq_less);
        std::vector<WmcOCreditRec> remain;
        remain.reserve(creditq.size());
        for (const WmcOCreditRec& rec : creditq) {
            if (rec.due_cycle <= cycle) {
                int ci = bidx((int)rec.node, (int)rec.port, (int)rec.vc);
                int64_t nv = credit[(size_t)ci] + (int64_t)rec.amount;
                if (nv > cap) {
                    credit[(size_t)ci] = cap;
                    counters[11]++;  // credit_clamps
                    emit(WMC_EV_CREDIT_CLAMP, rec.node, rec.port, rec.vc, UINT64_MAX,
                         UINT64_MAX, rec.amount, op_index);
                } else {
                    credit[(size_t)ci] = nv;
                    counters[10]++;  // credit_returns
                    emit(WMC_EV_CREDIT_RETURN, rec.node, rec.port, rec.vc, UINT64_MAX,
                         UINT64_MAX, rec.amount, op_index);
                }
            } else {
                remain.push_back(rec);
            }
        }
        creditq.swap(remain);

        // ---- Phase B: eject / drop local-front ----
        for (int n = 0; n < nodes; ++n) {
            for (int port = 0; port < WMC_PORT_COUNT; ++port) {
                for (int v = 0; v < vc; ++v) {
                    std::deque<WmcOFlit>& b = buf[(size_t)bidx(n, port, v)];
                    if (b.empty()) continue;
                    WmcOFlit f = b.front();
                    int pidx = find_packet(f.packet_id);
                    bool dropped = (pidx >= 0 && ptable[(size_t)pidx].state == WMC_PS_DROPPED);
                    if (dropped) {
                        b.pop_front();
                        schedule_credit_return(f, op_index);
                        counters[9]++;  // dropped_flits
                        emit(WMC_EV_DROPPED_FLIT, (uint32_t)n, (uint8_t)port, (uint8_t)v,
                             f.packet_id, f.flit_index, 0, op_index);
                    } else if ((int)f.dst == n) {
                        b.pop_front();
                        schedule_credit_return(f, op_index);
                        counters[7]++;  // flits_ejected
                        if (pidx >= 0) ptable[(size_t)pidx].delivered_flits++;
                        emit(WMC_EV_FLIT_EJECT, (uint32_t)n, (uint8_t)port, (uint8_t)v,
                             f.packet_id, f.flit_index, 0, op_index);
                        if (pidx >= 0 &&
                            ptable[(size_t)pidx].delivered_flits == ptable[(size_t)pidx].flit_count) {
                            ptable[(size_t)pidx].state = WMC_PS_DONE;
                            counters[8]++;  // packets_done
                            emit(WMC_EV_PACKET_DONE, (uint32_t)n, 255, 255,
                                 f.packet_id, UINT64_MAX, 0, op_index);
                        }
                    }
                }
            }
        }

        // ---- Phase C: inject ----
        for (int n = 0; n < nodes; ++n) {
            std::vector<uint64_t>& q = injq[(size_t)n];
            if (q.empty()) continue;
            uint64_t pid = q.front();
            int pidx = find_packet(pid);
            if (pidx < 0) { q.erase(q.begin()); continue; }
            WmcOPacket& p = ptable[(size_t)pidx];
            if (p.state == WMC_PS_DROPPED) { q.erase(q.begin()); continue; }
            if (p.chosen_vc == UINT32_MAX) {
                p.chosen_vc = (uint32_t)choose_vc_for_packet(p);
            }
            int v = (int)p.chosen_vc;
            std::deque<WmcOFlit>& b = buf[(size_t)bidx(n, WMC_PORT_LOCAL, v)];
            if ((int)b.size() >= cap) {
                counters[12]++;  // inject_stalls
                emit(WMC_EV_INJECT_STALL, (uint32_t)n, WMC_PORT_LOCAL, (uint8_t)v,
                     pid, UINT64_MAX, 0, op_index);
                continue;
            }
            WmcOFlit f;
            f.packet_id = pid;
            f.flit_index = p.injected_flits;
            f.flit_count = p.flit_count;
            f.src = p.src;
            f.dst = p.dst;
            f.vc = (uint8_t)v;
            if (p.flit_count == 1) f.kind = WMC_KIND_SINGLE;
            else if (p.injected_flits == 0) f.kind = WMC_KIND_HEAD;
            else if (p.injected_flits == p.flit_count - 1) f.kind = WMC_KIND_TAIL;
            else f.kind = WMC_KIND_BODY;
            f.inject_seq = p.inject_seq;
            f.entered_cycle = cycle;
            f.credit_return_node = UINT32_MAX;  // local-injected: no upstream
            f.credit_return_port = 255;
            b.push_back(f);
            uint64_t fidx = p.injected_flits;
            p.injected_flits++;
            if (p.state == WMC_PS_QUEUED) p.state = WMC_PS_ACTIVE;
            counters[4]++;  // flits_injected
            emit(WMC_EV_FLIT_INJECT, (uint32_t)n, WMC_PORT_LOCAL, (uint8_t)v,
                 pid, fidx, 0, op_index);
            if (p.injected_flits == p.flit_count) q.erase(q.begin());
        }

        // ---- Phase D: switch traversal ----
        // Track which output ports already used per router this cycle.
        std::vector<uint8_t> used((size_t)nodes * WMC_PORT_COUNT, 0);
        for (int n = 0; n < nodes; ++n) {
            for (int port = 0; port < WMC_PORT_COUNT; ++port) {
                for (int v = 0; v < vc; ++v) {
                    std::deque<WmcOFlit>& b = buf[(size_t)bidx(n, port, v)];
                    if (b.empty()) continue;
                    WmcOFlit f = b.front();
                    int pidx = find_packet(f.packet_id);
                    bool dropped = (pidx >= 0 && ptable[(size_t)pidx].state == WMC_PS_DROPPED);
                    if (dropped) {
                        b.pop_front();
                        schedule_credit_return(f, op_index);
                        counters[9]++;  // dropped_flits
                        emit(WMC_EV_DROPPED_FLIT, (uint32_t)n, (uint8_t)port, (uint8_t)v,
                             f.packet_id, f.flit_index, 0, op_index);
                        continue;
                    }
                    if ((int)f.dst == n) {
                        counters[16]++;  // eject_waits
                        emit(WMC_EV_EJECT_WAIT, (uint32_t)n, (uint8_t)port, (uint8_t)v,
                             f.packet_id, f.flit_index, 0, op_index);
                        continue;
                    }
                    int outp = -1;
                    if (v == 0) {
                        // escape XY
                        int rc = rowof(n), cco = colof(n);
                        int rd = rowof((int)f.dst), cd = colof((int)f.dst);
                        if (cd > cco) outp = WMC_PORT_E;
                        else if (cd < cco) outp = WMC_PORT_W;
                        else if (rd > rc) outp = WMC_PORT_S;
                        else if (rd < rc) outp = WMC_PORT_N;
                        if (outp < 0 || link_up[(size_t)lidx(n, outp)] == 0) {
                            route_stall(n, port, v, f, op_index);
                            continue;
                        }
                    } else {
                        // adaptive: minimal up ports, largest credit, tie lowest port
                        int ports[4]; int pc = 0;
                        minimal_ports(n, (int)f.dst, ports, &pc);
                        int chosen = -1; int64_t best = -1;
                        for (int i = 0; i < pc; ++i) {
                            int pp = ports[i];
                            if (link_up[(size_t)lidx(n, pp)] == 0) continue;
                            int64_t cr = credit[(size_t)bidx(n, pp, v)];
                            if (cr > best) { best = cr; chosen = pp; }
                        }
                        if (chosen < 0) {
                            route_stall(n, port, v, f, op_index);
                            continue;
                        }
                        outp = chosen;
                    }
                    if (used[(size_t)lidx(n, outp)]) {
                        counters[14]++;  // output_stalls
                        emit(WMC_EV_OUTPUT_STALL, (uint32_t)n, (uint8_t)outp, (uint8_t)v,
                             f.packet_id, f.flit_index, 0, op_index);
                        continue;
                    }
                    if (credit[(size_t)bidx(n, outp, v)] == 0) {
                        counters[15]++;  // credit_stalls
                        emit(WMC_EV_CREDIT_STALL, (uint32_t)n, (uint8_t)outp, (uint8_t)v,
                             f.packet_id, f.flit_index, 0, op_index);
                        continue;
                    }
                    int nb = neighbor(n, outp);
                    int oppp = opposite(outp);
                    std::deque<WmcOFlit>& db = buf[(size_t)bidx(nb, oppp, v)];
                    if ((int)db.size() >= cap) {
                        counters[17]++;  // credit_desyncs
                        emit(WMC_EV_CREDIT_DESYNC, (uint32_t)n, (uint8_t)outp, (uint8_t)v,
                             f.packet_id, f.flit_index, 0, op_index);
                        continue;
                    }
                    // move
                    credit[(size_t)bidx(n, outp, v)] -= 1;
                    WmcOFlit moved = f;
                    b.pop_front();
                    schedule_credit_return(f, op_index);
                    moved.credit_return_node = (uint32_t)n;
                    moved.credit_return_port = (uint8_t)outp;
                    moved.entered_cycle = cycle;
                    db.push_back(moved);
                    used[(size_t)lidx(n, outp)] = 1;
                    if (v == 0) counters[5]++;  // flits_moved_escape
                    else counters[6]++;         // flits_moved_adaptive
                    emit(WMC_EV_FLIT_MOVE, (uint32_t)n, (uint8_t)outp, (uint8_t)v,
                         f.packet_id, f.flit_index, (uint64_t)nb, op_index);
                }
            }
        }

        cycle += 1;
    }

    void route_stall(int n, int port, int v, const WmcOFlit& f, uint32_t op_index) {
        counters[13]++;  // route_stalls (index 13)
        emit(WMC_EV_ROUTE_STALL, (uint32_t)n, (uint8_t)port, (uint8_t)v,
             f.packet_id, f.flit_index, 0, op_index);
    }

    // ---- hashing ----
    uint64_t hash_packets() const {
        uint64_t h = 1469598103934665603ULL;
        // nonterminal packets by packet_id ascending
        std::vector<int> idxs;
        for (int i = 0; i < max_packets; ++i) {
            if (ptable[(size_t)i].used &&
                (ptable[(size_t)i].state == WMC_PS_QUEUED ||
                 ptable[(size_t)i].state == WMC_PS_ACTIVE)) {
                idxs.push_back(i);
            }
        }
        std::sort(idxs.begin(), idxs.end(), [&](int a, int b) {
            return ptable[(size_t)a].packet_id < ptable[(size_t)b].packet_id;
        });
        for (int i : idxs) {
            const WmcOPacket& p = ptable[(size_t)i];
            uint64_t pid = p.packet_id;
            wmc_o_fnv(&h, &pid, 8);
            wmc_o_fnv(&h, &p.src, 4);
            wmc_o_fnv(&h, &p.dst, 4);
            wmc_o_fnv(&h, &p.flit_count, 8);
            wmc_o_fnv(&h, &p.prefer_adaptive, 1);
            wmc_o_fnv(&h, &p.injected_flits, 8);
            wmc_o_fnv(&h, &p.delivered_flits, 8);
            wmc_o_fnv(&h, &p.inject_seq, 8);
            wmc_o_fnv(&h, &p.state, 1);
            uint32_t cv = p.chosen_vc;
            wmc_o_fnv(&h, &cv, 4);
        }
        return h;
    }

    uint64_t hash_buffers() const {
        uint64_t h = 1469598103934665603ULL;
        for (int n = 0; n < nodes; ++n) {
            for (int port = 0; port < WMC_PORT_COUNT; ++port) {
                for (int v = 0; v < vc; ++v) {
                    const std::deque<WmcOFlit>& b = buf[(size_t)bidx(n, port, v)];
                    uint64_t pos = 0;
                    for (const WmcOFlit& f : b) {
                        uint32_t nn = (uint32_t)n; uint8_t pp = (uint8_t)port; uint8_t vv = (uint8_t)v;
                        wmc_o_fnv(&h, &nn, 4);
                        wmc_o_fnv(&h, &pp, 1);
                        wmc_o_fnv(&h, &vv, 1);
                        wmc_o_fnv(&h, &pos, 8);
                        wmc_o_fnv(&h, &f.packet_id, 8);
                        wmc_o_fnv(&h, &f.flit_index, 8);
                        wmc_o_fnv(&h, &f.flit_count, 8);
                        wmc_o_fnv(&h, &f.src, 4);
                        wmc_o_fnv(&h, &f.dst, 4);
                        wmc_o_fnv(&h, &f.kind, 1);
                        wmc_o_fnv(&h, &f.inject_seq, 8);
                        wmc_o_fnv(&h, &f.entered_cycle, 8);
                        pos++;
                    }
                }
            }
        }
        return h;
    }

    uint64_t hash_credits() const {
        uint64_t h = 1469598103934665603ULL;
        for (int n = 0; n < nodes; ++n) {
            for (int port = 0; port < WMC_PORT_COUNT; ++port) {
                for (int v = 0; v < vc; ++v) {
                    uint32_t nn = (uint32_t)n; uint8_t pp = (uint8_t)port; uint8_t vv = (uint8_t)v;
                    uint64_t cr = (uint64_t)credit[(size_t)bidx(n, port, v)];
                    uint8_t lu = link_up[(size_t)lidx(n, port)];
                    wmc_o_fnv(&h, &nn, 4);
                    wmc_o_fnv(&h, &pp, 1);
                    wmc_o_fnv(&h, &vv, 1);
                    wmc_o_fnv(&h, &cr, 8);
                    wmc_o_fnv(&h, &lu, 1);
                }
            }
        }
        return h;
    }

    uint64_t hash_credit_queue() const {
        uint64_t h = 1469598103934665603ULL;
        std::vector<WmcOCreditRec> sorted = creditq;
        std::sort(sorted.begin(), sorted.end(), creditq_less);
        for (const WmcOCreditRec& r : sorted) {
            wmc_o_fnv(&h, &r.due_cycle, 8);
            wmc_o_fnv(&h, &r.node, 4);
            wmc_o_fnv(&h, &r.port, 1);
            wmc_o_fnv(&h, &r.vc, 1);
            wmc_o_fnv(&h, &r.amount, 8);
        }
        return h;
    }

    void step_once(const WmcRunSpec& run, WmcOracleResult* out) {
        run_event_hash = 1469598103934665603ULL;
        switch (run.op) {
            case WMC_OP_INJECT: op_inject(run); break;
            case WMC_OP_SET_LINK: op_set_link(run); break;
            case WMC_OP_DROP: op_drop(run); break;
            case WMC_OP_TICK: op_tick(run); break;
            default:
                counters[18]++;
                emit(WMC_EV_INVALID, UINT32_MAX, 255, 255, UINT64_MAX, UINT64_MAX, 0,
                     (uint32_t)run.op_index);
                break;
        }
        for (int i = 0; i < WMC_COUNTER_COUNT; ++i) out->counters[i] = counters[i];
        out->fabric_event_hash = run_event_hash;
        out->packet_hash = hash_packets();
        out->buffer_hash = hash_buffers();
        out->credit_hash = hash_credits();
        out->credit_queue_hash = hash_credit_queue();
        out->cycle_out = cycle;
        out->event_seq_out = event_seq;
    }
};

static inline bool wmc_check(const WmcOracleResult& exp, const WmcOracleResult& got,
                             std::string* err) {
    static const char* names[WMC_COUNTER_COUNT] = {
        "packets_accepted", "inject_rejected", "links_set", "packets_dropped",
        "flits_injected", "flits_moved_escape", "flits_moved_adaptive",
        "flits_ejected", "packets_done", "dropped_flits", "credit_returns",
        "credit_clamps", "inject_stalls", "route_stalls", "output_stalls",
        "credit_stalls", "eject_waits", "credit_desyncs", "invalid_count"};
    for (int i = 0; i < WMC_COUNTER_COUNT; ++i) {
        if (exp.counters[i] != got.counters[i]) {
            if (err) {
                std::ostringstream o;
                o << "counter " << names[i] << " got " << got.counters[i]
                  << " expected " << exp.counters[i];
                *err = o.str();
            }
            return false;
        }
    }
#define WMC_CK(field) \
    if (exp.field != got.field) { \
        if (err) { std::ostringstream o; o << #field << " got 0x" << std::hex \
            << got.field << " expected 0x" << exp.field; *err = o.str(); } \
        return false; }
    WMC_CK(fabric_event_hash);
    WMC_CK(packet_hash);
    WMC_CK(buffer_hash);
    WMC_CK(credit_hash);
    WMC_CK(credit_queue_hash);
    WMC_CK(cycle_out);
    WMC_CK(event_seq_out);
#undef WMC_CK
    return true;
}

#endif  // WORMHOLE_MESH_CREDITS_ORACLE_HPP_
