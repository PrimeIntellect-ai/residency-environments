// ============================================================================
// file: mk_priority_preempt_oracle.hpp
// Independent CPU oracle (stateful) + output check helpers.
// ============================================================================

#ifndef MK_PRIORITY_PREEMPT_ORACLE_HPP_
#define MK_PRIORITY_PREEMPT_ORACLE_HPP_

#include "mk_priority_preempt_common.h"

#include <stdint.h>
#include <stddef.h>

#include <sstream>
#include <string>
#include <vector>

struct MkpExpected {
    int32_t trace_len;
    std::vector<uint64_t> trace;
    std::vector<uint64_t> exec_digest;
    std::vector<uint64_t> y;          // [M]
    std::vector<int32_t> queue_len;   // [Q]
    std::vector<uint64_t> queue_dump; // [total live]
    uint64_t state_checksum;
};

static inline uint64_t mkp_oracle_fnv_byte(uint64_t h, uint8_t b) {
    return (h ^ static_cast<uint64_t>(b)) * MKP_FNV_PRIME;
}

static inline uint64_t mkp_oracle_fnv_u32(uint64_t h, uint32_t v) {
    for (int m = 0; m < 4; ++m) {
        h = mkp_oracle_fnv_byte(h, (uint8_t)((v >> (8 * m)) & 0xFF));
    }
    return h;
}

static inline uint64_t mkp_oracle_fnv_u64(uint64_t h, uint64_t v) {
    for (int m = 0; m < 8; ++m) {
        h = mkp_oracle_fnv_byte(h, (uint8_t)((v >> (8 * m)) & 0xFF));
    }
    return h;
}

struct MkpOracleJob {
    uint32_t gjid;
    int cls;
    int rem;
    int arrival;  // round; 0 for carryover
    std::vector<int8_t> w;
};

// Stateful CPU oracle mirroring the persistent job table.
struct MkpOracle {
    int K = 0;
    int Q = 0;
    uint32_t job_counter = 0;
    std::vector<MkpOracleJob> jobs;  // live + dead (rem == 0) kept; gjid asc

    void reset(int K_, int Q_) {
        K = K_;
        Q = Q_;
        job_counter = 0;
        jobs.clear();
    }

    void run_round(
        const MkpRunSpec& run,
        const int8_t* X,
        const int32_t* jarrival,
        const int32_t* jclass,
        const int32_t* jlen,
        const int8_t* jw,
        MkpExpected* expected) {
        const int M = run.M;
        const int B = M / MKP_ROWS;
        const int R = run.R;
        const int quantum = run.quantum;

        // Ingest.
        for (int j = 0; j < run.njobs; ++j) {
            MkpOracleJob job;
            job.gjid = job_counter + (uint32_t)j;
            job.cls = jclass[j];
            job.rem = jlen[j];
            job.arrival = jarrival[j];
            job.w.assign(jw + (size_t)j * K, jw + (size_t)(j + 1) * K);
            jobs.push_back(job);
        }
        job_counter += (uint32_t)run.njobs;

        expected->y.assign((size_t)M, 0ULL);
        expected->trace.clear();
        expected->exec_digest.clear();

        int g = 0;
        for (int r = 0; r < R; ++r) {
            // Active class.
            int c = -1;
            for (const MkpOracleJob& job : jobs) {
                if (job.rem > 0 && job.arrival <= r) {
                    if (c < 0 || job.cls < c) c = job.cls;
                }
            }
            if (c < 0) continue;

            // Execution list: pending jobs of class c, gjid ascending
            // (jobs vector is already gjid ascending).
            for (MkpOracleJob& job : jobs) {
                if (job.rem <= 0 || job.arrival > r || job.cls != c) {
                    continue;
                }
                const int n_s = quantum < job.rem ? quantum : job.rem;
                uint64_t ed = MKP_FNV_BASIS;
                for (int s = 0; s < n_s; ++s) {
                    const uint32_t salt =
                        MKP_SALT_ROUND * (uint32_t)r +
                        MKP_SALT_TRACE * (uint32_t)g +
                        MKP_SALT_JOB * job.gjid +
                        MKP_SALT_SLICE * (uint32_t)s;
                    uint8_t sb[4];
                    for (int m = 0; m < 4; ++m) {
                        sb[m] = (uint8_t)((salt >> (8 * m)) & 0xFF);
                    }
                    const int b = (int)(((uint32_t)g + (uint32_t)s) %
                                        (uint32_t)B);
                    uint32_t res[MKP_ROWS];
                    for (int i = 0; i < MKP_ROWS; ++i) {
                        const int row = b * MKP_ROWS + i;
                        const int8_t* xrow = X + (size_t)row * K;
                        int32_t acc = 0;
                        for (int k = 0; k < K; ++k) {
                            const int8_t mixed = (int8_t)(
                                (uint8_t)xrow[k] ^ sb[k & 3]);
                            acc += (int32_t)mixed * (int32_t)job.w[(size_t)k];
                        }
                        res[i] = (uint32_t)acc;
                        expected->y[(size_t)row] += (uint64_t)res[i];
                    }
                    uint64_t sd = MKP_FNV_BASIS;
                    for (int t = 0; t < 16; ++t) {
                        uint64_t sub = MKP_FNV_BASIS;
                        for (int i = 16 * t; i < 16 * t + 16; ++i) {
                            sub = mkp_oracle_fnv_u32(sub, res[i]);
                        }
                        sd = mkp_oracle_fnv_u64(sd, sub);
                    }
                    ed = mkp_oracle_fnv_u64(ed, sd);
                }
                expected->trace.push_back(mkp_trace_word(
                    job.gjid, r, c, n_s, (uint32_t)(job.rem - n_s)));
                expected->exec_digest.push_back(ed);
                job.rem -= n_s;
                ++g;
            }
        }
        expected->trace_len = g;

        // Carryover.
        for (MkpOracleJob& job : jobs) job.arrival = 0;

        // Queue outputs + checksum.
        expected->queue_len.assign((size_t)Q, 0);
        expected->queue_dump.clear();
        uint64_t root = MKP_FNV_BASIS;
        for (int c = 0; c < Q; ++c) {
            uint32_t cnt = 0;
            for (const MkpOracleJob& job : jobs) {
                if (job.rem > 0 && job.cls == c) ++cnt;
            }
            expected->queue_len[(size_t)c] = (int32_t)cnt;
            uint64_t cd = MKP_FNV_BASIS;
            cd = mkp_oracle_fnv_u32(cd, cnt);
            for (const MkpOracleJob& job : jobs) {
                if (job.rem <= 0 || job.cls != c) continue;
                expected->queue_dump.push_back(mkp_dump_word(
                    job.gjid, c, (uint32_t)job.rem));
                uint64_t jd = MKP_FNV_BASIS;
                jd = mkp_oracle_fnv_u32(jd, job.gjid);
                jd = mkp_oracle_fnv_u32(jd, (uint32_t)job.cls);
                jd = mkp_oracle_fnv_u32(jd, (uint32_t)job.rem);
                for (int k = 0; k < K; ++k) {
                    jd = mkp_oracle_fnv_byte(jd, (uint8_t)job.w[(size_t)k]);
                }
                cd = mkp_oracle_fnv_u64(cd, jd);
            }
            root = mkp_oracle_fnv_u64(root, cd);
        }
        expected->state_checksum = root;
    }
};

struct MkpHostOutputsView {
    const int32_t* trace_len;
    const uint64_t* trace;
    const uint64_t* exec_digest;
    const uint64_t* y;
    const int32_t* queue_len;
    const uint64_t* queue_dump;
    const uint64_t* state_checksum;
};

static inline bool mkp_check_outputs(
    const MkpRunSpec& run,
    const MkpExpected& expected,
    const MkpHostOutputsView& got,
    std::string* error) {
    if (got.trace_len[0] != expected.trace_len) {
        if (error) {
            std::ostringstream oss;
            oss << "trace_len mismatch: got " << got.trace_len[0]
                << ", expected " << expected.trace_len;
            *error = oss.str();
        }
        return false;
    }
    for (int i = 0; i < expected.trace_len; ++i) {
        if (got.trace[i] != expected.trace[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "trace mismatch at index " << i << ": got 0x"
                    << std::hex << got.trace[i] << ", expected 0x"
                    << expected.trace[(size_t)i];
                *error = oss.str();
            }
            return false;
        }
        if (got.exec_digest[i] != expected.exec_digest[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "exec_digest mismatch at index " << i;
                *error = oss.str();
            }
            return false;
        }
    }
    for (int row = 0; row < run.M; ++row) {
        if (got.y[row] != expected.y[(size_t)row]) {
            if (error) {
                std::ostringstream oss;
                oss << "y mismatch at row " << row << ": got "
                    << got.y[row] << ", expected " << expected.y[(size_t)row];
                *error = oss.str();
            }
            return false;
        }
    }
    int total_live = 0;
    for (int c = 0; c < run.Q; ++c) {
        if (got.queue_len[c] != expected.queue_len[(size_t)c]) {
            if (error) {
                std::ostringstream oss;
                oss << "queue_len mismatch at class " << c << ": got "
                    << got.queue_len[c] << ", expected "
                    << expected.queue_len[(size_t)c];
                *error = oss.str();
            }
            return false;
        }
        total_live += expected.queue_len[(size_t)c];
    }
    for (int i = 0; i < total_live; ++i) {
        if (got.queue_dump[i] != expected.queue_dump[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "queue_dump mismatch at index " << i << ": got 0x"
                    << std::hex << got.queue_dump[i] << ", expected 0x"
                    << expected.queue_dump[(size_t)i];
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
  1. trace_len + trace prefix   (the full preemptive schedule: active
                                 class per round, execution order,
                                 quantum pacing, remaining counters).
  2. exec_digest prefix         (per-execution slice arithmetic under the
                                 schedule-coupled salts and data blocks).
  3. y                          (commutative wraparound accumulation).
  4. queue_len / queue_dump     (final multi-queue state).
  5. state_checksum             (three-level FNV over live jobs incl. the
                                 STORED weight rows; any earlier-run state
                                 error cascades).

The grader should additionally enforce guard bytes, input immutability,
determinism replays, multi-run carryover chains with fresh X/weights per
run, and resets mid-case.
*/

#endif  // MK_PRIORITY_PREEMPT_ORACLE_HPP_
