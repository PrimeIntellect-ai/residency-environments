// PMPP_CANARY_44_44c1a6817b -- held-out canary; MUST NOT appear in any submission
// file: lsm_wal_compaction_reference.cu
//
// Reference implementation. Single-threaded device model. SoA-ish device
// state with explicit fixed-capacity arrays and insertion/selection sorts.
// This implementation is algorithmically independent from both the host oracle
// and the naive implementation; only the observable byte streams match.

#include "lsm_wal_compaction_common.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>

#define R_U64_MAX 0xFFFFFFFFFFFFFFFFULL
#define R_FNV_BASE 1469598103934665603ULL

// ---------------- device state ----------------
struct RefDevState {
    LsmProblemSpec spec;

    // scalars
    uint64_t* sc;   // [0]=next_seq [1]=next_file_id [2]=next_wal_id [3]=event_seq [4]=durable_flush_seq
    // hashes (persistent streaming)
    uint64_t* hsh;  // [0]=read [1]=write [2]=compaction
    // counts
    int64_t* counts;  // 24

    // WAL: segment-major. seg_count, seg_wal_id[S], seg_reccount[S]
    int32_t* wal_seg_count;
    uint64_t* wal_seg_id;       // [LSM_MAX_WAL_SEGMENTS]
    int32_t* wal_seg_reccount;  // [LSM_MAX_WAL_SEGMENTS]
    // records stored per segment slot, contiguous: seg s, slot j ->
    // index s*cap_rec_per_seg + j ; cap = LSM_MAX_WAL_RECORDS/segments? use a
    // flat pool with explicit (wal_id, offset, seq, kind, key, value).
    uint64_t* wr_walid;   // [LSM_MAX_WAL_RECORDS]
    uint64_t* wr_offset;
    uint64_t* wr_seq;
    int32_t*  wr_kind;
    uint64_t* wr_key;
    int64_t*  wr_value;
    int32_t*  wr_seg;     // which segment slot this record belongs to
    int32_t*  wr_count;   // total live wal records

    // memtable (append order)
    int32_t*  mt_count;
    uint64_t* mt_seq;     // [LSM_MAX_MEMTABLE]
    int32_t*  mt_kind;
    uint64_t* mt_key;
    int64_t*  mt_value;

    // files: flat pool. file f stored in slot f. file_used[f]
    int32_t*  file_count;        // number of allocated file slots (compacted dense)
    int32_t*  f_used;            // [LSM_MAX_TOTAL_FILES]
    uint64_t* f_file_id;
    int32_t*  f_level;
    uint64_t* f_min_key;
    uint64_t* f_max_key;
    uint64_t* f_max_seq;
    uint64_t* f_create_event;
    int32_t*  f_entry_count;
    // entries: pool entry e -> file slot = e / LSM_MAX_ENTRIES_PER_FILE,
    // local index = e % cap. We store entries per file slot region.
    uint64_t* e_key;    // [LSM_MAX_TOTAL_FILES * LSM_MAX_ENTRIES_PER_FILE]
    uint64_t* e_seq;
    int32_t*  e_kind;
    int64_t*  e_value;

    // snapshots
    int32_t*  snap_count;
    uint64_t* snap_id;     // [LSM_MAX_SNAPSHOTS]
    uint64_t* snap_seq;

    // scratch / workspace pointers filled at run
};

// ---------------- fnv ----------------
__device__ __forceinline__ uint64_t r_fnv_b(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= 1099511628211ULL; return h;
}
__device__ void r_fnv_u64(uint64_t* h, uint64_t v) {
    uint64_t hh = *h;
    for (int i = 0; i < 8; ++i) { hh = r_fnv_b(hh, (uint8_t)(v & 0xFF)); v >>= 8; }
    *h = hh;
}
__device__ void r_fnv_u32(uint64_t* h, uint32_t v) {
    uint64_t hh = *h;
    for (int i = 0; i < 4; ++i) { hh = r_fnv_b(hh, (uint8_t)(v & 0xFF)); v >>= 8; }
    *h = hh;
}
__device__ void r_fnv_u8(uint64_t* h, uint8_t v) { *h = r_fnv_b(*h, v); }
__device__ void r_fnv_i64(uint64_t* h, int64_t v) { r_fnv_u64(h, (uint64_t)v); }

// ---------------- entry pool helpers ----------------
__device__ __forceinline__ size_t r_ebase(int file_slot) {
    return (size_t)file_slot * (size_t)LSM_MAX_ENTRIES_PER_FILE;
}

// ---------------- reset ----------------
__global__ void ref_reset_kernel(RefDevState st) {
    if (blockIdx.x || threadIdx.x) return;
    st.sc[0] = 1;  // next_seq
    st.sc[1] = 1;  // next_file_id
    st.sc[2] = 1;  // next_wal_id
    st.sc[3] = 0;  // event_seq
    st.sc[4] = 0;  // durable_flush_seq
    st.hsh[0] = R_FNV_BASE;
    st.hsh[1] = R_FNV_BASE;
    st.hsh[2] = R_FNV_BASE;
    for (int i = 0; i < 24; ++i) st.counts[i] = 0;

    st.wal_seg_count[0] = 1;
    st.wal_seg_id[0] = 1;
    st.wal_seg_reccount[0] = 0;
    st.wr_count[0] = 0;
    st.mt_count[0] = 0;
    st.file_count[0] = 0;
    for (int i = 0; i < LSM_MAX_TOTAL_FILES; ++i) st.f_used[i] = 0;
    st.snap_count[0] = 0;
}

// counts indices
enum {
    C_put_ok=0, C_del_ok, C_write_stall, C_write_oom, C_wal_rolls,
    C_get_found, C_get_missing, C_snapshot_opened, C_snapshot_released,
    C_snapshots_dropped_by_crash, C_flush_files, C_flush_empty, C_flush_oom,
    C_compact_empty, C_compact_oom, C_compact_input_files, C_compact_output_files,
    C_versions_dropped, C_tombstones_dropped, C_obsolete_files, C_wal_archived,
    C_recovered_records, C_recover_stalled_records, C_invalid_count
};

__device__ uint64_t r_next_event(RefDevState& st) { return ++st.sc[3]; }

__device__ void r_emit_write(RefDevState& st, int ev, uint64_t seq_or_max,
                             uint64_t wal_id, uint64_t off_or_max,
                             uint64_t key_or_max, int kind_or_255,
                             int64_t value, uint32_t op_index) {
    uint64_t es = r_next_event(st);
    uint64_t* h = &st.hsh[1];
    r_fnv_u8(h, (uint8_t)ev);
    r_fnv_u64(h, es);
    r_fnv_u32(h, op_index);
    r_fnv_u64(h, seq_or_max);
    r_fnv_u64(h, wal_id);
    r_fnv_u64(h, off_or_max);
    r_fnv_u64(h, key_or_max);
    r_fnv_u8(h, (uint8_t)kind_or_255);
    r_fnv_i64(h, value);
}

__device__ void r_emit_read(RefDevState& st, uint64_t read_id, uint32_t op_index,
                            uint64_t key, uint64_t sid, uint64_t read_seq,
                            int found, int64_t value, int src_kind,
                            uint64_t src_file, uint64_t src_seq) {
    uint64_t* h = &st.hsh[0];
    r_fnv_u64(h, read_id);
    r_fnv_u32(h, op_index);
    r_fnv_u64(h, key);
    r_fnv_u64(h, sid);
    r_fnv_u64(h, read_seq);
    r_fnv_u8(h, (uint8_t)found);
    r_fnv_i64(h, value);
    r_fnv_u8(h, (uint8_t)src_kind);
    r_fnv_u64(h, src_file);
    r_fnv_u64(h, src_seq);
}

__device__ void r_emit_cfile(RefDevState& st, int ev, uint32_t op_index, int32_t level,
                             uint64_t file_id, uint64_t mn, uint64_t mx,
                             uint64_t ecount, uint64_t maxseq, uint64_t* out_es) {
    uint64_t es = r_next_event(st);
    if (out_es) *out_es = es;
    uint64_t* h = &st.hsh[2];
    r_fnv_u8(h, (uint8_t)ev);
    r_fnv_u64(h, es);
    r_fnv_u32(h, op_index);
    r_fnv_u32(h, (uint32_t)level);
    r_fnv_u64(h, file_id);
    r_fnv_u64(h, mn);
    r_fnv_u64(h, mx);
    r_fnv_u64(h, ecount);
    r_fnv_u64(h, maxseq);
}

__device__ void r_emit_cver(RefDevState& st, int ev, uint32_t op_index, uint64_t key,
                            uint64_t seq, int kind, int64_t value) {
    uint64_t es = r_next_event(st);
    uint64_t* h = &st.hsh[2];
    r_fnv_u8(h, (uint8_t)ev);
    r_fnv_u64(h, es);
    r_fnv_u32(h, op_index);
    r_fnv_u64(h, key);
    r_fnv_u64(h, seq);
    r_fnv_u8(h, (uint8_t)kind);
    r_fnv_i64(h, value);
}

__device__ int r_find_snap(RefDevState& st, uint64_t sid) {
    int n = st.snap_count[0];
    for (int i = 0; i < n; ++i) if (st.snap_id[i] == sid) return i;
    return -1;
}

// current segment = largest wal_id = last slot (we keep slots in wal_id order).
__device__ int r_cur_seg(RefDevState& st) {
    int n = st.wal_seg_count[0];
    return n - 1;
}

// ---------------- write ----------------
__device__ void r_do_write(RefDevState& st, int kind, uint64_t key, int64_t value, uint32_t op_index) {
    int kind255 = (kind == LSM_KIND_DEL) ? LSM_KIND_DEL : LSM_KIND_PUT;
    if (st.mt_count[0] >= st.spec.memtable_record_cap) {
        st.counts[C_write_stall] += 1;
        r_emit_write(st, LSM_EV_WRITE_STALL, R_U64_MAX, R_U64_MAX, R_U64_MAX, key, kind255, value, op_index);
        return;
    }
    int seg = r_cur_seg(st);
    if (st.wal_seg_reccount[seg] >= st.spec.wal_segment_record_cap) {
        if (st.wal_seg_count[0] >= st.spec.max_wal_segments) {
            st.counts[C_write_oom] += 1;
            r_emit_write(st, LSM_EV_WRITE_OOM, R_U64_MAX, R_U64_MAX, R_U64_MAX, key, kind255, value, op_index);
            return;
        }
        int ns = st.wal_seg_count[0];
        uint64_t nid = ++st.sc[2];
        st.wal_seg_id[ns] = nid;
        st.wal_seg_reccount[ns] = 0;
        st.wal_seg_count[0] = ns + 1;
        seg = ns;
        st.counts[C_wal_rolls] += 1;
        r_emit_write(st, LSM_EV_WAL_ROLL, R_U64_MAX, nid, R_U64_MAX, R_U64_MAX, 255, 0, op_index);
    }
    uint64_t seq = st.sc[0]++;
    int64_t storeval = (kind == LSM_KIND_DEL) ? 0 : value;
    int rc = st.wr_count[0];
    st.wr_walid[rc] = st.wal_seg_id[seg];
    st.wr_offset[rc] = (uint64_t)st.wal_seg_reccount[seg];
    st.wr_seq[rc] = seq;
    st.wr_kind[rc] = kind;
    st.wr_key[rc] = key;
    st.wr_value[rc] = storeval;
    st.wr_seg[rc] = seg;
    st.wr_count[0] = rc + 1;
    st.wal_seg_reccount[seg] += 1;

    int mc = st.mt_count[0];
    st.mt_seq[mc] = seq;
    st.mt_kind[mc] = kind;
    st.mt_key[mc] = key;
    st.mt_value[mc] = storeval;
    st.mt_count[0] = mc + 1;

    if (kind == LSM_KIND_DEL) {
        st.counts[C_del_ok] += 1;
        r_emit_write(st, LSM_EV_DEL_OK, seq, st.wal_seg_id[seg], (uint64_t)(st.wal_seg_reccount[seg]-1), key, LSM_KIND_DEL, storeval, op_index);
    } else {
        st.counts[C_put_ok] += 1;
        r_emit_write(st, LSM_EV_PUT_OK, seq, st.wal_seg_id[seg], (uint64_t)(st.wal_seg_reccount[seg]-1), key, LSM_KIND_PUT, storeval, op_index);
    }
}

// ---------------- get ----------------
__device__ void r_do_get(RefDevState& st, uint64_t read_id, uint64_t key, uint64_t sid, uint32_t op_index) {
    uint64_t read_seq;
    if (sid == 0) {
        read_seq = st.sc[0] - 1;
    } else {
        int si = r_find_snap(st, sid);
        if (si < 0) { st.counts[C_invalid_count] += 1; return; }
        read_seq = st.snap_seq[si];
    }

    int found = 0;
    int result_found = 0;
    int64_t result_value = 0;
    int src_kind = LSM_SRC_NONE;
    uint64_t src_file = R_U64_MAX, src_seq = R_U64_MAX;

    // memtable desc seq (reverse append).
    for (int i = st.mt_count[0] - 1; i >= 0 && !found; --i) {
        if (st.mt_key[i] == key && st.mt_seq[i] <= read_seq) {
            found = 1; src_kind = LSM_SRC_MEMTABLE; src_file = R_U64_MAX;
            src_seq = st.mt_seq[i];
            result_found = (st.mt_kind[i] == LSM_KIND_PUT) ? 1 : 0;
            result_value = (st.mt_kind[i] == LSM_KIND_PUT) ? st.mt_value[i] : 0;
        }
    }

    // L0 by descending file_id. Files at level 0; iterate by repeatedly finding
    // the max file_id not yet visited.
    if (!found) {
        // gather L0 slots
        // selection over file_id descending.
        // simple O(n^2): repeatedly pick largest unvisited L0 file_id.
        int nfiles = st.file_count[0];
        // use a visited mask in shared? just use linear scans with threshold.
        uint64_t last_id = R_U64_MAX;  // means "no upper bound yet"
        bool have_last = false;
        while (!found) {
            // find largest file_id among L0 files strictly less than last_id (or any if !have_last)
            int best = -1; uint64_t best_id = 0;
            for (int f = 0; f < nfiles; ++f) {
                if (!st.f_used[f] || st.f_level[f] != 0) continue;
                uint64_t fid = st.f_file_id[f];
                if (have_last && fid >= last_id) continue;
                if (best < 0 || fid > best_id) { best = f; best_id = fid; }
            }
            if (best < 0) break;
            have_last = true; last_id = best_id;
            // scan entries for key, desc seq (entries sorted key asc seq desc).
            size_t base = r_ebase(best);
            int ec = st.f_entry_count[best];
            for (int e = 0; e < ec && !found; ++e) {
                if (st.e_key[base + e] != key) continue;
                if (st.e_seq[base + e] <= read_seq) {
                    found = 1; src_kind = LSM_SRC_L0_FILE; src_file = st.f_file_id[best];
                    src_seq = st.e_seq[base + e];
                    int kd = st.e_kind[base + e];
                    result_found = (kd == LSM_KIND_PUT) ? 1 : 0;
                    result_value = (kd == LSM_KIND_PUT) ? st.e_value[base + e] : 0;
                }
            }
        }
    }

    // levels 1..L-1 ascending; unique covering file.
    for (int lvl = 1; lvl < st.spec.level_count && !found; ++lvl) {
        int cover = -1;
        int nfiles = st.file_count[0];
        for (int f = 0; f < nfiles; ++f) {
            if (!st.f_used[f] || st.f_level[f] != lvl) continue;
            if (key >= st.f_min_key[f] && key <= st.f_max_key[f]) { cover = f; break; }
        }
        if (cover < 0) continue;
        size_t base = r_ebase(cover);
        int ec = st.f_entry_count[cover];
        for (int e = 0; e < ec && !found; ++e) {
            if (st.e_key[base + e] != key) continue;
            if (st.e_seq[base + e] <= read_seq) {
                found = 1; src_kind = LSM_SRC_LEVEL_FILE; src_file = st.f_file_id[cover];
                src_seq = st.e_seq[base + e];
                int kd = st.e_kind[base + e];
                result_found = (kd == LSM_KIND_PUT) ? 1 : 0;
                result_value = (kd == LSM_KIND_PUT) ? st.e_value[base + e] : 0;
            }
        }
    }

    if (result_found) st.counts[C_get_found] += 1; else st.counts[C_get_missing] += 1;
    r_emit_read(st, read_id, op_index, key, sid, read_seq, result_found, result_value, src_kind, src_file, src_seq);
}

// ---------------- snapshots ----------------
__device__ void r_do_open_snap(RefDevState& st, uint64_t sid) {
    if (sid == 0 || r_find_snap(st, sid) >= 0 || st.snap_count[0] >= st.spec.max_snapshots) {
        st.counts[C_invalid_count] += 1; return;
    }
    int n = st.snap_count[0];
    st.snap_id[n] = sid;
    st.snap_seq[n] = st.sc[0] - 1;
    st.snap_count[0] = n + 1;
    // keep sorted by id (insertion sort last element).
    for (int i = n; i > 0; --i) {
        if (st.snap_id[i] < st.snap_id[i-1]) {
            uint64_t ti = st.snap_id[i]; st.snap_id[i] = st.snap_id[i-1]; st.snap_id[i-1] = ti;
            uint64_t ts = st.snap_seq[i]; st.snap_seq[i] = st.snap_seq[i-1]; st.snap_seq[i-1] = ts;
        } else break;
    }
    st.counts[C_snapshot_opened] += 1;
}

__device__ void r_do_release_snap(RefDevState& st, uint64_t sid) {
    int si = r_find_snap(st, sid);
    if (si < 0) { st.counts[C_invalid_count] += 1; return; }
    int n = st.snap_count[0];
    for (int i = si; i < n - 1; ++i) { st.snap_id[i] = st.snap_id[i+1]; st.snap_seq[i] = st.snap_seq[i+1]; }
    st.snap_count[0] = n - 1;
    st.counts[C_snapshot_released] += 1;
}

// ---------------- flush ----------------
__device__ int r_alloc_file(RefDevState& st) {
    // find a free slot in [0, LSM_MAX_TOTAL_FILES).
    for (int f = 0; f < LSM_MAX_TOTAL_FILES; ++f) {
        if (!st.f_used[f]) {
            if (f >= st.file_count[0]) st.file_count[0] = f + 1;
            return f;
        }
    }
    return -1;
}

__device__ int r_count_level(RefDevState& st, int lvl) {
    int n = 0, nf = st.file_count[0];
    for (int f = 0; f < nf; ++f) if (st.f_used[f] && st.f_level[f] == lvl) ++n;
    return n;
}

__device__ void r_do_flush(RefDevState& st, uint32_t op_index) {
    (void)op_index;
    if (st.mt_count[0] == 0) { st.counts[C_flush_empty] += 1; return; }
    if (r_count_level(st, 0) >= st.spec.max_files_per_level[0]) { st.counts[C_flush_oom] += 1; return; }
    int f = r_alloc_file(st);
    st.f_used[f] = 1;
    st.f_file_id[f] = st.sc[1]++;
    st.f_level[f] = 0;
    st.f_create_event[f] = r_next_event(st);
    int mc = st.mt_count[0];
    size_t base = r_ebase(f);
    // copy memtable entries.
    for (int i = 0; i < mc; ++i) {
        st.e_key[base + i] = st.mt_key[i];
        st.e_seq[base + i] = st.mt_seq[i];
        st.e_kind[base + i] = st.mt_kind[i];
        st.e_value[base + i] = st.mt_value[i];
    }
    st.f_entry_count[f] = mc;
    // sort entries key asc, seq desc (insertion sort).
    for (int i = 1; i < mc; ++i) {
        uint64_t k = st.e_key[base+i], s = st.e_seq[base+i]; int kd = st.e_kind[base+i]; int64_t v = st.e_value[base+i];
        int j = i - 1;
        while (j >= 0 && (st.e_key[base+j] > k || (st.e_key[base+j] == k && st.e_seq[base+j] < s))) {
            st.e_key[base+j+1] = st.e_key[base+j];
            st.e_seq[base+j+1] = st.e_seq[base+j];
            st.e_kind[base+j+1] = st.e_kind[base+j];
            st.e_value[base+j+1] = st.e_value[base+j];
            --j;
        }
        st.e_key[base+j+1] = k; st.e_seq[base+j+1] = s; st.e_kind[base+j+1] = kd; st.e_value[base+j+1] = v;
    }
    uint64_t mn = st.e_key[base], mx = st.e_key[base], ms = 0;
    for (int i = 0; i < mc; ++i) {
        if (st.e_key[base+i] < mn) mn = st.e_key[base+i];
        if (st.e_key[base+i] > mx) mx = st.e_key[base+i];
        if (st.e_seq[base+i] > ms) ms = st.e_seq[base+i];
    }
    st.f_min_key[f] = mn; st.f_max_key[f] = mx; st.f_max_seq[f] = ms;
    st.mt_count[0] = 0;
    if (ms > st.sc[4]) st.sc[4] = ms;
    st.counts[C_flush_files] += 1;
}

__device__ __forceinline__ bool r_overlap(uint64_t amn, uint64_t amx, uint64_t bmn, uint64_t bmx) {
    return amn <= bmx && bmn <= amx;
}

// ---------------- compaction ----------------
// workspace layout (provided in run). We use a merged-entry scratch.
struct CompactScratch {
    // merged entries
    uint64_t* key;
    uint64_t* seq;
    int32_t*  kind;
    int64_t*  value;
    int32_t*  src_level;
    uint64_t* src_file;
    int32_t   count;
    // retained
    uint64_t* rkey;
    uint64_t* rseq;
    int32_t*  rkind;
    int64_t*  rvalue;
    int32_t   rcount;
};

__device__ void r_do_compact(RefDevState& st, void* ws, size_t ws_bytes,
                             int level, int max_primary, uint32_t op_index) {
    if (level < 0 || level >= st.spec.level_count - 1 || max_primary == 0) {
        st.counts[C_invalid_count] += 1; return;
    }
    if (r_count_level(st, level) == 0) { st.counts[C_compact_empty] += 1; return; }

    // carve scratch arrays out of workspace.
    const int MAXM = LSM_MAX_TOTAL_ENTRIES;
    uint8_t* p = (uint8_t*)ws;
    uint64_t* m_key = (uint64_t*)p; p += sizeof(uint64_t)*MAXM;
    uint64_t* m_seq = (uint64_t*)p; p += sizeof(uint64_t)*MAXM;
    int32_t*  m_kind = (int32_t*)p; p += sizeof(int32_t)*MAXM;
    int64_t*  m_val = (int64_t*)p; p += sizeof(int64_t)*MAXM;
    int32_t*  m_lvl = (int32_t*)p; p += sizeof(int32_t)*MAXM;
    uint64_t* m_file = (uint64_t*)p; p += sizeof(uint64_t)*MAXM;
    uint64_t* r_key = (uint64_t*)p; p += sizeof(uint64_t)*MAXM;
    uint64_t* r_seq = (uint64_t*)p; p += sizeof(uint64_t)*MAXM;
    int32_t*  r_kind = (int32_t*)p; p += sizeof(int32_t)*MAXM;
    int64_t*  r_val = (int64_t*)p; p += sizeof(int64_t)*MAXM;
    int32_t*  prim = (int32_t*)p; p += sizeof(int32_t)*LSM_MAX_FILES_PER_LEVEL;
    int32_t*  nxt = (int32_t*)p; p += sizeof(int32_t)*LSM_MAX_FILES_PER_LEVEL;
    (void)ws_bytes;

    int nfiles = st.file_count[0];
    int nprim = 0, nnext = 0;
    uint64_t rmin = 0, rmax = 0;

    if (level == 0) {
        // primary = smallest file_id at L0.
        int best = -1; uint64_t bid = 0;
        for (int f = 0; f < nfiles; ++f) {
            if (!st.f_used[f] || st.f_level[f] != 0) continue;
            if (best < 0 || st.f_file_id[f] < bid) { best = f; bid = st.f_file_id[f]; }
        }
        prim[nprim++] = best;
        rmin = st.f_min_key[best]; rmax = st.f_max_key[best];
        // expand transitively, smallest file_id overlapping, bounded.
        while (nprim < max_primary) {
            int cand = -1; uint64_t cid = 0;
            for (int f = 0; f < nfiles; ++f) {
                if (!st.f_used[f] || st.f_level[f] != 0) continue;
                bool already = false;
                for (int q = 0; q < nprim; ++q) if (prim[q] == f) { already = true; break; }
                if (already) continue;
                if (r_overlap(st.f_min_key[f], st.f_max_key[f], rmin, rmax)) {
                    if (cand < 0 || st.f_file_id[f] < cid) { cand = f; cid = st.f_file_id[f]; }
                }
            }
            if (cand < 0) break;
            prim[nprim++] = cand;
            if (st.f_min_key[cand] < rmin) rmin = st.f_min_key[cand];
            if (st.f_max_key[cand] > rmax) rmax = st.f_max_key[cand];
        }
    } else {
        int best = -1;
        for (int f = 0; f < nfiles; ++f) {
            if (!st.f_used[f] || st.f_level[f] != level) continue;
            if (best < 0) { best = f; continue; }
            if (st.f_min_key[f] < st.f_min_key[best] ||
                (st.f_min_key[f] == st.f_min_key[best] && st.f_file_id[f] < st.f_file_id[best])) {
                best = f;
            }
        }
        prim[nprim++] = best;
        rmin = st.f_min_key[best]; rmax = st.f_max_key[best];
    }

    // overlapping files in level+1.
    for (int f = 0; f < nfiles; ++f) {
        if (!st.f_used[f] || st.f_level[f] != level + 1) continue;
        if (r_overlap(st.f_min_key[f], st.f_max_key[f], rmin, rmax)) nxt[nnext++] = f;
    }

    // merge.
    int mc = 0;
    for (int q = 0; q < nprim; ++q) {
        int f = prim[q]; size_t base = r_ebase(f); int ec = st.f_entry_count[f];
        for (int e = 0; e < ec; ++e) {
            m_key[mc] = st.e_key[base+e]; m_seq[mc] = st.e_seq[base+e];
            m_kind[mc] = st.e_kind[base+e]; m_val[mc] = st.e_value[base+e];
            m_lvl[mc] = st.f_level[f]; m_file[mc] = st.f_file_id[f]; ++mc;
        }
    }
    for (int q = 0; q < nnext; ++q) {
        int f = nxt[q]; size_t base = r_ebase(f); int ec = st.f_entry_count[f];
        for (int e = 0; e < ec; ++e) {
            m_key[mc] = st.e_key[base+e]; m_seq[mc] = st.e_seq[base+e];
            m_kind[mc] = st.e_kind[base+e]; m_val[mc] = st.e_value[base+e];
            m_lvl[mc] = st.f_level[f]; m_file[mc] = st.f_file_id[f]; ++mc;
        }
    }

    // sort merged: key asc, seq desc, level asc, file_id desc (insertion sort).
    for (int i = 1; i < mc; ++i) {
        uint64_t k=m_key[i], s=m_seq[i]; int kd=m_kind[i]; int64_t v=m_val[i]; int lv=m_lvl[i]; uint64_t fl=m_file[i];
        int j = i - 1;
        while (j >= 0) {
            bool greater;
            uint64_t jk=m_key[j], js=m_seq[j]; int jlv=m_lvl[j]; uint64_t jfl=m_file[j];
            if (jk != k) greater = jk > k;
            else if (js != s) greater = js < s;
            else if (jlv != lv) greater = jlv > lv;
            else greater = jfl < fl;
            if (!greater) break;
            m_key[j+1]=m_key[j]; m_seq[j+1]=m_seq[j]; m_kind[j+1]=m_kind[j];
            m_val[j+1]=m_val[j]; m_lvl[j+1]=m_lvl[j]; m_file[j+1]=m_file[j];
            --j;
        }
        m_key[j+1]=k; m_seq[j+1]=s; m_kind[j+1]=kd; m_val[j+1]=v; m_lvl[j+1]=lv; m_file[j+1]=fl;
    }
    // dedup (key,seq): keep first.
    int dc = 0;
    for (int i = 0; i < mc; ++i) {
        if (dc > 0 && m_key[i] == m_key[dc-1] && m_seq[i] == m_seq[dc-1]) {
            // skip duplicate; but we are compacting in place using same arrays.
            continue;
        }
        m_key[dc]=m_key[i]; m_seq[dc]=m_seq[i]; m_kind[dc]=m_kind[i];
        m_val[dc]=m_val[i]; m_lvl[dc]=m_lvl[i]; m_file[dc]=m_file[i];
        ++dc;
    }

    // oldest snapshot seq.
    uint64_t oldest = st.sc[0] - 1;
    if (st.snap_count[0] > 0) {
        oldest = st.snap_seq[0];
        for (int i = 1; i < st.snap_count[0]; ++i) if (st.snap_seq[i] < oldest) oldest = st.snap_seq[i];
    }

    // retention. iterate groups by key.
    int rc = 0;
    // We will emit drops AFTER mutation, so record them in scratch r_* won't work
    // (those are for retained). Use a separate drop buffer carved from remaining ws.
    uint64_t* d_key = (uint64_t*)p; p += sizeof(uint64_t)*MAXM;
    uint64_t* d_seq = (uint64_t*)p; p += sizeof(uint64_t)*MAXM;
    int32_t*  d_kind = (int32_t*)p; p += sizeof(int32_t)*MAXM;
    int64_t*  d_val = (int64_t*)p; p += sizeof(int64_t)*MAXM;
    int32_t*  d_ev = (int32_t*)p; p += sizeof(int32_t)*MAXM;
    int dn = 0;

    int i = 0;
    while (i < dc) {
        int j = i; uint64_t key = m_key[i];
        while (j < dc && m_key[j] == key) ++j;
        // deeper coverage?
        bool deeper = false;
        for (int dl = level + 2; dl < st.spec.level_count && !deeper; ++dl) {
            for (int f = 0; f < nfiles; ++f) {
                if (!st.f_used[f] || st.f_level[f] != dl) continue;
                if (key >= st.f_min_key[f] && key <= st.f_max_key[f]) { deeper = true; break; }
            }
        }
        bool stopped = false, kept_floor = false;
        for (int rIdx = i; rIdx < j; ++rIdx) {
            if (m_seq[rIdx] > oldest) {
                r_key[rc]=m_key[rIdx]; r_seq[rc]=m_seq[rIdx]; r_kind[rc]=m_kind[rIdx]; r_val[rc]=m_val[rIdx]; ++rc;
                continue;
            }
            if (stopped) {
                d_key[dn]=m_key[rIdx]; d_seq[dn]=m_seq[rIdx]; d_kind[dn]=m_kind[rIdx]; d_val[dn]=m_val[rIdx]; d_ev[dn]=LSM_EV_DROP_OLD_VERSION; ++dn;
                continue;
            }
            if (!kept_floor) {
                if (m_kind[rIdx] == LSM_KIND_DEL && !deeper) {
                    d_key[dn]=m_key[rIdx]; d_seq[dn]=m_seq[rIdx]; d_kind[dn]=m_kind[rIdx]; d_val[dn]=m_val[rIdx]; d_ev[dn]=LSM_EV_DROP_TOMBSTONE; ++dn;
                    stopped = true;
                } else {
                    r_key[rc]=m_key[rIdx]; r_seq[rc]=m_seq[rIdx]; r_kind[rc]=m_kind[rIdx]; r_val[rc]=m_val[rIdx]; ++rc;
                    kept_floor = true;
                }
            } else {
                d_key[dn]=m_key[rIdx]; d_seq[dn]=m_seq[rIdx]; d_kind[dn]=m_kind[rIdx]; d_val[dn]=m_val[rIdx]; d_ev[dn]=LSM_EV_DROP_OLD_VERSION; ++dn;
            }
        }
        i = j;
    }

    // output file count.
    int cap = st.spec.sst_record_cap;
    int out_count = (rc == 0) ? 0 : ((rc + cap - 1) / cap);
    int remaining_next = r_count_level(st, level + 1) - nnext;
    if (remaining_next + out_count > st.spec.max_files_per_level[level + 1]) {
        st.counts[C_compact_oom] += 1; return;
    }

    // collect obsolete descriptors (sorted level asc, file_id asc).
    // store into prim/nxt-like arrays: reuse stack arrays via scratch d-region? Use simple local arrays.
    // total inputs <= 2*LSM_MAX_FILES_PER_LEVEL
    int ninp = nprim + nnext;
    // sort indices by (level, file_id). We'll do selection emission.
    // Build arrays of (level,file_id,slot).
    // Use d_* region tail is consumed; allocate small fixed local arrays.
    int inp_slot[2 * LSM_MAX_FILES_PER_LEVEL];
    for (int q = 0; q < nprim; ++q) inp_slot[q] = prim[q];
    for (int q = 0; q < nnext; ++q) inp_slot[nprim + q] = nxt[q];
    // insertion sort by (level, file_id).
    for (int a = 1; a < ninp; ++a) {
        int s = inp_slot[a]; int j2 = a - 1;
        while (j2 >= 0) {
            int sj = inp_slot[j2];
            bool greater;
            if (st.f_level[sj] != st.f_level[s]) greater = st.f_level[sj] > st.f_level[s];
            else greater = st.f_file_id[sj] > st.f_file_id[s];
            if (!greater) break;
            inp_slot[j2+1] = inp_slot[j2]; --j2;
        }
        inp_slot[j2+1] = s;
    }

    // Capture obsolete descriptors before removal.
    uint64_t obs_fid[2*LSM_MAX_FILES_PER_LEVEL], obs_mn[2*LSM_MAX_FILES_PER_LEVEL],
             obs_mx[2*LSM_MAX_FILES_PER_LEVEL], obs_ec[2*LSM_MAX_FILES_PER_LEVEL],
             obs_ms[2*LSM_MAX_FILES_PER_LEVEL];
    int32_t  obs_lv[2*LSM_MAX_FILES_PER_LEVEL];
    for (int q = 0; q < ninp; ++q) {
        int f = inp_slot[q];
        obs_lv[q]=st.f_level[f]; obs_fid[q]=st.f_file_id[f]; obs_mn[q]=st.f_min_key[f];
        obs_mx[q]=st.f_max_key[f]; obs_ec[q]=(uint64_t)st.f_entry_count[f]; obs_ms[q]=st.f_max_seq[f];
    }

    // Build output files (allocate slots, assign file_ids in order).
    int out_slot[LSM_MAX_FILES_PER_LEVEL];
    {
        int pos = 0, oi = 0;
        while (pos < rc) {
            int end = pos + cap; if (end > rc) end = rc;
            int f = r_alloc_file(st);
            st.f_used[f] = 1;
            st.f_file_id[f] = st.sc[1]++;
            st.f_level[f] = level + 1;
            size_t base = r_ebase(f);
            uint64_t mn = r_key[pos], mx = r_key[pos], ms = 0;
            int ec = 0;
            for (int t = pos; t < end; ++t) {
                st.e_key[base+ec]=r_key[t]; st.e_seq[base+ec]=r_seq[t];
                st.e_kind[base+ec]=r_kind[t]; st.e_value[base+ec]=r_val[t];
                if (r_key[t] < mn) mn = r_key[t];
                if (r_key[t] > mx) mx = r_key[t];
                if (r_seq[t] > ms) ms = r_seq[t];
                ++ec;
            }
            st.f_entry_count[f] = ec;
            st.f_min_key[f] = mn; st.f_max_key[f] = mx; st.f_max_seq[f] = ms;
            out_slot[oi++] = f;
            pos = end;
        }
    }

    // Remove input files.
    for (int q = 0; q < ninp; ++q) { int f = inp_slot[q]; st.f_used[f] = 0; st.f_entry_count[f] = 0; }

    // Emit obsolete events (already sorted level asc, file_id asc).
    for (int q = 0; q < ninp; ++q) {
        r_emit_cfile(st, LSM_EV_COMPACT_INPUT_OBSOLETE, op_index, obs_lv[q], obs_fid[q],
                     obs_mn[q], obs_mx[q], obs_ec[q], obs_ms[q], 0);
    }
    // Emit output events in new file_id order (out_slot already in id order).
    for (int q = 0; q < out_count; ++q) {
        int f = out_slot[q]; uint64_t es = 0;
        r_emit_cfile(st, LSM_EV_COMPACT_OUTPUT_FILE, op_index, st.f_level[f], st.f_file_id[f],
                     st.f_min_key[f], st.f_max_key[f], (uint64_t)st.f_entry_count[f], st.f_max_seq[f], &es);
        st.f_create_event[f] = es;
    }
    // Emit drops in scan order.
    for (int q = 0; q < dn; ++q) {
        r_emit_cver(st, d_ev[q], op_index, d_key[q], d_seq[q], d_kind[q], d_val[q]);
        if (d_ev[q] == LSM_EV_DROP_OLD_VERSION) st.counts[C_versions_dropped] += 1;
        else st.counts[C_tombstones_dropped] += 1;
    }

    st.counts[C_compact_input_files] += ninp;
    st.counts[C_compact_output_files] += out_count;
    st.counts[C_obsolete_files] += ninp;
}

// ---------------- checkpoint wal ----------------
__device__ void r_do_checkpoint(RefDevState& st, uint32_t op_index) {
    (void)op_index;
    int n = st.wal_seg_count[0];
    // compute per-segment max seq from records.
    int archived = 0;
    // determine which segments to archive.
    // segment max seq: scan wr.
    bool keep[LSM_MAX_WAL_SEGMENTS];
    for (int s = 0; s < n; ++s) keep[s] = true;
    for (int s = 0; s < n; ++s) {
        if (st.wal_seg_reccount[s] == 0) { keep[s] = true; continue; }
        uint64_t mx = 0; bool any = false;
        int wc = st.wr_count[0];
        for (int r = 0; r < wc; ++r) {
            if (st.wr_seg[r] == s) { if (!any || st.wr_seq[r] > mx) { mx = st.wr_seq[r]; any = true; } }
        }
        if (any && mx <= st.sc[4]) { keep[s] = false; ++archived; }
    }
    if (archived == 0) return;

    // rebuild wal: drop archived segments and their records, keep order by wal_id.
    // First remap kept segments to new slots.
    int newmap[LSM_MAX_WAL_SEGMENTS];
    int newn = 0;
    uint64_t new_seg_id[LSM_MAX_WAL_SEGMENTS];
    int32_t new_seg_rec[LSM_MAX_WAL_SEGMENTS];
    for (int s = 0; s < n; ++s) {
        if (keep[s]) {
            newmap[s] = newn;
            new_seg_id[newn] = st.wal_seg_id[s];
            new_seg_rec[newn] = st.wal_seg_reccount[s];
            ++newn;
        } else {
            newmap[s] = -1;
        }
    }
    // compact records.
    int wc = st.wr_count[0];
    int wn = 0;
    for (int r = 0; r < wc; ++r) {
        int s = st.wr_seg[r];
        if (newmap[s] < 0) continue;
        st.wr_walid[wn]=st.wr_walid[r]; st.wr_offset[wn]=st.wr_offset[r];
        st.wr_seq[wn]=st.wr_seq[r]; st.wr_kind[wn]=st.wr_kind[r];
        st.wr_key[wn]=st.wr_key[r]; st.wr_value[wn]=st.wr_value[r];
        st.wr_seg[wn]=newmap[s];
        ++wn;
    }
    st.wr_count[0] = wn;
    for (int s = 0; s < newn; ++s) { st.wal_seg_id[s]=new_seg_id[s]; st.wal_seg_reccount[s]=new_seg_rec[s]; }
    st.wal_seg_count[0] = newn;
    if (newn == 0) {
        uint64_t nid = ++st.sc[2];
        st.wal_seg_id[0] = nid; st.wal_seg_reccount[0] = 0;
        st.wal_seg_count[0] = 1;
    }
    st.counts[C_wal_archived] += archived;
}

// ---------------- crash recover ----------------
__device__ void r_do_crash(RefDevState& st, uint64_t cut_wal_id, uint64_t cut_off, uint32_t op_index) {
    (void)op_index;
    st.mt_count[0] = 0;
    // truncate WAL records by (wal_id, offset) <= cut.
    int wc = st.wr_count[0]; int wn = 0;
    for (int r = 0; r < wc; ++r) {
        uint64_t wid = st.wr_walid[r], off = st.wr_offset[r];
        if (wid < cut_wal_id || (wid == cut_wal_id && off <= cut_off)) {
            st.wr_walid[wn]=wid; st.wr_offset[wn]=off; st.wr_seq[wn]=st.wr_seq[r];
            st.wr_kind[wn]=st.wr_kind[r]; st.wr_key[wn]=st.wr_key[r]; st.wr_value[wn]=st.wr_value[r];
            st.wr_seg[wn]=st.wr_seg[r];
            ++wn;
        }
    }
    st.wr_count[0] = wn;
    // rebuild segments: keep only segments that retain records; recompute reccount.
    int n = st.wal_seg_count[0];
    int32_t segrec[LSM_MAX_WAL_SEGMENTS];
    for (int s = 0; s < n; ++s) segrec[s] = 0;
    for (int r = 0; r < wn; ++r) segrec[st.wr_seg[r]]++;
    // remap kept segments (those with >0 records).
    int newmap[LSM_MAX_WAL_SEGMENTS]; int newn = 0;
    uint64_t new_id[LSM_MAX_WAL_SEGMENTS]; int32_t new_rec[LSM_MAX_WAL_SEGMENTS];
    for (int s = 0; s < n; ++s) {
        if (segrec[s] > 0) { newmap[s]=newn; new_id[newn]=st.wal_seg_id[s]; new_rec[newn]=segrec[s]; ++newn; }
        else newmap[s] = -1;
    }
    for (int r = 0; r < wn; ++r) st.wr_seg[r] = newmap[st.wr_seg[r]];
    for (int s = 0; s < newn; ++s) { st.wal_seg_id[s]=new_id[s]; st.wal_seg_reccount[s]=new_rec[s]; }
    st.wal_seg_count[0] = newn;
    // D11: ensure a current segment exists for subsequent writes.
    if (newn == 0) {
        uint64_t nid = ++st.sc[2];
        st.wal_seg_id[0] = nid; st.wal_seg_reccount[0] = 0;
        st.wal_seg_count[0] = 1;
    }

    // drop snapshots (ascending id) -> count, snapshots already sorted asc.
    int sc = st.snap_count[0];
    st.counts[C_snapshots_dropped_by_crash] += sc;
    st.snap_count[0] = 0;

    // surviving records in WAL order (wal_id asc, offset asc). They are already
    // grouped by segment in wal_id order? Not necessarily by storage order;
    // sort indices.
    // Build an index array and selection-sort small.
    // Replay records with seq > durable into memtable in WAL order.
    // Create order array.
    // We sort surviving records by (wal_id, offset).
    // Use insertion sort on a permutation.
    int perm[LSM_MAX_WAL_RECORDS];
    for (int r = 0; r < wn; ++r) perm[r] = r;
    for (int a = 1; a < wn; ++a) {
        int pa = perm[a]; int j = a - 1;
        while (j >= 0) {
            int pj = perm[j];
            bool greater;
            if (st.wr_walid[pj] != st.wr_walid[pa]) greater = st.wr_walid[pj] > st.wr_walid[pa];
            else greater = st.wr_offset[pj] > st.wr_offset[pa];
            if (!greater) break;
            perm[j+1] = perm[j]; --j;
        }
        perm[j+1] = pa;
    }

    bool stalled = false;
    for (int a = 0; a < wn; ++a) {
        int r = perm[a];
        if (st.wr_seq[r] <= st.sc[4]) continue;
        if (stalled) { st.counts[C_recover_stalled_records] += 1; continue; }
        if (st.mt_count[0] >= st.spec.memtable_record_cap) {
            stalled = true; st.counts[C_recover_stalled_records] += 1; continue;
        }
        int mc = st.mt_count[0];
        st.mt_seq[mc]=st.wr_seq[r]; st.mt_kind[mc]=st.wr_kind[r];
        st.mt_key[mc]=st.wr_key[r]; st.mt_value[mc]=st.wr_value[r];
        st.mt_count[0] = mc + 1;
        st.counts[C_recovered_records] += 1;
    }

    // next_seq = 1 + max seq over SSTs, memtable, surviving WAL.
    uint64_t maxseq = 0; bool any = false;
    int nf = st.file_count[0];
    for (int f = 0; f < nf; ++f) {
        if (!st.f_used[f]) continue;
        size_t base = r_ebase(f); int ec = st.f_entry_count[f];
        for (int e = 0; e < ec; ++e) { if (!any || st.e_seq[base+e] > maxseq) { maxseq = st.e_seq[base+e]; any = true; } }
    }
    int mc = st.mt_count[0];
    for (int i = 0; i < mc; ++i) { if (!any || st.mt_seq[i] > maxseq) { maxseq = st.mt_seq[i]; any = true; } }
    for (int r = 0; r < wn; ++r) { if (!any || st.wr_seq[r] > maxseq) { maxseq = st.wr_seq[r]; any = true; } }
    st.sc[0] = any ? (maxseq + 1) : 1;
}

// ---------------- structural hashes ----------------
__device__ uint64_t r_wal_hash(RefDevState& st) {
    uint64_t h = R_FNV_BASE;
    int n = st.wal_seg_count[0];
    // segments by wal_id asc. Slots already in wal_id asc order (we maintain).
    // To be safe, emit by ascending wal_id via selection.
    bool done[LSM_MAX_WAL_SEGMENTS];
    for (int s = 0; s < n; ++s) done[s] = false;
    for (int it = 0; it < n; ++it) {
        int best = -1;
        for (int s = 0; s < n; ++s) {
            if (done[s]) continue;
            if (best < 0 || st.wal_seg_id[s] < st.wal_seg_id[best]) best = s;
        }
        done[best] = true;
        // records of this segment by offset asc.
        int wc = st.wr_count[0];
        // gather indices then sort by offset.
        // simple: iterate offsets 0.. by scanning min offset not yet emitted.
        // Use a linear emit by ascending offset.
        uint64_t last_off = 0; bool have_last = false;
        for (int emitted = 0; ; ++emitted) {
            int pick = -1; uint64_t pick_off = 0;
            for (int r = 0; r < wc; ++r) {
                if (st.wr_seg[r] != best) continue;
                uint64_t off = st.wr_offset[r];
                if (have_last && off <= last_off) continue;
                if (pick < 0 || off < pick_off) { pick = r; pick_off = off; }
            }
            if (pick < 0) break;
            have_last = true; last_off = pick_off;
            r_fnv_u64(&h, st.wr_walid[pick]);
            r_fnv_u64(&h, st.wr_offset[pick]);
            r_fnv_u64(&h, st.wr_seq[pick]);
            r_fnv_u8(&h, (uint8_t)st.wr_kind[pick]);
            r_fnv_u64(&h, st.wr_key[pick]);
            r_fnv_i64(&h, st.wr_value[pick]);
        }
    }
    return h;
}

__device__ uint64_t r_state_hash(RefDevState& st) {
    uint64_t h = R_FNV_BASE;
    // memtable descending seq.
    int mc = st.mt_count[0];
    uint64_t last_seq = 0; bool have_last = false;
    for (int emitted = 0; emitted < mc; ++emitted) {
        int pick = -1; uint64_t pick_seq = 0;
        for (int i = 0; i < mc; ++i) {
            uint64_t s = st.mt_seq[i];
            if (have_last && s >= last_seq) continue;
            if (pick < 0 || s > pick_seq) { pick = i; pick_seq = s; }
        }
        if (pick < 0) break;
        have_last = true; last_seq = pick_seq;
        r_fnv_u8(&h, 0);
        r_fnv_u64(&h, st.mt_seq[pick]);
        r_fnv_u8(&h, (uint8_t)st.mt_kind[pick]);
        r_fnv_u64(&h, st.mt_key[pick]);
        r_fnv_i64(&h, st.mt_value[pick]);
    }
    // files by level asc, file_id asc.
    int nf = st.file_count[0];
    for (int lvl = 0; lvl < st.spec.level_count; ++lvl) {
        uint64_t last_id = 0; bool hl = false;
        while (true) {
            int pick = -1; uint64_t pid = 0;
            for (int f = 0; f < nf; ++f) {
                if (!st.f_used[f] || st.f_level[f] != lvl) continue;
                uint64_t fid = st.f_file_id[f];
                if (hl && fid <= last_id) continue;
                if (pick < 0 || fid < pid) { pick = f; pid = fid; }
            }
            if (pick < 0) break;
            hl = true; last_id = pid;
            size_t base = r_ebase(pick); int ec = st.f_entry_count[pick];
            for (int e = 0; e < ec; ++e) {
                r_fnv_u8(&h, 1);
                r_fnv_u32(&h, (uint32_t)lvl);
                r_fnv_u64(&h, st.f_file_id[pick]);
                r_fnv_u64(&h, st.f_min_key[pick]);
                r_fnv_u64(&h, st.f_max_key[pick]);
                r_fnv_u64(&h, st.f_max_seq[pick]);
                r_fnv_u64(&h, (uint64_t)e);
                r_fnv_u64(&h, st.e_key[base+e]);
                r_fnv_u64(&h, st.e_seq[base+e]);
                r_fnv_u8(&h, (uint8_t)st.e_kind[base+e]);
                r_fnv_i64(&h, st.e_value[base+e]);
            }
        }
    }
    return h;
}

__device__ uint64_t r_snap_hash(RefDevState& st) {
    uint64_t h = R_FNV_BASE;
    int n = st.snap_count[0];
    // snapshots stored sorted asc already.
    for (int i = 0; i < n; ++i) {
        r_fnv_u64(&h, st.snap_id[i]);
        r_fnv_u64(&h, st.snap_seq[i]);
    }
    return h;
}

// ---------------- step kernel ----------------
__global__ void ref_step_kernel(RefDevState st, const LsmOp* ops, int num_ops,
                                void* ws, size_t ws_bytes,
                                LsmCounts* out_counts, uint64_t* out_read,
                                uint64_t* out_write, uint64_t* out_compact,
                                uint64_t* out_wal, uint64_t* out_state,
                                uint64_t* out_snap) {
    if (blockIdx.x || threadIdx.x) return;
    for (int i = 0; i < num_ops; ++i) {
        const LsmOp o = ops[i];
        uint32_t oi = (uint32_t)i;
        switch (o.kind) {
            case LSM_OP_PUT: r_do_write(st, LSM_KIND_PUT, (uint64_t)(uint32_t)o.i_a, o.value, oi); break;
            case LSM_OP_DEL: r_do_write(st, LSM_KIND_DEL, (uint64_t)(uint32_t)o.i_a, 0, oi); break;
            case LSM_OP_GET: r_do_get(st, o.u_a, (uint64_t)(uint32_t)o.i_a, o.u_b, oi); break;
            case LSM_OP_OPEN_SNAPSHOT: r_do_open_snap(st, o.u_a); break;
            case LSM_OP_RELEASE_SNAPSHOT: r_do_release_snap(st, o.u_a); break;
            case LSM_OP_FLUSH: r_do_flush(st, oi); break;
            case LSM_OP_COMPACT: r_do_compact(st, ws, ws_bytes, o.i_a, o.i_b, oi); break;
            case LSM_OP_CHECKPOINT_WAL: r_do_checkpoint(st, oi); break;
            case LSM_OP_CRASH_RECOVER: r_do_crash(st, (uint64_t)(uint32_t)o.i_a, (uint64_t)(uint32_t)o.i_b, oi); break;
            default: st.counts[C_invalid_count] += 1; break;
        }
    }
    // outputs.
    LsmCounts c;
    int64_t* cp = (int64_t*)&c;
    for (int i = 0; i < 24; ++i) cp[i] = st.counts[i];
    *out_counts = c;
    *out_read = st.hsh[0];
    *out_write = st.hsh[1];
    *out_compact = st.hsh[2];
    *out_wal = r_wal_hash(st);
    *out_state = r_state_hash(st);
    *out_snap = r_snap_hash(st);
}

// ---------------- host ABI ----------------
static void ref_free_state(RefDevState* st) {
#define FREE(p) if (st->p) cudaFree(st->p)
    FREE(sc); FREE(hsh); FREE(counts);
    FREE(wal_seg_count); FREE(wal_seg_id); FREE(wal_seg_reccount);
    FREE(wr_walid); FREE(wr_offset); FREE(wr_seq); FREE(wr_kind); FREE(wr_key); FREE(wr_value); FREE(wr_seg); FREE(wr_count);
    FREE(mt_count); FREE(mt_seq); FREE(mt_kind); FREE(mt_key); FREE(mt_value);
    FREE(file_count); FREE(f_used); FREE(f_file_id); FREE(f_level); FREE(f_min_key); FREE(f_max_key); FREE(f_max_seq); FREE(f_create_event); FREE(f_entry_count);
    FREE(e_key); FREE(e_seq); FREE(e_kind); FREE(e_value);
    FREE(snap_count); FREE(snap_id); FREE(snap_seq);
#undef FREE
}

extern "C" size_t solution_workspace_bytes(const LsmProblemSpec* spec) {
    if (!lsm_validate_problem_spec(spec)) return 0;
    // merged + retained + drops, plus index arrays.
    const size_t MAXM = LSM_MAX_TOTAL_ENTRIES;
    size_t bytes = 0;
    bytes += MAXM * (sizeof(uint64_t)*2 + sizeof(int32_t) + sizeof(int64_t) + sizeof(int32_t) + sizeof(uint64_t)); // merged
    bytes += MAXM * (sizeof(uint64_t)*2 + sizeof(int32_t) + sizeof(int64_t)); // retained
    bytes += LSM_MAX_FILES_PER_LEVEL * sizeof(int32_t) * 2; // prim, nxt
    bytes += MAXM * (sizeof(uint64_t)*2 + sizeof(int32_t)*2 + sizeof(int64_t)); // drops
    bytes += 4096; // slack
    return bytes;
}

extern "C" cudaError_t solution_init(const LsmProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!lsm_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    RefDevState* st = (RefDevState*)malloc(sizeof(RefDevState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(RefDevState));
    memcpy(&st->spec, spec, sizeof(LsmProblemSpec));

    cudaError_t err = cudaSuccess;
    const size_t F = LSM_MAX_TOTAL_FILES;
    const size_t E = (size_t)LSM_MAX_TOTAL_FILES * (size_t)LSM_MAX_ENTRIES_PER_FILE;
    const size_t W = LSM_MAX_WAL_RECORDS;
    const size_t S = LSM_MAX_WAL_SEGMENTS;
    const size_t M = LSM_MAX_MEMTABLE;
    const size_t SN = LSM_MAX_SNAPSHOTS;

#define ALLOC(p, n) do { err = cudaMalloc((void**)&st->p, (n)); if (err != cudaSuccess) goto fail; } while(0)
    ALLOC(sc, sizeof(uint64_t)*8);
    ALLOC(hsh, sizeof(uint64_t)*3);
    ALLOC(counts, sizeof(int64_t)*24);
    ALLOC(wal_seg_count, sizeof(int32_t));
    ALLOC(wal_seg_id, sizeof(uint64_t)*S);
    ALLOC(wal_seg_reccount, sizeof(int32_t)*S);
    ALLOC(wr_walid, sizeof(uint64_t)*W);
    ALLOC(wr_offset, sizeof(uint64_t)*W);
    ALLOC(wr_seq, sizeof(uint64_t)*W);
    ALLOC(wr_kind, sizeof(int32_t)*W);
    ALLOC(wr_key, sizeof(uint64_t)*W);
    ALLOC(wr_value, sizeof(int64_t)*W);
    ALLOC(wr_seg, sizeof(int32_t)*W);
    ALLOC(wr_count, sizeof(int32_t));
    ALLOC(mt_count, sizeof(int32_t));
    ALLOC(mt_seq, sizeof(uint64_t)*M);
    ALLOC(mt_kind, sizeof(int32_t)*M);
    ALLOC(mt_key, sizeof(uint64_t)*M);
    ALLOC(mt_value, sizeof(int64_t)*M);
    ALLOC(file_count, sizeof(int32_t));
    ALLOC(f_used, sizeof(int32_t)*F);
    ALLOC(f_file_id, sizeof(uint64_t)*F);
    ALLOC(f_level, sizeof(int32_t)*F);
    ALLOC(f_min_key, sizeof(uint64_t)*F);
    ALLOC(f_max_key, sizeof(uint64_t)*F);
    ALLOC(f_max_seq, sizeof(uint64_t)*F);
    ALLOC(f_create_event, sizeof(uint64_t)*F);
    ALLOC(f_entry_count, sizeof(int32_t)*F);
    ALLOC(e_key, sizeof(uint64_t)*E);
    ALLOC(e_seq, sizeof(uint64_t)*E);
    ALLOC(e_kind, sizeof(int32_t)*E);
    ALLOC(e_value, sizeof(int64_t)*E);
    ALLOC(snap_count, sizeof(int32_t));
    ALLOC(snap_id, sizeof(uint64_t)*SN);
    ALLOC(snap_seq, sizeof(uint64_t)*SN);
#undef ALLOC

    ref_reset_kernel<<<1,1,0,stream>>>(*st);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;
fail:
    ref_free_state(st);
    free(st);
    return err;
}

extern "C" cudaError_t solution_run(void* state, const LsmRunSpec* run, const void* inputs_void,
                                    void* outputs_void, void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    if (!state || !outputs_void) return cudaErrorInvalidValue;
    RefDevState* st = (RefDevState*)state;
    if (!lsm_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;
    const LsmInputs* in = (const LsmInputs*)inputs_void;
    LsmOutputs* out = (LsmOutputs*)outputs_void;
    if (run->num_ops > 0 && (!in || !in->ops)) return cudaErrorInvalidValue;
    if (!out->counts || !out->read_hash || !out->write_hash || !out->compaction_hash ||
        !out->wal_hash || !out->lsm_state_hash || !out->snapshot_hash) return cudaErrorInvalidValue;

    ref_step_kernel<<<1,1,0,stream>>>(*st, run->num_ops > 0 ? in->ops : nullptr, run->num_ops,
                                      workspace, workspace_bytes,
                                      out->counts, out->read_hash, out->write_hash,
                                      out->compaction_hash, out->wal_hash, out->lsm_state_hash,
                                      out->snapshot_hash);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    RefDevState* st = (RefDevState*)state;
    ref_reset_kernel<<<1,1,0,stream>>>(*st);
    return cudaPeekAtLastError();
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    RefDevState* st = (RefDevState*)state;
    ref_free_state(st);
    free(st);
}
