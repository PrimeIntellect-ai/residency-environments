// file: lsm_wal_compaction_oracle.hpp

#ifndef LSM_WAL_COMPACTION_ORACLE_HPP_
#define LSM_WAL_COMPACTION_ORACLE_HPP_

#include "lsm_wal_compaction_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

/*
ORACLE: independent host ground-truth model for lsm_wal_compaction.

Disambiguations (most-specific deterministic interpretations):
  D1. WAL record offset_in_segment = 0-based index within its segment,
      assigned as records are appended.
  D2. "current WAL segment" = active segment with the largest wal_id
      (the last in the wal_id-sorted active list).
  D3. A WAL segment is full when it already holds wal_segment_record_cap
      records.
  D4. CHECKPOINT_WAL archives a segment iff it is NON-EMPTY and its maximum
      record seq <= durable_flush_seq. Empty segments are never archived.
  D5. Memtable canonical lookup order (descending seq) equals reverse append
      order, since seq is strictly increasing across appends.
  D6. File create_event_seq = the event_seq counter value sampled at the
      instant of file creation (each created file consumes one event_seq via
      ++event_seq, matching the global emission counter).
  D7. event_seq is a single global monotone counter incremented (pre-inc,
      so first emitted event has event_seq 1) for every emitted hashed event
      across write/compaction/wal/snapshot/crash streams, in emission order.
  D8. For GET, "first record with matching key and seq <= read_seq" — within
      a single source the entries are scanned in descending seq; the first
      such entry whose seq <= read_seq wins for that source, and sources are
      tried in the contract order; the first source that yields a record wins.
  D9. Tombstone-drop deeper-coverage test uses the post-merge file layout
      (input files already conceptually removed) of levels level+2 ..
      level_count-1.
  D10. COMPACT input/output OOM check: a successful compaction that would
       make level+1 exceed its max_files_per_level emits COMPACT_OOM and
       mutates nothing.
  D11. If CRASH_RECOVER truncation deletes every active WAL segment, create
       one fresh empty segment with wal_id = ++next_wal_id (mirroring
       CHECKPOINT_WAL), so later writes always have a current segment.
*/

struct LsmWalRecord {
    uint64_t wal_id = 0;
    uint64_t offset = 0;
    uint64_t seq = 0;
    int32_t kind = 0;
    uint64_t key = 0;
    int64_t value = 0;
};

struct LsmWalSegment {
    uint64_t wal_id = 0;
    std::vector<LsmWalRecord> records;
};

struct LsmMemRecord {
    uint64_t seq = 0;
    int32_t kind = 0;
    uint64_t key = 0;
    int64_t value = 0;
};

struct LsmEntry {
    uint64_t key = 0;
    uint64_t seq = 0;
    int32_t kind = 0;
    int64_t value = 0;
};

struct LsmFile {
    uint64_t file_id = 0;
    int32_t level = 0;
    uint64_t min_key = 0;
    uint64_t max_key = 0;
    uint64_t max_seq = 0;
    uint64_t create_event_seq = 0;
    std::vector<LsmEntry> entries;  // sorted by key asc, then seq desc
};

struct LsmSnapshot {
    uint64_t snapshot_id = 0;
    uint64_t read_seq = 0;
};

struct LsmExpected {
    LsmCounts counts{};
    uint64_t read_hash = 0;
    uint64_t write_hash = 0;
    uint64_t compaction_hash = 0;
    uint64_t wal_hash = 0;
    uint64_t lsm_state_hash = 0;
    uint64_t snapshot_hash = 0;
};

static inline uint64_t lsm_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void lsm_fnv_u64(uint64_t* h, uint64_t v) {
    uint64_t hh = *h;
    for (int i = 0; i < 8; ++i) {
        hh = lsm_fnv_byte(hh, static_cast<uint8_t>(v & 0xFF));
        v >>= 8;
    }
    *h = hh;
}

static inline void lsm_fnv_u32(uint64_t* h, uint32_t v) {
    uint64_t hh = *h;
    for (int i = 0; i < 4; ++i) {
        hh = lsm_fnv_byte(hh, static_cast<uint8_t>(v & 0xFF));
        v >>= 8;
    }
    *h = hh;
}

static inline void lsm_fnv_u8(uint64_t* h, uint8_t v) {
    *h = lsm_fnv_byte(*h, v);
}

static inline void lsm_fnv_i64(uint64_t* h, int64_t v) {
    lsm_fnv_u64(h, static_cast<uint64_t>(v));
}

#define LSM_FNV_BASE 1469598103934665603ULL
#define LSM_U64_MAX 0xFFFFFFFFFFFFFFFFULL

struct LsmOracleState {
    LsmProblemSpec spec{};

    uint64_t next_seq = 1;
    uint64_t next_file_id = 1;
    uint64_t next_wal_id = 1;
    uint64_t event_seq = 0;

    uint64_t durable_flush_seq = 0;

    std::vector<LsmWalSegment> wal;             // active segments, sorted by wal_id
    std::vector<LsmMemRecord> memtable;         // append order
    std::vector<std::vector<LsmFile>> levels;   // per level, file list
    std::vector<LsmSnapshot> snapshots;         // sorted by snapshot_id

    // streaming hash accumulators (updated as events are emitted, never reset
    // between steps; counts and hashes are cumulative).
    uint64_t read_hash = LSM_FNV_BASE;
    uint64_t write_hash = LSM_FNV_BASE;
    uint64_t compaction_hash = LSM_FNV_BASE;

    LsmCounts counts{};

    void init(const LsmProblemSpec& s) {
        spec = s;
        reset();
    }

    void reset() {
        next_seq = 1;
        next_file_id = 1;
        next_wal_id = 1;
        event_seq = 0;
        durable_flush_seq = 0;
        wal.clear();
        memtable.clear();
        levels.assign((size_t)spec.level_count, {});
        snapshots.clear();
        read_hash = LSM_FNV_BASE;
        write_hash = LSM_FNV_BASE;
        compaction_hash = LSM_FNV_BASE;
        counts = LsmCounts{};
        // Initial single empty WAL segment with wal_id = 1.
        LsmWalSegment seg;
        seg.wal_id = 1;
        wal.push_back(seg);
    }

    uint64_t next_event() { return ++event_seq; }

    LsmWalSegment* current_segment() {
        if (wal.empty()) return nullptr;
        return &wal.back();  // largest wal_id is last (D2)
    }

    int find_snapshot(uint64_t sid) const {
        for (size_t i = 0; i < snapshots.size(); ++i) {
            if (snapshots[i].snapshot_id == sid) return (int)i;
        }
        return -1;
    }

    // ---- write_hash emission ----
    void emit_write(int ev_kind, uint64_t seq_or_max, uint64_t wal_id,
                    uint64_t offset_or_max, uint64_t key_or_max,
                    int kind_or_255, int64_t value, uint32_t op_index) {
        uint64_t es = next_event();
        lsm_fnv_u8(&write_hash, (uint8_t)ev_kind);
        lsm_fnv_u64(&write_hash, es);
        lsm_fnv_u32(&write_hash, op_index);
        lsm_fnv_u64(&write_hash, seq_or_max);
        lsm_fnv_u64(&write_hash, wal_id);
        lsm_fnv_u64(&write_hash, offset_or_max);
        lsm_fnv_u64(&write_hash, key_or_max);
        lsm_fnv_u8(&write_hash, (uint8_t)kind_or_255);
        lsm_fnv_i64(&write_hash, value);
    }

    // ---- read_hash emission ----
    void emit_read(uint64_t read_id, uint32_t op_index, uint64_t key,
                   uint64_t snapshot_id, uint64_t read_seq, int found,
                   int64_t value, int source_kind, uint64_t src_file,
                   uint64_t src_seq) {
        lsm_fnv_u64(&read_hash, read_id);
        lsm_fnv_u32(&read_hash, op_index);
        lsm_fnv_u64(&read_hash, key);
        lsm_fnv_u64(&read_hash, snapshot_id);
        lsm_fnv_u64(&read_hash, read_seq);
        lsm_fnv_u8(&read_hash, (uint8_t)found);
        lsm_fnv_i64(&read_hash, value);
        lsm_fnv_u8(&read_hash, (uint8_t)source_kind);
        lsm_fnv_u64(&read_hash, src_file);
        lsm_fnv_u64(&read_hash, src_seq);
    }

    // ---- compaction_hash emission ----
    void emit_compact_file(int ev_kind, uint32_t op_index, int32_t level,
                           uint64_t file_id, uint64_t min_key, uint64_t max_key,
                           uint64_t entry_count, uint64_t max_seq) {
        uint64_t es = next_event();
        lsm_fnv_u8(&compaction_hash, (uint8_t)ev_kind);
        lsm_fnv_u64(&compaction_hash, es);
        lsm_fnv_u32(&compaction_hash, op_index);
        lsm_fnv_u32(&compaction_hash, (uint32_t)level);
        lsm_fnv_u64(&compaction_hash, file_id);
        lsm_fnv_u64(&compaction_hash, min_key);
        lsm_fnv_u64(&compaction_hash, max_key);
        lsm_fnv_u64(&compaction_hash, entry_count);
        lsm_fnv_u64(&compaction_hash, max_seq);
    }

    void emit_compact_version(int ev_kind, uint32_t op_index, uint64_t key,
                              uint64_t seq, int kind, int64_t value) {
        uint64_t es = next_event();
        lsm_fnv_u8(&compaction_hash, (uint8_t)ev_kind);
        lsm_fnv_u64(&compaction_hash, es);
        lsm_fnv_u32(&compaction_hash, op_index);
        lsm_fnv_u64(&compaction_hash, key);
        lsm_fnv_u64(&compaction_hash, seq);
        lsm_fnv_u8(&compaction_hash, (uint8_t)kind);
        lsm_fnv_i64(&compaction_hash, value);
    }

    // ============ OPERATIONS ============

    bool memtable_full() const {
        return (int)memtable.size() >= spec.memtable_record_cap;
    }

    void do_write(int kind, uint64_t key, int64_t value, uint32_t op_index) {
        // 1. memtable-full stall (before WAL).
        if (memtable_full()) {
            counts.write_stall += 1;
            emit_write(LSM_EV_WRITE_STALL, LSM_U64_MAX, LSM_U64_MAX, LSM_U64_MAX,
                       key, (kind == LSM_KIND_DEL) ? LSM_KIND_DEL : LSM_KIND_PUT,
                       value, op_index);
            return;
        }
        // 2. WAL roll / OOM.
        LsmWalSegment* seg = current_segment();
        if ((int)seg->records.size() >= spec.wal_segment_record_cap) {
            if ((int)wal.size() >= spec.max_wal_segments) {
                counts.write_oom += 1;
                emit_write(LSM_EV_WRITE_OOM, LSM_U64_MAX, LSM_U64_MAX, LSM_U64_MAX,
                           key, (kind == LSM_KIND_DEL) ? LSM_KIND_DEL : LSM_KIND_PUT,
                           value, op_index);
                return;
            }
            LsmWalSegment ns;
            ns.wal_id = ++next_wal_id;
            wal.push_back(ns);
            seg = &wal.back();
            counts.wal_rolls += 1;
            emit_write(LSM_EV_WAL_ROLL, LSM_U64_MAX, ns.wal_id, LSM_U64_MAX,
                       LSM_U64_MAX, 255, 0, op_index);
        }
        // 3. assign seq, append WAL + memtable.
        uint64_t seq = next_seq++;
        LsmWalRecord wr;
        wr.wal_id = seg->wal_id;
        wr.offset = (uint64_t)seg->records.size();
        wr.seq = seq;
        wr.kind = kind;
        wr.key = key;
        wr.value = (kind == LSM_KIND_DEL) ? 0 : value;
        seg->records.push_back(wr);

        LsmMemRecord mr;
        mr.seq = seq;
        mr.kind = kind;
        mr.key = key;
        mr.value = wr.value;
        memtable.push_back(mr);

        if (kind == LSM_KIND_DEL) {
            counts.del_ok += 1;
            emit_write(LSM_EV_DEL_OK, seq, wr.wal_id, wr.offset, key,
                       LSM_KIND_DEL, wr.value, op_index);
        } else {
            counts.put_ok += 1;
            emit_write(LSM_EV_PUT_OK, seq, wr.wal_id, wr.offset, key,
                       LSM_KIND_PUT, wr.value, op_index);
        }
    }

    void do_get(uint64_t read_id, uint64_t key, uint64_t snapshot_id,
                uint32_t op_index) {
        uint64_t read_seq;
        if (snapshot_id == 0) {
            read_seq = next_seq - 1;
        } else {
            int si = find_snapshot(snapshot_id);
            if (si < 0) {
                counts.invalid_count += 1;
                return;  // invalid GET: no read event emitted
            }
            read_seq = snapshots[(size_t)si].read_seq;
        }

        int found = -1;          // -1 = not yet decided; result follows source
        int result_found = 0;    // 0 missing, 1 found
        int64_t result_value = 0;
        int source_kind = LSM_SRC_NONE;
        uint64_t src_file = LSM_U64_MAX;
        uint64_t src_seq = LSM_U64_MAX;

        // Source 1: memtable by descending seq (reverse append order).
        for (int i = (int)memtable.size() - 1; i >= 0 && found < 0; --i) {
            const LsmMemRecord& m = memtable[(size_t)i];
            if (m.key == key && m.seq <= read_seq) {
                found = 1;
                source_kind = LSM_SRC_MEMTABLE;
                src_file = LSM_U64_MAX;
                src_seq = m.seq;
                result_found = (m.kind == LSM_KIND_PUT) ? 1 : 0;
                result_value = (m.kind == LSM_KIND_PUT) ? m.value : 0;
            }
        }

        // Source 2: L0 files by descending file_id; within file desc seq.
        if (found < 0) {
            std::vector<const LsmFile*> l0;
            for (const LsmFile& f : levels[0]) l0.push_back(&f);
            std::sort(l0.begin(), l0.end(),
                      [](const LsmFile* a, const LsmFile* b) {
                          return a->file_id > b->file_id;
                      });
            for (const LsmFile* f : l0) {
                if (found >= 0) break;
                // entries sorted key asc, seq desc; scan matching key desc seq.
                for (const LsmEntry& e : f->entries) {
                    if (e.key != key) continue;
                    if (e.seq <= read_seq) {
                        found = 1;
                        source_kind = LSM_SRC_L0_FILE;
                        src_file = f->file_id;
                        src_seq = e.seq;
                        result_found = (e.kind == LSM_KIND_PUT) ? 1 : 0;
                        result_value = (e.kind == LSM_KIND_PUT) ? e.value : 0;
                        break;
                    }
                }
            }
        }

        // Source 3: levels 1..level_count-1 ascending; unique covering file.
        for (int lvl = 1; lvl < spec.level_count && found < 0; ++lvl) {
            const LsmFile* cover = nullptr;
            for (const LsmFile& f : levels[(size_t)lvl]) {
                if (key >= f.min_key && key <= f.max_key) {
                    cover = &f;
                    break;
                }
            }
            if (!cover) continue;
            for (const LsmEntry& e : cover->entries) {
                if (e.key != key) continue;
                if (e.seq <= read_seq) {
                    found = 1;
                    source_kind = LSM_SRC_LEVEL_FILE;
                    src_file = cover->file_id;
                    src_seq = e.seq;
                    result_found = (e.kind == LSM_KIND_PUT) ? 1 : 0;
                    result_value = (e.kind == LSM_KIND_PUT) ? e.value : 0;
                    break;
                }
            }
        }

        if (found < 0) {
            result_found = 0;
            result_value = 0;
            source_kind = LSM_SRC_NONE;
            src_file = LSM_U64_MAX;
            src_seq = LSM_U64_MAX;
        }

        if (result_found) counts.get_found += 1;
        else counts.get_missing += 1;

        emit_read(read_id, op_index, key, snapshot_id, read_seq, result_found,
                  result_value, source_kind, src_file, src_seq);
    }

    void do_open_snapshot(uint64_t sid, uint32_t op_index) {
        (void)op_index;
        if (sid == 0 || find_snapshot(sid) >= 0 ||
            (int)snapshots.size() >= spec.max_snapshots) {
            counts.invalid_count += 1;
            return;
        }
        LsmSnapshot s;
        s.snapshot_id = sid;
        s.read_seq = next_seq - 1;
        snapshots.push_back(s);
        std::sort(snapshots.begin(), snapshots.end(),
                  [](const LsmSnapshot& a, const LsmSnapshot& b) {
                      return a.snapshot_id < b.snapshot_id;
                  });
        counts.snapshot_opened += 1;
    }

    void do_release_snapshot(uint64_t sid, uint32_t op_index) {
        (void)op_index;
        int si = find_snapshot(sid);
        if (si < 0) {
            counts.invalid_count += 1;
            return;
        }
        snapshots.erase(snapshots.begin() + si);
        counts.snapshot_released += 1;
    }

    void do_flush(uint32_t op_index) {
        (void)op_index;
        if (memtable.empty()) {
            counts.flush_empty += 1;
            return;
        }
        if ((int)levels[0].size() >= spec.max_files_per_level[0]) {
            counts.flush_oom += 1;
            return;
        }
        LsmFile f;
        f.file_id = next_file_id++;
        f.level = 0;
        f.create_event_seq = next_event();
        // entries: all memtable records, sorted key asc then seq desc.
        for (const LsmMemRecord& m : memtable) {
            LsmEntry e;
            e.key = m.key;
            e.seq = m.seq;
            e.kind = m.kind;
            e.value = m.value;
            f.entries.push_back(e);
        }
        std::sort(f.entries.begin(), f.entries.end(),
                  [](const LsmEntry& a, const LsmEntry& b) {
                      if (a.key != b.key) return a.key < b.key;
                      return a.seq > b.seq;
                  });
        uint64_t mn = f.entries[0].key, mx = f.entries[0].key, ms = 0;
        for (const LsmEntry& e : f.entries) {
            if (e.key < mn) mn = e.key;
            if (e.key > mx) mx = e.key;
            if (e.seq > ms) ms = e.seq;
        }
        f.min_key = mn;
        f.max_key = mx;
        f.max_seq = ms;
        levels[0].push_back(f);
        memtable.clear();
        if (f.max_seq > durable_flush_seq) durable_flush_seq = f.max_seq;
        counts.flush_files += 1;
    }

    static bool ranges_overlap(uint64_t amin, uint64_t amax,
                               uint64_t bmin, uint64_t bmax) {
        return amin <= bmax && bmin <= amax;
    }

    void do_compact(int level, int max_primary_files, uint32_t op_index) {
        if (level < 0 || level >= spec.level_count - 1 || max_primary_files == 0) {
            counts.invalid_count += 1;
            return;
        }
        if (levels[(size_t)level].empty()) {
            counts.compact_empty += 1;
            return;
        }

        // ---- Input selection ----
        std::vector<int> primary_idx;   // indices into levels[level]
        uint64_t rmin = 0, rmax = 0;

        if (level == 0) {
            // primary = smallest file_id.
            int best = 0;
            for (int i = 1; i < (int)levels[0].size(); ++i) {
                if (levels[0][(size_t)i].file_id < levels[0][(size_t)best].file_id) best = i;
            }
            primary_idx.push_back(best);
            rmin = levels[0][(size_t)best].min_key;
            rmax = levels[0][(size_t)best].max_key;
            // transitive overlap expansion, bounded by max_primary_files,
            // adding the smallest-file-id overlapping L0 file each round.
            std::vector<char> selected(levels[0].size(), 0);
            selected[(size_t)best] = 1;
            while ((int)primary_idx.size() < max_primary_files) {
                int cand = -1;
                for (int i = 0; i < (int)levels[0].size(); ++i) {
                    if (selected[(size_t)i]) continue;
                    if (ranges_overlap(levels[0][(size_t)i].min_key,
                                       levels[0][(size_t)i].max_key, rmin, rmax)) {
                        if (cand < 0 ||
                            levels[0][(size_t)i].file_id < levels[0][(size_t)cand].file_id) {
                            cand = i;
                        }
                    }
                }
                if (cand < 0) break;
                selected[(size_t)cand] = 1;
                primary_idx.push_back(cand);
                if (levels[0][(size_t)cand].min_key < rmin) rmin = levels[0][(size_t)cand].min_key;
                if (levels[0][(size_t)cand].max_key > rmax) rmax = levels[0][(size_t)cand].max_key;
            }
        } else {
            // primary = smallest min_key; tie smallest file_id.
            int best = 0;
            for (int i = 1; i < (int)levels[(size_t)level].size(); ++i) {
                const LsmFile& a = levels[(size_t)level][(size_t)i];
                const LsmFile& b = levels[(size_t)level][(size_t)best];
                if (a.min_key < b.min_key ||
                    (a.min_key == b.min_key && a.file_id < b.file_id)) {
                    best = i;
                }
            }
            primary_idx.push_back(best);
            rmin = levels[(size_t)level][(size_t)best].min_key;
            rmax = levels[(size_t)level][(size_t)best].max_key;
        }

        // overlapping files in level+1.
        std::vector<int> next_idx;
        for (int i = 0; i < (int)levels[(size_t)level + 1].size(); ++i) {
            const LsmFile& f = levels[(size_t)level + 1][(size_t)i];
            if (ranges_overlap(f.min_key, f.max_key, rmin, rmax)) next_idx.push_back(i);
        }

        // ---- Merge selected entries ----
        // For duplicate (key,seq): keep entry from lower numeric level; tie ->
        // larger file_id.
        struct Merged {
            uint64_t key, seq;
            int32_t kind;
            int64_t value;
            int32_t src_level;
            uint64_t src_file_id;
        };
        std::vector<Merged> merged;
        auto add_file = [&](const LsmFile& f) {
            for (const LsmEntry& e : f.entries) {
                merged.push_back(Merged{e.key, e.seq, e.kind, e.value, f.level, f.file_id});
            }
        };
        for (int idx : primary_idx) add_file(levels[(size_t)level][(size_t)idx]);
        for (int idx : next_idx) add_file(levels[(size_t)level + 1][(size_t)idx]);

        // dedup (key,seq): lower level wins, tie larger file_id.
        std::sort(merged.begin(), merged.end(), [](const Merged& a, const Merged& b) {
            if (a.key != b.key) return a.key < b.key;
            if (a.seq != b.seq) return a.seq > b.seq;  // desc seq within key
            if (a.src_level != b.src_level) return a.src_level < b.src_level;
            return a.src_file_id > b.src_file_id;
        });
        std::vector<Merged> dedup;
        for (size_t i = 0; i < merged.size(); ++i) {
            if (i > 0 && merged[i].key == merged[i - 1].key &&
                merged[i].seq == merged[i - 1].seq) {
                continue;  // first kept is the winner due to sort order
            }
            dedup.push_back(merged[i]);
        }

        // oldest snapshot seq.
        uint64_t oldest = next_seq - 1;
        if (!snapshots.empty()) {
            oldest = snapshots[0].read_seq;
            for (const LsmSnapshot& s : snapshots) {
                if (s.read_seq < oldest) oldest = s.read_seq;
            }
        }

        // Deeper-level coverage test for a key: any file in levels
        // level+2 .. level_count-1 whose range covers key (D9: current layout,
        // inputs are from level and level+1 only so deeper levels unaffected).
        auto deeper_covers = [&](uint64_t key) -> bool {
            for (int dl = level + 2; dl < spec.level_count; ++dl) {
                for (const LsmFile& f : levels[(size_t)dl]) {
                    if (key >= f.min_key && key <= f.max_key) return true;
                }
            }
            return false;
        };

        // ---- Retention per key (key ascending, seq descending) ----
        // dedup is already key asc, seq desc.
        std::vector<LsmEntry> retained;
        struct DropEv { int kind; uint64_t key, seq; int32_t entry_kind; int64_t value; };
        std::vector<DropEv> drops;  // in retention scan order

        size_t i = 0;
        while (i < dedup.size()) {
            size_t j = i;
            uint64_t key = dedup[i].key;
            while (j < dedup.size() && dedup[j].key == key) ++j;
            // group [i,j) for this key, already seq desc.
            bool deeper = deeper_covers(key);
            bool stopped = false;  // once we drop the surviving-record chain
            bool kept_floor = false;  // already kept the highest seq <= oldest
            for (size_t r = i; r < j; ++r) {
                const Merged& m = dedup[r];
                if (m.seq > oldest) {
                    // keep all records strictly above oldest snapshot seq.
                    retained.push_back(LsmEntry{m.key, m.seq, m.kind, m.value});
                    continue;
                }
                // m.seq <= oldest.
                if (stopped) {
                    // already dropped tombstone+older for this key.
                    drops.push_back(DropEv{LSM_EV_DROP_OLD_VERSION, m.key, m.seq, m.kind, m.value});
                    continue;
                }
                if (!kept_floor) {
                    // This is the highest-seq record with seq <= oldest.
                    if (m.kind == LSM_KIND_DEL && !deeper) {
                        // drop this tombstone and all older records.
                        drops.push_back(DropEv{LSM_EV_DROP_TOMBSTONE, m.key, m.seq, m.kind, m.value});
                        stopped = true;
                    } else {
                        retained.push_back(LsmEntry{m.key, m.seq, m.kind, m.value});
                        kept_floor = true;
                    }
                } else {
                    // older record beyond the kept floor.
                    drops.push_back(DropEv{LSM_EV_DROP_OLD_VERSION, m.key, m.seq, m.kind, m.value});
                }
            }
            i = j;
        }

        // ---- Partition retained into new files (entry order, sst_record_cap) ----
        int out_file_count = 0;
        if (!retained.empty()) {
            out_file_count = (int)((retained.size() + (size_t)spec.sst_record_cap - 1) /
                                   (size_t)spec.sst_record_cap);
        }

        // ---- OOM check (D10): would level+1 exceed cap? ----
        int remaining_next = (int)levels[(size_t)level + 1].size() - (int)next_idx.size();
        if (remaining_next + out_file_count > spec.max_files_per_level[(size_t)level + 1]) {
            counts.compact_oom += 1;
            return;  // no mutation, no events
        }

        // ---- Build output files (assign file_ids now, in order) ----
        std::vector<LsmFile> out_files;
        {
            size_t pos = 0;
            while (pos < retained.size()) {
                size_t end = std::min(pos + (size_t)spec.sst_record_cap, retained.size());
                LsmFile nf;
                nf.file_id = next_file_id++;
                nf.level = level + 1;
                nf.create_event_seq = 0;  // set at emission time
                uint64_t mn = retained[pos].key, mx = retained[pos].key, ms = 0;
                for (size_t p = pos; p < end; ++p) {
                    nf.entries.push_back(retained[p]);
                    if (retained[p].key < mn) mn = retained[p].key;
                    if (retained[p].key > mx) mx = retained[p].key;
                    if (retained[p].seq > ms) ms = retained[p].seq;
                }
                nf.min_key = mn;
                nf.max_key = mx;
                nf.max_seq = ms;
                out_files.push_back(nf);
                pos = end;
            }
        }

        // ---- Collect obsolete input descriptors before removal ----
        struct ObsRec { int32_t level; uint64_t file_id, min_key, max_key, entry_count, max_seq; };
        std::vector<ObsRec> obs;
        for (int idx : primary_idx) {
            const LsmFile& f = levels[(size_t)level][(size_t)idx];
            obs.push_back(ObsRec{f.level, f.file_id, f.min_key, f.max_key,
                                 (uint64_t)f.entries.size(), f.max_seq});
        }
        for (int idx : next_idx) {
            const LsmFile& f = levels[(size_t)level + 1][(size_t)idx];
            obs.push_back(ObsRec{f.level, f.file_id, f.min_key, f.max_key,
                                 (uint64_t)f.entries.size(), f.max_seq});
        }
        std::sort(obs.begin(), obs.end(), [](const ObsRec& a, const ObsRec& b) {
            if (a.level != b.level) return a.level < b.level;
            return a.file_id < b.file_id;
        });

        // ---- Mutate: remove inputs, add outputs ----
        {
            std::vector<char> rm0(levels[(size_t)level].size(), 0);
            for (int idx : primary_idx) rm0[(size_t)idx] = 1;
            std::vector<LsmFile> keep0;
            for (int k = 0; k < (int)levels[(size_t)level].size(); ++k) {
                if (!rm0[(size_t)k]) keep0.push_back(levels[(size_t)level][(size_t)k]);
            }
            levels[(size_t)level] = keep0;

            std::vector<char> rm1(levels[(size_t)level + 1].size(), 0);
            for (int idx : next_idx) rm1[(size_t)idx] = 1;
            std::vector<LsmFile> keep1;
            for (int k = 0; k < (int)levels[(size_t)level + 1].size(); ++k) {
                if (!rm1[(size_t)k]) keep1.push_back(levels[(size_t)level + 1][(size_t)k]);
            }
            for (LsmFile& nf : out_files) keep1.push_back(nf);
            levels[(size_t)level + 1] = keep1;
        }

        // ---- Emit events: obsolete first, then outputs, then drops ----
        for (const ObsRec& o : obs) {
            emit_compact_file(LSM_EV_COMPACT_INPUT_OBSOLETE, op_index, o.level,
                              o.file_id, o.min_key, o.max_key, o.entry_count, o.max_seq);
        }
        // outputs in new file_id order (they are appended in id order already).
        // Need to find them back in levels[level+1] by file_id; we still have
        // out_files in order with assigned ids.
        for (LsmFile& nf : out_files) {
            // set create_event_seq from the emission event.
            uint64_t es = ++event_seq;  // pre-inc, matches next_event semantics
            // re-implement emission to also capture es into the file record.
            lsm_fnv_u8(&compaction_hash, (uint8_t)LSM_EV_COMPACT_OUTPUT_FILE);
            lsm_fnv_u64(&compaction_hash, es);
            lsm_fnv_u32(&compaction_hash, op_index);
            lsm_fnv_u32(&compaction_hash, (uint32_t)nf.level);
            lsm_fnv_u64(&compaction_hash, nf.file_id);
            lsm_fnv_u64(&compaction_hash, nf.min_key);
            lsm_fnv_u64(&compaction_hash, nf.max_key);
            lsm_fnv_u64(&compaction_hash, (uint64_t)nf.entries.size());
            lsm_fnv_u64(&compaction_hash, nf.max_seq);
            // write create_event_seq into the live file in levels[level+1].
            for (LsmFile& live : levels[(size_t)level + 1]) {
                if (live.file_id == nf.file_id) { live.create_event_seq = es; break; }
            }
        }
        for (const DropEv& d : drops) {
            emit_compact_version(d.kind, op_index, d.key, d.seq, d.entry_kind, d.value);
        }

        // ---- Counts ----
        counts.compact_input_files += (int64_t)obs.size();
        counts.compact_output_files += (int64_t)out_files.size();
        counts.obsolete_files += (int64_t)obs.size();
        for (const DropEv& d : drops) {
            if (d.kind == LSM_EV_DROP_OLD_VERSION) counts.versions_dropped += 1;
            else counts.tombstones_dropped += 1;
        }
    }

    void do_checkpoint_wal(uint32_t op_index) {
        (void)op_index;
        // archive every NON-EMPTY segment whose max record seq <= durable_flush_seq.
        std::vector<LsmWalSegment> keep;
        std::vector<uint64_t> archived_ids;
        for (LsmWalSegment& seg : wal) {
            if (seg.records.empty()) {
                keep.push_back(seg);
                continue;
            }
            uint64_t mx = 0;
            for (const LsmWalRecord& r : seg.records) {
                if (r.seq > mx) mx = r.seq;
            }
            if (mx <= durable_flush_seq) {
                archived_ids.push_back(seg.wal_id);
            } else {
                keep.push_back(seg);
            }
        }
        if (archived_ids.empty()) return;  // nothing archived
        wal = keep;
        if (wal.empty()) {
            LsmWalSegment ns;
            ns.wal_id = ++next_wal_id;
            wal.push_back(ns);
        }
        std::sort(archived_ids.begin(), archived_ids.end());
        for (uint64_t id : archived_ids) {
            (void)id;
            counts.wal_archived += 1;
        }
    }

    void do_crash_recover(uint64_t cut_wal_id, uint64_t cut_offset, uint32_t op_index) {
        (void)op_index;
        // Discard memtable.
        memtable.clear();
        // Truncate WAL: keep records with (wal_id, offset) <= (cut_wal_id, cut_offset).
        std::vector<LsmWalSegment> new_wal;
        for (LsmWalSegment& seg : wal) {
            LsmWalSegment ns;
            ns.wal_id = seg.wal_id;
            for (const LsmWalRecord& r : seg.records) {
                if (r.wal_id < cut_wal_id ||
                    (r.wal_id == cut_wal_id && r.offset <= cut_offset)) {
                    ns.records.push_back(r);
                }
            }
            // Empty segments after the cut are deleted; a segment is "after the
            // cut" if its wal_id > cut_wal_id (no surviving records possible) OR
            // it ends up empty due to truncation.
            if (!ns.records.empty()) {
                new_wal.push_back(ns);
            }
        }
        wal = new_wal;
        // D11: crash truncation may delete every active segment. Subsequent
        // writes require a current segment, so (mirroring CHECKPOINT_WAL's
        // last-segment rule) create one fresh empty segment with
        // wal_id = ++next_wal_id when none remain.
        if (wal.empty()) {
            LsmWalSegment ns;
            ns.wal_id = ++next_wal_id;
            wal.push_back(ns);
        }

        // Drop all open snapshots (ascending id) -> SNAPSHOT_DROP_CRASH events.
        std::sort(snapshots.begin(), snapshots.end(),
                  [](const LsmSnapshot& a, const LsmSnapshot& b) {
                      return a.snapshot_id < b.snapshot_id;
                  });
        for (const LsmSnapshot& s : snapshots) {
            (void)s;
            counts.snapshots_dropped_by_crash += 1;
        }
        snapshots.clear();

        // Collect surviving WAL records in WAL order (by wal_id asc, offset asc).
        std::vector<LsmWalRecord> survivors;
        for (const LsmWalSegment& seg : wal) {
            for (const LsmWalRecord& r : seg.records) survivors.push_back(r);
        }
        std::sort(survivors.begin(), survivors.end(),
                  [](const LsmWalRecord& a, const LsmWalRecord& b) {
                      if (a.wal_id != b.wal_id) return a.wal_id < b.wal_id;
                      return a.offset < b.offset;
                  });

        // Replay records with seq > durable_flush_seq into memtable in WAL order.
        bool stalled = false;
        for (const LsmWalRecord& r : survivors) {
            if (r.seq <= durable_flush_seq) continue;
            if (stalled) {
                // remaining unreplayed surviving records with seq > durable are discarded
                counts.recover_stalled_records += 1;
                continue;
            }
            if ((int)memtable.size() >= spec.memtable_record_cap) {
                stalled = true;
                counts.recover_stalled_records += 1;  // this record is discarded
                continue;
            }
            LsmMemRecord mr;
            mr.seq = r.seq;
            mr.kind = r.kind;
            mr.key = r.key;
            mr.value = r.value;
            memtable.push_back(mr);
            counts.recovered_records += 1;
        }

        // next_seq = 1 + max_seq_present over SSTs, memtable, surviving WAL.
        uint64_t maxseq = 0;
        bool any = false;
        for (int l = 0; l < spec.level_count; ++l) {
            for (const LsmFile& f : levels[(size_t)l]) {
                for (const LsmEntry& e : f.entries) {
                    if (!any || e.seq > maxseq) { maxseq = e.seq; any = true; }
                }
            }
        }
        for (const LsmMemRecord& m : memtable) {
            if (!any || m.seq > maxseq) { maxseq = m.seq; any = true; }
        }
        for (const LsmWalRecord& r : survivors) {
            if (!any || r.seq > maxseq) { maxseq = r.seq; any = true; }
        }
        next_seq = any ? (maxseq + 1) : 1;
    }

    void apply_op(const LsmOp& op, uint32_t op_index) {
        switch (op.kind) {
            case LSM_OP_PUT:
                do_write(LSM_KIND_PUT, (uint64_t)(uint32_t)op.i_a, op.value, op_index);
                break;
            case LSM_OP_DEL:
                do_write(LSM_KIND_DEL, (uint64_t)(uint32_t)op.i_a, 0, op_index);
                break;
            case LSM_OP_GET:
                do_get(op.u_a /*read_id*/, (uint64_t)(uint32_t)op.i_a /*key*/,
                       op.u_b /*snapshot_id*/, op_index);
                break;
            case LSM_OP_OPEN_SNAPSHOT:
                do_open_snapshot(op.u_a, op_index);
                break;
            case LSM_OP_RELEASE_SNAPSHOT:
                do_release_snapshot(op.u_a, op_index);
                break;
            case LSM_OP_FLUSH:
                do_flush(op_index);
                break;
            case LSM_OP_COMPACT:
                do_compact(op.i_a, op.i_b, op_index);
                break;
            case LSM_OP_CHECKPOINT_WAL:
                do_checkpoint_wal(op_index);
                break;
            case LSM_OP_CRASH_RECOVER:
                do_crash_recover((uint64_t)(uint32_t)op.i_a, (uint64_t)(uint32_t)op.i_b, op_index);
                break;
            default:
                counts.invalid_count += 1;
                break;
        }
    }

    // ---- structural hashes (recomputed each step over current state) ----
    uint64_t compute_wal_hash() const {
        uint64_t h = LSM_FNV_BASE;
        // active segments by wal_id then offset.
        std::vector<const LsmWalSegment*> segs;
        for (const LsmWalSegment& s : wal) segs.push_back(&s);
        std::sort(segs.begin(), segs.end(),
                  [](const LsmWalSegment* a, const LsmWalSegment* b) {
                      return a->wal_id < b->wal_id;
                  });
        for (const LsmWalSegment* s : segs) {
            std::vector<const LsmWalRecord*> recs;
            for (const LsmWalRecord& r : s->records) recs.push_back(&r);
            std::sort(recs.begin(), recs.end(),
                      [](const LsmWalRecord* a, const LsmWalRecord* b) {
                          return a->offset < b->offset;
                      });
            for (const LsmWalRecord* r : recs) {
                lsm_fnv_u64(&h, r->wal_id);
                lsm_fnv_u64(&h, r->offset);
                lsm_fnv_u64(&h, r->seq);
                lsm_fnv_u8(&h, (uint8_t)r->kind);
                lsm_fnv_u64(&h, r->key);
                lsm_fnv_i64(&h, r->value);
            }
        }
        return h;
    }

    uint64_t compute_lsm_state_hash() const {
        uint64_t h = LSM_FNV_BASE;
        // memtable by descending seq.
        std::vector<const LsmMemRecord*> mem;
        for (const LsmMemRecord& m : memtable) mem.push_back(&m);
        std::sort(mem.begin(), mem.end(),
                  [](const LsmMemRecord* a, const LsmMemRecord* b) {
                      return a->seq > b->seq;
                  });
        for (const LsmMemRecord* m : mem) {
            lsm_fnv_u8(&h, 0);  // source_kind = 0
            lsm_fnv_u64(&h, m->seq);
            lsm_fnv_u8(&h, (uint8_t)m->kind);
            lsm_fnv_u64(&h, m->key);
            lsm_fnv_i64(&h, m->value);
        }
        // files by level asc; within level file_id asc; within file entry order.
        for (int l = 0; l < spec.level_count; ++l) {
            std::vector<const LsmFile*> fs;
            for (const LsmFile& f : levels[(size_t)l]) fs.push_back(&f);
            std::sort(fs.begin(), fs.end(),
                      [](const LsmFile* a, const LsmFile* b) {
                          return a->file_id < b->file_id;
                      });
            for (const LsmFile* f : fs) {
                for (size_t ei = 0; ei < f->entries.size(); ++ei) {
                    const LsmEntry& e = f->entries[ei];
                    lsm_fnv_u8(&h, 1);  // source_kind = 1
                    lsm_fnv_u32(&h, (uint32_t)l);
                    lsm_fnv_u64(&h, f->file_id);
                    lsm_fnv_u64(&h, f->min_key);
                    lsm_fnv_u64(&h, f->max_key);
                    lsm_fnv_u64(&h, f->max_seq);
                    lsm_fnv_u64(&h, (uint64_t)ei);
                    lsm_fnv_u64(&h, e.key);
                    lsm_fnv_u64(&h, e.seq);
                    lsm_fnv_u8(&h, (uint8_t)e.kind);
                    lsm_fnv_i64(&h, e.value);
                }
            }
        }
        return h;
    }

    uint64_t compute_snapshot_hash() const {
        uint64_t h = LSM_FNV_BASE;
        std::vector<const LsmSnapshot*> ss;
        for (const LsmSnapshot& s : snapshots) ss.push_back(&s);
        std::sort(ss.begin(), ss.end(),
                  [](const LsmSnapshot* a, const LsmSnapshot* b) {
                      return a->snapshot_id < b->snapshot_id;
                  });
        for (const LsmSnapshot* s : ss) {
            lsm_fnv_u64(&h, s->snapshot_id);
            lsm_fnv_u64(&h, s->read_seq);
        }
        return h;
    }

    void step_once(const LsmRunSpec& run, const LsmOp* ops, LsmExpected* exp) {
        for (int i = 0; i < run.num_ops; ++i) {
            apply_op(ops[i], (uint32_t)i);
        }
        exp->counts = counts;
        exp->read_hash = read_hash;
        exp->write_hash = write_hash;
        exp->compaction_hash = compaction_hash;
        exp->wal_hash = compute_wal_hash();
        exp->lsm_state_hash = compute_lsm_state_hash();
        exp->snapshot_hash = compute_snapshot_hash();
    }
};

static inline bool lsm_check_outputs(
    const LsmExpected& e,
    const LsmCounts& gc,
    uint64_t g_read, uint64_t g_write, uint64_t g_compact,
    uint64_t g_wal, uint64_t g_state, uint64_t g_snap,
    std::string* error) {
#define LSM_CK_COUNT(field)                                                  \
    if (gc.field != e.counts.field) {                                        \
        if (error) {                                                         \
            std::ostringstream oss;                                          \
            oss << "count " #field " mismatch: got " << gc.field             \
                << ", expected " << e.counts.field;                          \
            *error = oss.str();                                              \
        }                                                                    \
        return false;                                                        \
    }
    LSM_CK_COUNT(put_ok);
    LSM_CK_COUNT(del_ok);
    LSM_CK_COUNT(write_stall);
    LSM_CK_COUNT(write_oom);
    LSM_CK_COUNT(wal_rolls);
    LSM_CK_COUNT(get_found);
    LSM_CK_COUNT(get_missing);
    LSM_CK_COUNT(snapshot_opened);
    LSM_CK_COUNT(snapshot_released);
    LSM_CK_COUNT(snapshots_dropped_by_crash);
    LSM_CK_COUNT(flush_files);
    LSM_CK_COUNT(flush_empty);
    LSM_CK_COUNT(flush_oom);
    LSM_CK_COUNT(compact_empty);
    LSM_CK_COUNT(compact_oom);
    LSM_CK_COUNT(compact_input_files);
    LSM_CK_COUNT(compact_output_files);
    LSM_CK_COUNT(versions_dropped);
    LSM_CK_COUNT(tombstones_dropped);
    LSM_CK_COUNT(obsolete_files);
    LSM_CK_COUNT(wal_archived);
    LSM_CK_COUNT(recovered_records);
    LSM_CK_COUNT(recover_stalled_records);
    LSM_CK_COUNT(invalid_count);
#undef LSM_CK_COUNT

#define LSM_CK_HASH(name, got, expv)                                         \
    if ((got) != (expv)) {                                                   \
        if (error) {                                                         \
            std::ostringstream oss;                                          \
            oss << name " mismatch: got 0x" << std::hex << (got)             \
                << ", expected 0x" << (expv);                               \
            *error = oss.str();                                              \
        }                                                                    \
        return false;                                                        \
    }
    LSM_CK_HASH("read_hash", g_read, e.read_hash);
    LSM_CK_HASH("write_hash", g_write, e.write_hash);
    LSM_CK_HASH("compaction_hash", g_compact, e.compaction_hash);
    LSM_CK_HASH("wal_hash", g_wal, e.wal_hash);
    LSM_CK_HASH("lsm_state_hash", g_state, e.lsm_state_hash);
    LSM_CK_HASH("snapshot_hash", g_snap, e.snapshot_hash);
#undef LSM_CK_HASH

    return true;
}

#endif  // LSM_WAL_COMPACTION_ORACLE_HPP_
