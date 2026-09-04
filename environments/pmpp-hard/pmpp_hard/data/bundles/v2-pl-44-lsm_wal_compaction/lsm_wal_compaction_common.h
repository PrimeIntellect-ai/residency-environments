// file: lsm_wal_compaction_common.h

#ifndef LSM_WAL_COMPACTION_COMMON_H_
#define LSM_WAL_COMPACTION_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define LSM_ABI_VERSION 1

// Capacity bounds (compile-time maxima used to size persistent buffers).
#define LSM_MIN_LEVELS 2
#define LSM_MAX_LEVELS 6
#define LSM_MAX_FILES_PER_LEVEL 64
#define LSM_MAX_TOTAL_FILES 256
#define LSM_MAX_ENTRIES_PER_FILE 512
#define LSM_MAX_TOTAL_ENTRIES 16384
#define LSM_MAX_MEMTABLE 512
#define LSM_MAX_WAL_SEGMENTS 32
#define LSM_MAX_WAL_RECORDS 8192
#define LSM_MAX_SNAPSHOTS 16
#define LSM_MAX_OPS 256
#define LSM_MAX_STEPS 64

// Operation kinds.
#define LSM_OP_PUT 0
#define LSM_OP_DEL 1
#define LSM_OP_GET 2
#define LSM_OP_OPEN_SNAPSHOT 3
#define LSM_OP_RELEASE_SNAPSHOT 4
#define LSM_OP_FLUSH 5
#define LSM_OP_COMPACT 6
#define LSM_OP_CHECKPOINT_WAL 7
#define LSM_OP_CRASH_RECOVER 8

// Record / entry kinds.
#define LSM_KIND_PUT 0
#define LSM_KIND_DEL 1

// GET source kinds.
#define LSM_SRC_MEMTABLE 0
#define LSM_SRC_L0_FILE 1
#define LSM_SRC_LEVEL_FILE 2
#define LSM_SRC_NONE 3

// write_hash event kinds.
#define LSM_EV_PUT_OK 0
#define LSM_EV_DEL_OK 1
#define LSM_EV_WAL_ROLL 2
#define LSM_EV_WRITE_STALL 3
#define LSM_EV_WRITE_OOM 4

// compaction_hash event kinds.
#define LSM_EV_COMPACT_INPUT_OBSOLETE 0
#define LSM_EV_COMPACT_OUTPUT_FILE 1
#define LSM_EV_DROP_OLD_VERSION 2
#define LSM_EV_DROP_TOMBSTONE 3

/*
CONTRACT: lsm_wal_compaction

A persistent LSM key/value engine with:
  - Write-ahead log (WAL) segments and records.
  - An append-list memtable (canonical lookup order = descending seq).
  - Leveled SSTables (L0 overlapping, L>=1 non-overlapping).
  - Snapshot-pinned read sequences.
  - Leveled compaction with overlap selection.
  - Tombstone GC that drops a tombstone only when no deeper level can hold
    older data for that key, and only below the oldest live snapshot.
  - WAL checkpoint (archiving segments fully covered by durable_flush_seq).
  - Crash recovery: WAL truncation, durable-flush filtering, snapshot drop,
    bounded memtable replay, and next_seq reconstruction.

All graded outputs are EXACT integers (counts + FNV-1a-64 checksums).

Each step is a batch of ops applied in order. After each step, the harness
reads cumulative counts plus six structural checksums.

ABI: solution_init may allocate persistent device state. solution_run may
launch kernels and use the provided workspace, but may not call
cudaMalloc/cudaFree.

The full, normative per-op semantics and EXACT output serialization are
inlined below in the CONTRACT section. Nothing is deferred to any external
file; this header is self-contained and complete for grading purposes.
*/

// === CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section specifies EVERY enforced rule. It is the single source of
// truth; an implementation that reproduces exactly the integers and byte
// streams below passes the grader (24-entry LsmCounts + 6 FNV-1a-64 hashes).
//
// --------------------------------------------------------------------------
// 0. GLOBAL MODEL / PERSISTENT STATE
// --------------------------------------------------------------------------
// State is created at solution_init and persists across steps. solution_reset
// restores the initial state. The model is single-threaded / deterministic.
//
// Scalar counters (all start at the RESET values shown):
//   next_seq          = 1   // monotone per-record sequence; assigned post-inc
//                           //   (first written record gets seq=1).
//   next_file_id      = 1   // SSTable file id; assigned post-inc.
//   next_wal_id       = 1   // WAL segment id; the INITIAL segment already
//                           //   uses wal_id=1 (NOT consumed from next_wal_id;
//                           //   next ++next_wal_id yields 2). See below.
//   event_seq         = 0   // global monotone event counter; PRE-increment,
//                           //   so the first emitted event has event_seq=1.
//                           //   Incremented once per emitted hashed event that
//                           //   carries an event_seq field (write events,
//                           //   compaction file events, compaction version
//                           //   (drop) events, and each FLUSH/COMPACT file
//                           //   creation). Shared across all streams in
//                           //   emission order.
//   durable_flush_seq = 0   // max seq made durable by FLUSH; raised by flush.
//
// Containers:
//   wal[]       : ACTIVE WAL segments, kept sorted ascending by wal_id. Each
//                 segment = { wal_id, records[] }. Each record =
//                 { wal_id, offset, seq, kind, key, value }.
//   memtable[]  : records in APPEND order. record = { seq, kind, key, value }.
//   levels[L][] : per-level file lists, L in [0, level_count). Each file =
//                 { file_id, level, min_key, max_key, max_seq,
//                   create_event_seq, entries[] }, entries sorted key asc then
//                   seq desc.
//   snapshots[] : kept sorted ascending by snapshot_id. snapshot =
//                 { snapshot_id, read_seq }.
//
// Streaming hash accumulators (persist across steps, NEVER reset between
// steps; cumulative like counts): read_hash, write_hash, compaction_hash,
// each initialized to FNV offset basis 1469598103934665603 at reset.
//
// RESET creates exactly one empty WAL segment with wal_id = 1 (memtable empty,
// all levels empty, no snapshots, all counts 0). next_wal_id stays 1, so the
// SECOND segment ever created (via WAL roll / checkpoint-empty / crash-empty)
// gets wal_id = ++next_wal_id = 2.
//
// --------------------------------------------------------------------------
// 1. FNV-1a-64 PRIMITIVE (confirmed in reference)
// --------------------------------------------------------------------------
//   offset basis (init) = 1469598103934665603  (0x14650FB0739D0383)
//   prime               = 1099511628211        (0x100000001B3)
//   fnv_byte(h,b): h ^= (uint64_t)b; h *= prime; return h;
//   u64 folded LITTLE-ENDIAN: for i in 0..7: byte = v&0xFF; h=fnv_byte; v>>=8.
//   u32 folded LITTLE-ENDIAN: for i in 0..3 (4 bytes), same.
//   u8 : single fnv_byte.
//   i64: reinterpret as uint64_t (two's complement) then fold as u64.
// All multi-byte values below use these exact widths and little-endian fold.
//
// --------------------------------------------------------------------------
// 2. OPERATION DISPATch (LsmOp field mapping)
// --------------------------------------------------------------------------
//   LSM_OP_PUT(0):  key=(uint64_t)(uint32_t)i_a, value=value -> do_write PUT.
//   LSM_OP_DEL(1):  key=(uint64_t)(uint32_t)i_a, value forced 0 -> write DEL.
//   LSM_OP_GET(2):  read_id=u_a, key=(uint64_t)(uint32_t)i_a, snapshot_id=u_b.
//   LSM_OP_OPEN_SNAPSHOT(3):    snapshot_id=u_a.
//   LSM_OP_RELEASE_SNAPSHOT(4): snapshot_id=u_a.
//   LSM_OP_FLUSH(5):            no args.
//   LSM_OP_COMPACT(6):  level=i_a, max_primary_files=i_b.
//   LSM_OP_CHECKPOINT_WAL(7):   no args.
//   LSM_OP_CRASH_RECOVER(8): cut_wal_id=(uint64_t)(uint32_t)i_a,
//                            cut_offset=(uint64_t)(uint32_t)i_b.
//   any other kind: counts.invalid_count += 1 (no event).
//   op_index = the 0-based index i of the op within the step's ops[] array
//   (uint32_t). op_index is fed into write/read/compaction events as shown.
//
// --------------------------------------------------------------------------
// 3. WRITE (PUT / DEL) -- do_write(kind,key,value,op_index)
// --------------------------------------------------------------------------
// "current WAL segment" = the active segment with the largest wal_id = wal.back()
// (segments kept in ascending wal_id order). A segment is FULL when it already
// holds wal_segment_record_cap records. WAL offset = 0-based index within its
// segment in append order.
//
// Order of checks (FIRST match wins, then return):
//  (a) memtable-full stall: if memtable.size() >= memtable_record_cap:
//        counts.write_stall += 1;
//        emit_write(WRITE_STALL, seq=MAX, wal_id=MAX, off=MAX, key=key,
//                   kind255=(kind==DEL?DEL:PUT), value=value, op_index);
//        return.  (Note: kind255 here is the ORIGINAL op kind 0/1, value is the
//        original value; checked BEFORE the WAL, so no WAL/memtable mutation.)
//  (b) WAL roll / OOM: if current_segment.records.size() >= wal_segment_record_cap:
//        if wal.size() >= max_wal_segments:
//          counts.write_oom += 1;
//          emit_write(WRITE_OOM, MAX, MAX, MAX, key, (kind==DEL?DEL:PUT),
//                     value, op_index); return.
//        else roll: create new segment with wal_id = ++next_wal_id, append to
//          wal (now current); counts.wal_rolls += 1;
//          emit_write(WAL_ROLL, seq=MAX, wal_id=new_wal_id, off=MAX, key=MAX,
//                     kind255=255, value=0, op_index);  // then continue (c).
//  (c) commit: seq = next_seq++ (post-inc). storeval = (kind==DEL)?0:value.
//        append WAL record { wal_id=cur.wal_id, offset=cur.records.size()
//        (BEFORE push), seq, kind(=0/1), key, value=storeval }.
//        append memtable record { seq, kind, key, value=storeval }.
//        if kind==DEL: counts.del_ok += 1;
//          emit_write(DEL_OK, seq, cur.wal_id, offset, key, DEL, storeval, op_index);
//        else:        counts.put_ok += 1;
//          emit_write(PUT_OK, seq, cur.wal_id, offset, key, PUT, storeval, op_index);
//        (offset = the just-assigned record offset = records.size() before push.)
//
// write_hash event field order (emit_write), ALL into write_hash:
//   u8  ev_kind            // LSM_EV_* (PUT_OK0/DEL_OK1/WAL_ROLL2/STALL3/OOM4)
//   u64 event_seq          // pre-incremented global counter
//   u32 op_index
//   u64 seq_or_max         // seq, or 0xFFFFFFFFFFFFFFFF sentinel
//   u64 wal_id             // or MAX sentinel
//   u64 offset_or_max      // or MAX sentinel
//   u64 key_or_max         // or MAX sentinel
//   u8  kind_or_255        // PUT=0 / DEL=1 / 255 for WAL_ROLL
//   i64 value
//
// --------------------------------------------------------------------------
// 4. GET -- do_get(read_id,key,snapshot_id,op_index)
// --------------------------------------------------------------------------
// read_seq selection:
//   if snapshot_id == 0: read_seq = next_seq - 1 (latest committed seq).
//   else: si = find snapshot by id; if not found -> counts.invalid_count += 1
//         and RETURN with NO read event emitted. Otherwise read_seq =
//         snapshot.read_seq.
//
// Source search order; the FIRST source that yields a matching record wins;
// within a source the FIRST entry with key match and seq <= read_seq wins
// (scanning in descending seq), and that record's kind determines found/value:
//   Source 1 MEMTABLE (src_kind=0=LSM_SRC_MEMTABLE, src_file=MAX):
//     scan memtable in REVERSE append order (== descending seq); first record
//     with key==key && seq<=read_seq. src_seq = that record's seq.
//   Source 2 L0 files (src_kind=1=LSM_SRC_L0_FILE, src_file=file_id):
//     iterate level-0 files by DESCENDING file_id; within a file scan entries
//     (sorted key asc, seq desc) for first key==key with seq<=read_seq.
//   Source 3 levels 1..level_count-1 ASCENDING (src_kind=2=LSM_SRC_LEVEL_FILE):
//     for each level pick the UNIQUE covering file (first file with
//     key>=min_key && key<=max_key; non-overlapping at L>=1); scan its entries
//     for first key==key with seq<=read_seq.
//   If no source matches: result_found=0, value=0, src_kind=3=LSM_SRC_NONE,
//     src_file=MAX, src_seq=MAX.
//   For a matched record: result_found = (entry.kind==PUT)?1:0;
//     result_value = (entry.kind==PUT)?entry.value:0  (tombstone => missing,0).
//
// Counts: if result_found -> counts.get_found += 1 else counts.get_missing += 1.
// (A valid GET that resolves to a tombstone counts as get_missing.)
//
// read_hash event field order (emit_read), ALL into read_hash:
//   u64 read_id
//   u32 op_index
//   u64 key
//   u64 snapshot_id        // the raw op snapshot_id (0 for latest)
//   u64 read_seq
//   u8  found              // result_found (0/1)
//   i64 value              // result_value
//   u8  source_kind        // 0 mem / 1 L0 / 2 level / 3 none
//   u64 src_file           // file_id, or MAX for memtable/none
//   u64 src_seq            // matched seq, or MAX for none
// (No event_seq in read events. read events do NOT consume event_seq.)
//
// --------------------------------------------------------------------------
// 5. SNAPSHOTS
// --------------------------------------------------------------------------
// OPEN(sid): invalid (counts.invalid_count += 1, no other effect) iff sid==0,
//   OR a snapshot with that id already exists, OR snapshots.size() >=
//   max_snapshots. Otherwise create { snapshot_id=sid, read_seq=next_seq-1 },
//   insert keeping snapshots sorted ascending by snapshot_id;
//   counts.snapshot_opened += 1.
// RELEASE(sid): if not found -> counts.invalid_count += 1. Else erase it;
//   counts.snapshot_released += 1.
// (No hashed events for snapshot open/release; only counts and the structural
//  snapshot_hash reflect them.)
//
// --------------------------------------------------------------------------
// 6. FLUSH -- do_flush(op_index)
// --------------------------------------------------------------------------
//   if memtable empty: counts.flush_empty += 1; return.
//   else if levels[0].size() >= max_files_per_level[0]:
//       counts.flush_oom += 1; return.
//   else create file f: file_id = next_file_id++; level = 0;
//       create_event_seq = next_event() (PRE-inc event_seq; consumes an
//       event_seq but emits NO hashed event into compaction_hash).
//       entries = ALL memtable records (key,seq,kind,value), then SORT entries
//       by key ascending, seq descending.
//       min_key = min entry.key; max_key = max entry.key; max_seq = max
//       entry.seq (all over the entries).
//       Append f to levels[0]; clear memtable.
//       if f.max_seq > durable_flush_seq: durable_flush_seq = f.max_seq.
//       counts.flush_files += 1.
//   FLUSH emits no write/read/compaction hashed event; it only consumes one
//   event_seq (the create_event_seq) and mutates structural state.
//
// --------------------------------------------------------------------------
// 7. COMPACT -- do_compact(level, max_primary_files, op_index)
// --------------------------------------------------------------------------
// Validity:
//   if level<0 || level>=level_count-1 || max_primary_files==0:
//       counts.invalid_count += 1; return.
//   if levels[level] empty: counts.compact_empty += 1; return.
//
// Input selection:
//   if level==0:
//     primary set starts with the L0 file of SMALLEST file_id. range
//     [rmin,rmax] = that file's [min_key,max_key]. Then transitively expand,
//     bounded by max_primary_files, each round adding the SMALLEST-file_id not
//     yet selected L0 file whose [min_key,max_key] OVERLAPS [rmin,rmax]
//     (overlap = amin<=bmax && bmin<=amax), extending rmin/rmax by the added
//     file. Stop when no overlapping candidate remains or size==max_primary.
//   else (level>=1):
//     primary = the single file with smallest min_key; tie -> smallest file_id.
//     [rmin,rmax] = that file's range. (max_primary_files not used to expand.)
//   next set = ALL files in level+1 whose range OVERLAPS [rmin,rmax].
//
// Merge: gather every entry of every primary file then every next file, each
//   tagged with (src_level=file.level, src_file_id=file.file_id). SORT merged
//   by: key ASC, then seq DESC, then src_level ASC, then src_file_id DESC.
//   DEDUP on (key,seq): keep the FIRST in sort order (=> lower level wins; tie
//   => larger file_id). Result `dedup` is key asc, seq desc, unique (key,seq).
//
// oldest snapshot seq: oldest = next_seq - 1 if no snapshots; else the MINIMUM
//   read_seq over all snapshots.
//
// Deeper coverage test for a key (D9, post-merge layout; inputs are only from
//   level and level+1 so deeper levels are unchanged): true iff any file in
//   levels level+2 .. level_count-1 has key>=min_key && key<=max_key.
//
// Retention, scanning dedup grouped by key (key ascending), within a key
//   group in seq-descending order. Per key: deeper = deeper_covers(key);
//   stopped=false; kept_floor=false. For each record m in the group:
//     - if m.seq > oldest: RETAIN m (keep every version above oldest snap).
//     - else (m.seq <= oldest):
//         if stopped: DROP m as DROP_OLD_VERSION(2).
//         else if !kept_floor:  // highest-seq record with seq<=oldest
//             if m.kind==DEL && !deeper:
//                 DROP m as DROP_TOMBSTONE(3); stopped=true.
//             else: RETAIN m; kept_floor=true.
//         else: DROP m as DROP_OLD_VERSION(2).
//   Retained records preserve dedup (key asc, seq desc) order. Drops are
//   recorded in this scan order (for later emission).
//
// out_file_count = ceil(retained.size / sst_record_cap) (0 if none).
//
// OOM check (D10) -- performed BEFORE any output file is built and BEFORE any
//   next_file_id is consumed: remaining_next = (#files in level+1) - (#next
//   inputs). if remaining_next + out_file_count > max_files_per_level[level+1]:
//       counts.compact_oom += 1; return  -- NO mutation, NO events,
//   next_file_id unchanged (no file_ids consumed on OOM).
//
// Output files (built only if OOM check passes): partition `retained` IN ORDER
//   into chunks of at most sst_record_cap entries. Each output file: file_id =
//   next_file_id++ (assigned in chunk order), level = level+1, entries = its
//   chunk, min_key/max_key = min/max entry.key of chunk, max_seq = max
//   entry.seq of chunk.
//
// Mutation + emission ORDER (all into compaction_hash unless noted):
//   1. Collect obsolete input descriptors (level, file_id, min_key, max_key,
//      entry_count, max_seq) for every primary and next input; SORT them by
//      level ASC then file_id ASC.
//   2. Remove input files from levels[level] and levels[level+1]; append the
//      output files to levels[level+1].
//   3. Emit COMPACT_INPUT_OBSOLETE(ev=0) for each obsolete descriptor in the
//      sorted order.
//   4. Emit COMPACT_OUTPUT_FILE(ev=1) for each output file in new-file_id
//      order; the create_event_seq of each output file = the event_seq sampled
//      at its emission (these output events DO consume event_seq).
//   5. Emit drop events in retention scan order: DROP_OLD_VERSION(2) or
//      DROP_TOMBSTONE(3).
//
// Counts:
//   counts.compact_input_files  += (#primary + #next inputs) = #obsolete.
//   counts.compact_output_files += out_file_count.
//   counts.obsolete_files       += #obsolete.
//   per drop: DROP_OLD_VERSION -> counts.versions_dropped += 1;
//             DROP_TOMBSTONE   -> counts.tombstones_dropped += 1.
//
// compaction_hash COMPACT_INPUT_OBSOLETE / COMPACT_OUTPUT_FILE field order
// (emit_compact_file), ALL into compaction_hash:
//   u8  ev_kind            // 0 = INPUT_OBSOLETE, 1 = OUTPUT_FILE
//   u64 event_seq
//   u32 op_index
//   u32 level              // file's level (input: source level; output: lvl+1)
//   u64 file_id
//   u64 min_key
//   u64 max_key
//   u64 entry_count
//   u64 max_seq
//
// compaction_hash DROP_OLD_VERSION / DROP_TOMBSTONE field order
// (emit_compact_version), ALL into compaction_hash:
//   u8  ev_kind            // 2 = DROP_OLD_VERSION, 3 = DROP_TOMBSTONE
//   u64 event_seq
//   u32 op_index
//   u64 key
//   u64 seq
//   u8  kind               // the dropped entry's kind (0 PUT / 1 DEL)
//   i64 value              // the dropped entry's value
//
// --------------------------------------------------------------------------
// 8. CHECKPOINT_WAL -- do_checkpoint_wal(op_index)
// --------------------------------------------------------------------------
// Archive (delete) every NON-EMPTY active segment whose MAX record seq <=
// durable_flush_seq (D4). Empty segments are NEVER archived (always kept).
//   - If nothing archived: return with no change.
//   - Else keep the surviving segments (order preserved, still wal_id asc);
//     if NO segments remain, create one fresh empty segment with
//     wal_id = ++next_wal_id.
//   - counts.wal_archived += (number of archived segments).
// No hashed event; only counts + structural wal_hash change.
//
// --------------------------------------------------------------------------
// 9. CRASH_RECOVER -- do_crash_recover(cut_wal_id, cut_offset, op_index)
// --------------------------------------------------------------------------
//  1. Discard memtable (clear).
//  2. Truncate WAL: keep ONLY records with (wal_id < cut_wal_id) OR
//     (wal_id == cut_wal_id && offset <= cut_offset). Segments that become
//     empty after truncation are DELETED (not kept). Surviving segments keep
//     their wal_id and ascending order.
//  3. D11: if NO segment remains, create one fresh empty segment with
//     wal_id = ++next_wal_id (so later writes have a current segment).
//  4. Drop ALL open snapshots: counts.snapshots_dropped_by_crash += (#open
//     snapshots); then clear snapshots. (Iterated ascending id; count only.)
//  5. Replay: collect surviving WAL records in WAL order (wal_id ASC, then
//     offset ASC). For each record with seq > durable_flush_seq, in that order:
//       - if already stalled: counts.recover_stalled_records += 1 (discard).
//       - else if memtable.size() >= memtable_record_cap: set stalled=true and
//         counts.recover_stalled_records += 1 (this record discarded too).
//       - else append { seq, kind, key, value } to memtable;
//         counts.recovered_records += 1.
//     Records with seq <= durable_flush_seq are skipped silently (not counted).
//  6. next_seq = 1 + (max seq present over all SSTable entries, current
//     memtable records, and surviving WAL records); if none present, next_seq=1.
// No hashed event; only counts + structural wal_hash/lsm_state_hash/
// snapshot_hash change.
//
// --------------------------------------------------------------------------
// 10. STRUCTURAL HASHES (recomputed from scratch each step over CURRENT state)
// --------------------------------------------------------------------------
// Each starts h = FNV offset basis 1469598103934665603.
//
// wal_hash: iterate ACTIVE segments by wal_id ASCENDING; within a segment
//   iterate records by offset ASCENDING. For each record fold:
//     u64 wal_id, u64 offset, u64 seq, u8 kind, u64 key, i64 value.
//
// lsm_state_hash:
//   (a) memtable by DESCENDING seq; per record fold:
//         u8 0 (constant tag), u64 seq, u8 kind, u64 key, i64 value.
//   (b) files: for level = 0 .. level_count-1 ASC; within a level by file_id
//       ASCENDING; within a file iterate entries in stored order (index ei
//       from 0). For EACH entry fold:
//         u8 1 (constant tag), u32 level, u64 file_id, u64 file.min_key,
//         u64 file.max_key, u64 file.max_seq, u64 ei (entry index within file),
//         u64 entry.key, u64 entry.seq, u8 entry.kind, i64 entry.value.
//       (Files with zero entries contribute nothing.)
//
// snapshot_hash: snapshots by snapshot_id ASCENDING; per snapshot fold:
//   u64 snapshot_id, u64 read_seq.
//
// --------------------------------------------------------------------------
// 11. GRADED COUNT VECTOR -- LsmCounts (24 entries, struct field order ==
//     index order; the harness compares ALL 24). Definitive mapping:
// --------------------------------------------------------------------------
//   [ 0] put_ok                     : successful PUT commits (do_write PUT (c)).
//   [ 1] del_ok                     : successful DEL commits (do_write DEL (c)).
//   [ 2] write_stall                : writes rejected by memtable-full (3a).
//   [ 3] write_oom                  : writes rejected by WAL OOM (3b, no roll).
//   [ 4] wal_rolls                  : WAL segment rolls (3b roll branch).
//   [ 5] get_found                  : GETs resolving to a live PUT.
//   [ 6] get_missing                : GETs resolving to missing/tombstone
//                                     (valid GET only; invalid GET not counted).
//   [ 7] snapshot_opened            : successful OPEN_SNAPSHOT.
//   [ 8] snapshot_released          : successful RELEASE_SNAPSHOT.
//   [ 9] snapshots_dropped_by_crash : snapshots dropped during CRASH_RECOVER.
//   [10] flush_files                : files created by FLUSH.
//   [11] flush_empty                : FLUSH on empty memtable.
//   [12] flush_oom                  : FLUSH rejected (L0 at cap).
//   [13] compact_empty              : COMPACT on empty source level.
//   [14] compact_oom                : COMPACT rejected by level+1 cap (D10).
//   [15] compact_input_files        : total input files consumed by compactions
//                                     (primary + next), summed.
//   [16] compact_output_files       : total output files produced, summed.
//   [17] versions_dropped           : DROP_OLD_VERSION events.
//   [18] tombstones_dropped         : DROP_TOMBSTONE events.
//   [19] obsolete_files             : input files made obsolete (== input_files).
//   [20] wal_archived               : WAL segments archived by CHECKPOINT_WAL.
//   [21] recovered_records          : WAL records replayed into memtable.
//   [22] recover_stalled_records    : WAL records NOT replayed (memtable full /
//                                     after stall) during recovery.
//   [23] invalid_count              : invalid ops -- unknown op kind, GET with
//                                     unknown snapshot, OPEN invalid (dup/0/cap),
//                                     RELEASE of unknown snapshot, COMPACT with
//                                     bad level or max_primary_files==0.
// All counts are CUMULATIVE across steps (never reset between steps).
//
// === END CONTRACT ===

struct alignas(8) LsmProblemSpec {
    int32_t abi_version;
    int32_t level_count;             // >= 2, <= LSM_MAX_LEVELS
    int32_t max_files_per_level[LSM_MAX_LEVELS];
    int32_t memtable_record_cap;     // <= LSM_MAX_MEMTABLE
    int32_t sst_record_cap;          // <= LSM_MAX_ENTRIES_PER_FILE
    int32_t wal_segment_record_cap;  // records per WAL segment
    int32_t max_wal_segments;        // <= LSM_MAX_WAL_SEGMENTS
    int32_t max_snapshots;           // <= LSM_MAX_SNAPSHOTS
    int32_t max_ops;                 // <= LSM_MAX_OPS
    int32_t max_steps;               // <= LSM_MAX_STEPS
    int32_t flags;
    int32_t reserved[3];
};

// One operation within a step.
struct alignas(8) LsmOp {
    int32_t kind;          // LSM_OP_*
    int32_t i_a;           // PUT/DEL/GET: key; COMPACT: level; CRASH: cut_wal_id
    int32_t i_b;           // COMPACT: max_primary_files; CRASH: cut_offset
    int32_t i_c;           // reserved
    int64_t value;         // PUT value (i64)
    uint64_t u_a;          // GET: read_id; OPEN/RELEASE: snapshot_id
    uint64_t u_b;          // GET: snapshot_id
};

struct alignas(8) LsmRunSpec {
    int32_t abi_version;
    int32_t num_ops;
    int32_t step_id;
    int32_t reserved[5];
};

struct alignas(8) LsmInputs {
    const LsmOp* ops;
};

// All outputs are cumulative state after applying the step's ops.
struct alignas(8) LsmCounts {
    int64_t put_ok;
    int64_t del_ok;
    int64_t write_stall;
    int64_t write_oom;
    int64_t wal_rolls;
    int64_t get_found;
    int64_t get_missing;
    int64_t snapshot_opened;
    int64_t snapshot_released;
    int64_t snapshots_dropped_by_crash;
    int64_t flush_files;
    int64_t flush_empty;
    int64_t flush_oom;
    int64_t compact_empty;
    int64_t compact_oom;
    int64_t compact_input_files;
    int64_t compact_output_files;
    int64_t versions_dropped;
    int64_t tombstones_dropped;
    int64_t obsolete_files;
    int64_t wal_archived;
    int64_t recovered_records;
    int64_t recover_stalled_records;
    int64_t invalid_count;
};

struct alignas(8) LsmOutputs {
    LsmCounts* counts;          // 1
    uint64_t* read_hash;        // 1
    uint64_t* write_hash;       // 1
    uint64_t* compaction_hash;  // 1
    uint64_t* wal_hash;         // 1
    uint64_t* lsm_state_hash;   // 1
    uint64_t* snapshot_hash;    // 1
};

static_assert(sizeof(LsmOp) == 40, "LsmOp layout drift");
static_assert(sizeof(LsmRunSpec) == 32, "LsmRunSpec layout drift");
static_assert(sizeof(LsmInputs) == 8, "LsmInputs layout drift");
static_assert(sizeof(LsmCounts) == 24 * 8, "LsmCounts layout drift");
static_assert(sizeof(LsmOutputs) == 56, "LsmOutputs layout drift");

static inline int lsm_validate_problem_spec(const LsmProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != LSM_ABI_VERSION) return 0;
    if (spec->level_count < LSM_MIN_LEVELS || spec->level_count > LSM_MAX_LEVELS) return 0;
    for (int l = 0; l < spec->level_count; ++l) {
        if (spec->max_files_per_level[l] < 1 ||
            spec->max_files_per_level[l] > LSM_MAX_FILES_PER_LEVEL) {
            return 0;
        }
    }
    if (spec->memtable_record_cap < 1 || spec->memtable_record_cap > LSM_MAX_MEMTABLE) return 0;
    if (spec->sst_record_cap < 1 || spec->sst_record_cap > LSM_MAX_ENTRIES_PER_FILE) return 0;
    if (spec->wal_segment_record_cap < 1) return 0;
    if (spec->max_wal_segments < 1 || spec->max_wal_segments > LSM_MAX_WAL_SEGMENTS) return 0;
    if (spec->max_snapshots < 1 || spec->max_snapshots > LSM_MAX_SNAPSHOTS) return 0;
    if (spec->max_ops < 1 || spec->max_ops > LSM_MAX_OPS) return 0;
    if (spec->max_steps < 1 || spec->max_steps > LSM_MAX_STEPS) return 0;
    return 1;
}

static inline int lsm_validate_run_spec(const LsmRunSpec* run, const LsmProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != LSM_ABI_VERSION) return 0;
    if (run->num_ops < 0 || run->num_ops > spec->max_ops) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const LsmProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const LsmProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const LsmRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // LSM_WAL_COMPACTION_COMMON_H_
