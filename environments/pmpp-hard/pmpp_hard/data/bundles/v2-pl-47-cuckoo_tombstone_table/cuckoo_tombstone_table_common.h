// file: cuckoo_tombstone_table_common.h

#ifndef CUCKOO_TOMBSTONE_TABLE_COMMON_H_
#define CUCKOO_TOMBSTONE_TABLE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define CKT_ABI_VERSION 1

// ---------------------------------------------------------------------------
// Bounds (used by validation; harness must respect them).
// ---------------------------------------------------------------------------
#define CKT_MIN_SLOT_COUNT 2
#define CKT_MAX_SLOT_COUNT 4096
#define CKT_MIN_PAGE_SIZE 1
#define CKT_MAX_PAGE_SIZE 4096
#define CKT_MIN_NEIGHBORHOOD 1
#define CKT_MAX_STASH 4096
#define CKT_MAX_TOMBSTONES 4096
#define CKT_MAX_OPS 8192
#define CKT_MAX_STEPS 64

// ---------------------------------------------------------------------------
// FNV-1a-64 constants.
// ---------------------------------------------------------------------------
#define CKT_FNV_OFFSET 1469598103934665603ULL
#define CKT_FNV_PRIME  1099511628211ULL

// ---------------------------------------------------------------------------
// Sentinels used inside hashed event records.
// ---------------------------------------------------------------------------
#define CKT_U64_MAX 0xFFFFFFFFFFFFFFFFULL
#define CKT_I64_MIN (-9223372036854775807LL - 1LL)
#define CKT_HK_NONE 255

// ---------------------------------------------------------------------------
// Slot states.
// ---------------------------------------------------------------------------
#define CKT_EMPTY 0
#define CKT_LIVE 1
#define CKT_TOMBSTONE 2

// Home kinds.
#define CKT_HOME0 0
#define CKT_HOME1 1

// Source kinds for GET_RESULT / read_hash.
#define CKT_SRC_TABLE 0
#define CKT_SRC_STASH 1
#define CKT_SRC_NONE 2

// ---------------------------------------------------------------------------
// Op encoding. Each op has an op_type and two 64-bit arguments.
//   GET              : a0 = read_id, a1 = key
//   PUT              : a0 = key,     a1 = value (interpreted as int64)
//   DELETE           : a0 = key
//   PIN_PAGE         : a0 = page
//   UNPIN_PAGE       : a0 = page
//   SWEEP_TOMBSTONES : a0 = limit
//   REPLAY_STASH     : a0 = limit
// ---------------------------------------------------------------------------
#define CKT_OP_GET 0
#define CKT_OP_PUT 1
#define CKT_OP_DELETE 2
#define CKT_OP_PIN_PAGE 3
#define CKT_OP_UNPIN_PAGE 4
#define CKT_OP_SWEEP_TOMBSTONES 5
#define CKT_OP_REPLAY_STASH 6

// ---------------------------------------------------------------------------
// Event kinds (emission order is hashed by op_event_hash).
// ---------------------------------------------------------------------------
#define CKT_EV_GET_RESULT 0
#define CKT_EV_UPDATE_EXISTING 1
#define CKT_EV_UPDATE_STASH 2
#define CKT_EV_RESURRECT_TOMBSTONE 3
#define CKT_EV_RELOCATE_SLOT 4
#define CKT_EV_REUSE_TOMBSTONE 5
#define CKT_EV_PUT_INSERT 6
#define CKT_EV_PUT_STASH 7
#define CKT_EV_PUT_OOM 8
#define CKT_EV_DELETE_TABLE 9
#define CKT_EV_DELETE_STASH 10
#define CKT_EV_DELETE_MISS 11
#define CKT_EV_TOMBSTONE_SWEEP 12
#define CKT_EV_STASH_REPLAY_OK 13
#define CKT_EV_PIN_PAGE_OK 14
#define CKT_EV_UNPIN_PAGE_OK 15
#define CKT_EV_INVALID 16

/*
CONTRACT: cuckoo_tombstone_table  (T47)

Tombstone-Resurrecting Paginated Hopscotch-Cuckoo Table.

PERSISTENT STATE (survives across run() steps; cleared by solution_reset):
  Per-slot SoA arrays of length slot_count:
    state[i]    : CKT_EMPTY | CKT_LIVE | CKT_TOMBSTONE
    key[i]      : u64
    val[i]      : i64    (LIVE value, or TOMBSTONE old_value)
    home_kind[i]: u8     CKT_HOME0 | CKT_HOME1 (selected_home_kind)
    home_slot[i]: u64    selected_home_slot
    iseq[i]     : u64    insert_seq
    aux[i]      : u64    version_seq (LIVE) or tomb_seq (TOMBSTONE)
  Per-page:
    pin_count[p]: u64    (n_pages = ceil(slot_count/page_size))
  Stash FIFO (kept ordered by insert_seq ascending):
    stash_key[], stash_val[] (i64), stash_iseq[], stash_vseq[]; stash_size
  Scalars:
    event_seq      : u64, starts 0
    insert_seq_next: u64, starts 1

DERIVED:
  page_id(slot)  = slot / page_size
  dist(home,slot)= (slot + slot_count - home) mod slot_count
  in neighborhood iff dist < neighborhood
  home0(key) = FNV1a64(seed0,key) mod slot_count
  raw1       = FNV1a64(seed1,key) mod slot_count
  home1(key) = raw1 if raw1 != home0 else (raw1+1) mod slot_count

  FNV1a64(seed,key): h = CKT_FNV_OFFSET; hash the 8 little-endian bytes of
  seed (u64), then the 8 little-endian bytes of key (u64); each byte:
  h ^= byte; h *= CKT_FNV_PRIME.

EVENT SEQUENCING:
  event_seq is a u64 counter starting at 0. Each emitted event pre-increments
  it: ev = ++event_seq, and that event's event_seq field is ev (first event
  has event_seq = 1). version_seq / tomb_seq assigned to a slot equal the
  event_seq of the emitting event. Counters wrap mod 2^64.

OPERATIONS: fully specified below in the normative DETERMINISM section. The
semantics are identical across all three implementations and exactly match
the field-by-field hashing described under OUTPUTS.

OUTPUTS per run() step (all exact integers; counts are for THIS run only):
  counts[0..16] (i32):
    0 get_found      1 get_missing    2 put_inserted   3 put_updated
    4 put_resurrected 5 put_stashed   6 put_oom         7 delete_table
    8 delete_stash   9 delete_miss   10 slot_relocations 11 tombstone_reused
   12 tombstone_swept 13 stash_replayed 14 page_pins     15 page_unpins
   16 invalid_count
  op_event_hash (u64): FNV1a64 over every emitted event record, in emission
    order. Each record hashes these fields, in order, as raw little-endian:
      event_kind:u8, event_seq:u64, op_index:u32, key_or_MAX:u64,
      slot_or_MAX:u64, home_kind_or_255:u8, home_slot_or_MAX:u64,
      value_or_INT64_MIN:i64, aux_u64:u64
    Starts at CKT_FNV_OFFSET. op_index = position of the op within this run.
  read_hash (u64): FNV1a64 over GET_RESULT records this run, each:
      read_id:u64, key:u64, found:u8, source_kind:u8, slot_or_MAX:u64,
      stash_insert_seq_or_MAX:u64, value_or_INT64_MIN:i64,
      version_seq_or_MAX:u64. Starts at CKT_FNV_OFFSET.
  slot_state_hash (u64): FNV1a64 over slots by slot index ascending. For each:
      EMPTY:     slot:u64, state:u8
      LIVE:      slot:u64, state:u8, key:u64, value:i64, home_kind:u8,
                 home_slot:u64, insert_seq:u64, version_seq:u64
      TOMBSTONE: slot:u64, state:u8, key:u64, old_value:i64, home_kind:u8,
                 home_slot:u64, insert_seq:u64, tomb_seq:u64
  stash_hash (u64): FNV1a64 over stash entries by insert_seq ascending, each:
      key:u64, value:i64, insert_seq:u64, version_seq:u64
  page_hash (u64): FNV1a64 over pages ascending, each: page:u64, pin_count:u64
  All four state hashes start at CKT_FNV_OFFSET.

RULES:
  - solution_init may allocate persistent state (cudaMalloc).
  - solution_run may NOT cudaMalloc/cudaFree; it may launch kernels and use
    the provided workspace.
  - All outputs are exact integers; no floats, no tolerance.

// === DETERMINISM & EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section INLINES the complete operational and serialization contract so
// that a solver reading ONLY this header can reproduce every graded checksum
// bit-for-bit. It is derived exactly from the reference implementation; nothing
// here is approximate. The single-thread reference executes ops strictly in
// input order (op_index = 0,1,2,... = position within this run()).
//
// -------------------------------------------------------------------------
// 0. FNV-1a-64 PRIMITIVES (all hashing, no exceptions)
// -------------------------------------------------------------------------
//   offset basis = CKT_FNV_OFFSET = 1469598103934665603 (0x14650FB0739D0383)
//   prime        = CKT_FNV_PRIME  = 1099511628211        (0x00000100000001B3)
//   per byte b:  h ^= (uint64_t)b;  h *= CKT_FNV_PRIME;  (mod 2^64)
//   A field of width W is hashed as its W raw IN-MEMORY little-endian bytes,
//   in increasing address order (byte 0 = least significant). u8 = 1 byte;
//   u32 = 4 bytes; u64 / i64 = 8 bytes (i64 is two's-complement reinterpreted
//   as the same 8 bytes). NO separators, NO length prefixes, NO alignment
//   padding are ever inserted between fields.
//   There are SIX independent hashes. The two stream hashes (op_event_hash,
//   read_hash) each START at CKT_FNV_OFFSET at the BEGINNING of EACH run()
//   step (they are NOT persistent across steps; the running h is reset to
//   the offset basis at step entry and accumulated over that step's events).
//   The three final-state hashes (slot_state_hash, stash_hash, page_hash) and
//   the seed-key helper each (re)start at CKT_FNV_OFFSET when computed.
//
//   FNV1a64(seed,key)  [used ONLY for home indices]:
//     h = CKT_FNV_OFFSET; hash 8 LE bytes of seed (u64); hash 8 LE bytes of
//     key (u64); return h. (This is a FRESH hash per call, not the running
//     stream.)
//
// -------------------------------------------------------------------------
// 1. HOME INDEX (bucket) FUNCTIONS
// -------------------------------------------------------------------------
//   home0(key) = FNV1a64(seed0,key) % slot_count
//   raw1       = FNV1a64(seed1,key) % slot_count
//   home1(key) = (raw1 != home0(key)) ? raw1 : (raw1 + 1) % slot_count
//   page_of(slot) = slot / page_size      (integer division)
//   dist(home,slot) = ( (int64_t)slot + slot_count - home ) % slot_count
//   "within neighborhood" iff dist(home,slot) < neighborhood.
//   All % here are non-negative because operands are normalized non-negative.
//
// -------------------------------------------------------------------------
// 2. EVENT SEQUENCING / SEQ NUMBERS
// -------------------------------------------------------------------------
//   event_seq (u64, persistent, starts 0) is pre-incremented for EVERY emitted
//   event: ev = ++event_seq (first event of the whole table life = 1). The
//   event's event_seq field IS ev. When a LIVE slot's version_seq (aux) or a
//   TOMBSTONE's tomb_seq (aux) is assigned "= ev", it equals that emitting
//   event's event_seq. insert_seq_next (u64, persistent, starts 1) yields a
//   fresh insert_seq via post-increment (new_iseq = insert_seq_next++) ONLY on
//   a genuinely new key (not update/resurrect). Both wrap mod 2^64. These
//   scalars PERSIST across run() steps; only solution_reset clears them
//   (event_seq=0, insert_seq_next=1, tomb_count=0, stash emptied, all slots
//   EMPTY with all fields 0, all pin_count 0).
//
// -------------------------------------------------------------------------
// 3. EVENT RECORD FIELD LAYOUT (op_event_hash) -- emission order hashed
// -------------------------------------------------------------------------
//   Every emitted event folds these 9 fields, in THIS order, raw LE:
//     event_kind : u8   (CKT_EV_* value)
//     event_seq  : u64  (= ev = ++event_seq)
//     op_index   : u32  (position of op in this run, 0-based)
//     key_f      : u64  (key, or CKT_U64_MAX when N/A)
//     slot_f     : u64  (slot index, or CKT_U64_MAX when N/A)
//     hk_f       : u8   (home_kind 0/1, or CKT_HK_NONE=255 when N/A)
//     hslot_f    : u64  (home_slot / target_home, or CKT_U64_MAX when N/A)
//     val_f      : i64  (value, or CKT_I64_MIN when N/A)
//     aux_f      : u64  (event-specific, see per-event table below)
//   Folded into the per-step running op_event_hash in emission order.
//
//   read_hash record (ONLY GET ops emit one), 8 fields in order, raw LE:
//     read_id:u64, key:u64, found:u8, source_kind:u8, slot_f:u64,
//     stash_iseq_f:u64, val_f:i64, vseq_f:u64.   source_kind in
//     {CKT_SRC_TABLE=0, CKT_SRC_STASH=1, CKT_SRC_NONE=2}.
//
// -------------------------------------------------------------------------
// 4. OPERATION SEMANTICS (exact, in execution order)
// -------------------------------------------------------------------------
// Dispatch by op_type; unknown op_type -> INVALID (see below). For each op the
// emitted events (kind + field values) and the counts[] increments are listed.
// "ev" denotes ++event_seq taken at the moment of emission (events within one
// op are emitted in the listed textual order, so event_seq is monotonic).
//
// find_live_table(k): scan offsets 0..neighborhood-1 from home0(k); for each
//   s=(home0+off)%slot_count return s if state[s]==LIVE && key[s]==k. If none,
//   repeat the same scan from home1(k). Return first match's slot, else -1.
//   (Probe/displacement order is strictly ascending offset; home0 cluster is
//   fully scanned before home1 cluster.)
// find_stash(k): linear scan stash indices 0..stash_size-1 (== insert_seq
//   ascending); return first index with s_key[i]==k, else -1.
// find_tombstone_for_key(k): over ALL slots s=0..slot_count-1, among slots with
//   state==TOMBSTONE && key[s]==k pick the one minimizing (tomb_seq=aux[s])
//   ascending, ties broken by slot index ascending. Return slot or -1.
//
// GET (op_type=CKT_OP_GET, a0=read_id, a1=key):
//   slot=find_live_table(key).
//     if slot>=0: found=1, src=TABLE, slot_f=slot, stash_iseq_f=U64_MAX,
//                 val_f=val[slot], vseq_f=aux[slot]; counts[0]++ (get_found).
//     else si=find_stash(key):
//       if si>=0: found=1, src=STASH, slot_f=U64_MAX, stash_iseq_f=s_iseq[si],
//                 val_f=s_val[si], vseq_f=s_vseq[si]; counts[0]++.
//       else: found=0, src=NONE, slot_f=U64_MAX, stash_iseq_f=U64_MAX,
//             val_f=CKT_I64_MIN, vseq_f=U64_MAX; counts[1]++ (get_missing).
//   Emit GET_RESULT: ev; key_f=key, slot_f=slot_f, hk_f=CKT_HK_NONE,
//     hslot_f=U64_MAX, val_f=val_f, aux_f=(uint64_t)src.
//   Then fold the read_hash record (read_id,key,found,src,slot_f,
//     stash_iseq_f,val_f,vseq_f). GET never mutates table/stash/pages.
//
// PUT (op_type=CKT_OP_PUT, a0=key, a1=value(i64)):
//   1) slot=find_live_table(key): if slot>=0 -> UPDATE-IN-PLACE:
//        ev; val[slot]=value; aux[slot]=ev (new version_seq).
//        Emit UPDATE_EXISTING: key_f=key, slot_f=slot, hk_f=home_kind[slot],
//          hslot_f=home_slot[slot], val_f=value, aux_f=ev. counts[3]++. STOP.
//   2) si=find_stash(key): if si>=0 -> UPDATE-IN-STASH:
//        ev; s_val[si]=value; s_vseq[si]=ev.
//        Emit UPDATE_STASH: key_f=key, slot_f=U64_MAX, hk_f=CKT_HK_NONE,
//          hslot_f=U64_MAX, val_f=value, aux_f=s_iseq[si]. counts[3]++. STOP.
//   3) ts=find_tombstone_for_key(key): if ts>=0 -> RESURRECT:
//        ev; state[ts]=LIVE; val[ts]=value; aux[ts]=ev (version_seq).
//        key/home_kind/home_slot/iseq[ts] are PRESERVED. tomb_count--.
//        Emit RESURRECT_TOMBSTONE: key_f=key, slot_f=ts, hk_f=home_kind[ts],
//          hslot_f=home_slot[ts], val_f=value, aux_f=iseq[ts]. counts[4]++. STOP
//   4) FRESH INSERT: new_iseq = insert_seq_next++ (post-increment).
//        r = insert_with_home(HOME0, key,value,new_iseq, use_existing_vseq=false)
//        if r<0: r = insert_with_home(HOME1, key,value,new_iseq, false)
//        if r>=0: done (PUT_INSERT already emitted/counted inside). STOP.
//        else (both homes failed):
//          if stash_size < stash_capacity -> STASH:
//             ev; append (key,value,new_iseq, s_vseq=ev) at index stash_size;
//             stash_size++. Emit PUT_STASH: key_f=key, slot_f=U64_MAX,
//             hk_f=CKT_HK_NONE, hslot_f=U64_MAX, val_f=value, aux_f=new_iseq.
//             counts[5]++.
//          else -> OOM: ev; Emit PUT_OOM: same fields as PUT_STASH (key_f=key,
//             slot_f=U64_MAX, hk_f=CKT_HK_NONE, hslot_f=U64_MAX, val_f=value,
//             aux_f=new_iseq). counts[6]++. No state change.
//   NOTE: a failed HOME0 attempt may have already performed relocations and/or
//   reused a tombstone (emitting events / mutating state) before returning -1;
//   those mutations are NOT rolled back. The subsequent HOME1 attempt then runs
//   on the mutated table.
//
// insert_with_home(hk_attempt, pk, pv, p_iseq, use_existing_vseq, existing_vseq):
//   target_home = (hk_attempt==HOME0)? home0(pk) : home1(pk).
//   a) FIND VACANCY: scan off=0..slot_count-1, s=(target_home+off)%slot_count;
//      first s with state[s] in {EMPTY,TOMBSTONE} AND pin_count[page_of(s)]==0
//      is the vacancy. If none found in the whole table -> return -1.
//   b) HOPSCOTCH DISPLACEMENT: while dist(target_home, vacancy) >= neighborhood:
//        if disp >= max_displacements_per_home -> return -1.
//        choose a slot to evict toward the vacancy: scan d=1..neighborhood-1,
//          cc = ((vacancy - d) % slot_count + slot_count) % slot_count; pick the
//          FIRST cc with state[cc]==LIVE AND pin_count[page_of(cc)]==0 AND
//          dist(home_slot[cc], vacancy) < neighborhood. If none -> return -1.
//        Move cc's entry into vacancy (overwrite): if vacancy was TOMBSTONE,
//          tomb_count-- (silent, NO event). Copy key/val/home_kind/home_slot/
//          iseq/aux from cc to vacancy; state[vacancy]=LIVE.
//        ev; Emit RELOCATE_SLOT: key_f=key[vacancy], slot_f=vacancy,
//          hk_f=home_kind[vacancy], hslot_f=home_slot[vacancy],
//          val_f=val[vacancy], aux_f=(uint64_t)cc (the source slot). counts[10]++
//        Clear cc to EMPTY with all fields 0. vacancy=cc; disp++.
//   c) PLACE: if state[vacancy]==TOMBSTONE: old_ts=aux[vacancy]; ev;
//        Emit REUSE_TOMBSTONE: key_f=pk, slot_f=vacancy, hk_f=hk_attempt,
//          hslot_f=target_home, val_f=CKT_I64_MIN, aux_f=old_ts. counts[11]++;
//          tomb_count--.
//      ev; vseq = use_existing_vseq ? existing_vseq : ev.
//      state[vacancy]=LIVE; key=pk; val=pv; home_kind=hk_attempt;
//      home_slot=target_home; iseq=p_iseq; aux=vseq.
//      Emit PUT_INSERT: key_f=pk, slot_f=vacancy, hk_f=hk_attempt,
//        hslot_f=target_home, val_f=pv, aux_f=p_iseq. counts[2]++.
//      Return vacancy (>=0).
//
// DELETE (op_type=CKT_OP_DELETE, a0=key):
//   slot=find_live_table(key): if slot>=0 -> TABLE DELETE (tombstone):
//      ev; state[slot]=TOMBSTONE; aux[slot]=ev (tomb_seq); tomb_count++.
//      key/val/home_kind/home_slot/iseq PRESERVED (val becomes old_value).
//      Emit DELETE_TABLE: key_f=key, slot_f=slot, hk_f=home_kind[slot],
//        hslot_f=home_slot[slot], val_f=val[slot], aux_f=ev. counts[7]++.
//      THEN if tomb_count > max_tombstones: run AUTO-SWEEP
//        (sweep_step until_threshold=true, see below). STOP.
//   else si=find_stash(key): if si>=0 -> STASH DELETE:
//      ev; removed_val=s_val[si]; removed_iseq=s_iseq[si].
//      Emit DELETE_STASH: key_f=key, slot_f=U64_MAX, hk_f=CKT_HK_NONE,
//        hslot_f=U64_MAX, val_f=removed_val, aux_f=removed_iseq. counts[8]++.
//      Erase stash index si (shift entries >si down by one; stash_size--). STOP.
//   else -> MISS: ev; Emit DELETE_MISS: key_f=key, slot_f=U64_MAX,
//      hk_f=CKT_HK_NONE, hslot_f=U64_MAX, val_f=CKT_I64_MIN, aux_f=U64_MAX.
//      counts[9]++.
//
// sweep_step(limit, until_threshold):  removed=0; loop:
//   if until_threshold: stop when tomb_count <= max_tombstones.
//   else: stop when removed >= limit.
//   pick target = removable tombstone minimizing (tomb_seq=aux) ascending,
//   ties slot ascending, among slots with state==TOMBSTONE AND
//   pin_count[page_of(s)]==0 (PINNED tombstones are SKIPPED, never removed).
//   If no removable tombstone remains -> stop.
//   ev; Emit TOMBSTONE_SWEEP: key_f=key[target], slot_f=target,
//     hk_f=home_kind[target], hslot_f=home_slot[target], val_f=val[target],
//     aux_f=aux[target]. counts[12]++. Clear target to EMPTY (all fields 0);
//   tomb_count--; removed++.
//
// SWEEP_TOMBSTONES (op_type=CKT_OP_SWEEP_TOMBSTONES, a0=limit):
//   lim = (limit<0)?0:(int)limit; sweep_step(lim, until_threshold=false).
//
// REPLAY_STASH (op_type=CKT_OP_REPLAY_STASH, a0=limit):
//   lim=(limit<0)?0:(int)limit; success=0; i=0;
//   while i < stash_size and success < lim:
//     take entry i: k=s_key[i], v=s_val[i], es=s_iseq[i], vs=s_vseq[i].
//     r=insert_with_home(HOME0, k,v,es, use_existing_vseq=true, existing_vseq=vs)
//     if r<0: r=insert_with_home(HOME1, k,v,es, true, vs)
//     if r>=0: ev; Emit STASH_REPLAY_OK: key_f=k, slot_f=r, hk_f=home_kind[r],
//        hslot_f=home_slot[r], val_f=v, aux_f=es. counts[13]++.
//        Erase stash index i (do NOT advance i; entries shift down). success++.
//     else: i++ (leave entry in stash, advance). PUT_INSERT and any RELOCATE/
//        REUSE events from insert_with_home are emitted/counted as usual; the
//        preserved insert_seq (es) and version_seq (vs) are reused for the
//        placed entry. Replayed entry preserves its original insert_seq order.
//
// PIN_PAGE (op_type=CKT_OP_PIN_PAGE, a0=page):
//   if page<0 || page>=n_pages || pin_count[page]==CKT_U64_MAX -> INVALID:
//     ev; Emit INVALID: key_f=U64_MAX, slot_f=U64_MAX, hk_f=CKT_HK_NONE,
//     hslot_f=U64_MAX, val_f=CKT_I64_MIN, aux_f=(uint64_t)page. counts[16]++.
//   else pin_count[page]++; ev; Emit PIN_PAGE_OK with the SAME field pattern
//     (key_f=U64_MAX, slot_f=U64_MAX, hk_f=CKT_HK_NONE, hslot_f=U64_MAX,
//      val_f=CKT_I64_MIN, aux_f=(uint64_t)page). counts[14]++.
//
// UNPIN_PAGE (op_type=CKT_OP_UNPIN_PAGE, a0=page):
//   if page<0 || page>=n_pages || pin_count[page]==0 -> INVALID (same INVALID
//     record as above, aux_f=(uint64_t)page). counts[16]++.
//   else pin_count[page]--; ev; Emit UNPIN_PAGE_OK (key_f=U64_MAX,
//     slot_f=U64_MAX, hk_f=CKT_HK_NONE, hslot_f=U64_MAX, val_f=CKT_I64_MIN,
//     aux_f=(uint64_t)page). counts[15]++.
//   (page argument here is the RAW signed a0; for the INVALID case the out-of-
//    range/negative a0 is reinterpreted as (uint64_t)page in aux_f.)
//
// UNKNOWN op_type (default): ev; Emit INVALID: key_f=U64_MAX, slot_f=U64_MAX,
//   hk_f=CKT_HK_NONE, hslot_f=U64_MAX, val_f=CKT_I64_MIN, aux_f=U64_MAX.
//   counts[16]++.
//
// -------------------------------------------------------------------------
// 5. PER-EVENT aux_f QUICK TABLE (the 9th field of each event record)
// -------------------------------------------------------------------------
//   GET_RESULT(0)          aux_f = (uint64_t)source_kind
//   UPDATE_EXISTING(1)     aux_f = ev (new version_seq)
//   UPDATE_STASH(2)        aux_f = s_iseq[si]
//   RESURRECT_TOMBSTONE(3) aux_f = iseq[ts] (preserved insert_seq)
//   RELOCATE_SLOT(4)       aux_f = (uint64_t)source_slot (cc)
//   REUSE_TOMBSTONE(5)     aux_f = old tomb_seq; val_f = CKT_I64_MIN
//   PUT_INSERT(6)          aux_f = p_iseq (insert_seq placed)
//   PUT_STASH(7)           aux_f = new_iseq
//   PUT_OOM(8)             aux_f = new_iseq
//   DELETE_TABLE(9)        aux_f = ev (tomb_seq)
//   DELETE_STASH(10)       aux_f = removed_iseq; val_f = removed_val
//   DELETE_MISS(11)        aux_f = U64_MAX; val_f = CKT_I64_MIN
//   TOMBSTONE_SWEEP(12)    aux_f = aux[target] (tomb_seq)
//   STASH_REPLAY_OK(13)    aux_f = es (preserved insert_seq)
//   PIN_PAGE_OK(14)        aux_f = (uint64_t)page
//   UNPIN_PAGE_OK(15)      aux_f = (uint64_t)page
//   INVALID(16)            aux_f = (uint64_t)page for pin/unpin invalid,
//                                  else U64_MAX
//
// -------------------------------------------------------------------------
// 6. FINAL-STATE HASHES (computed AFTER all ops of the step)
// -------------------------------------------------------------------------
//   slot_state_hash: start CKT_FNV_OFFSET; iterate slot s = 0..slot_count-1
//     (ASCENDING slot index). For EACH slot fold slot:u64 then state:u8. If
//     state==EMPTY, fold nothing further and continue. Otherwise (LIVE or
//     TOMBSTONE) ADDITIONALLY fold, in this order: key:u64, value/old_value:i64,
//     home_kind:u8, home_slot:u64, insert_seq:u64, aux:u64 (aux = version_seq
//     for LIVE, tomb_seq for TOMBSTONE).
//   stash_hash: start CKT_FNV_OFFSET; iterate stash entries by ARRAY index
//     0..stash_size-1 (== insert_seq ASCENDING). For each fold:
//     key:u64, value:i64, insert_seq:u64, version_seq(s_vseq):u64.
//   page_hash: start CKT_FNV_OFFSET; iterate page p = 0..n_pages-1 ASCENDING;
//     for each fold page:u64, pin_count:u64.
//   These three are recomputed from scratch each step; they are NOT chained
//   and carry NO running state from prior steps beyond the persistent table.
//
//   There is NO separate master "state_checksum": the grader compares all of
//   {counts[0..16], op_event_hash, read_hash, slot_state_hash, stash_hash,
//   page_hash} field-by-field. The six outputs together ARE the verified
//   fingerprint; equality on every one is required to pass.
//
// === END DETERMINISM & EXACT OUTPUT SERIALIZATION ===
*/

struct alignas(8) CktProblemSpec {
    int32_t abi_version;
    int32_t slot_count;
    int32_t page_size;
    int32_t neighborhood;
    int32_t max_displacements_per_home;
    int32_t stash_capacity;
    int32_t max_tombstones;
    int32_t max_ops;     // max ops per run
    int32_t max_steps;
    int32_t flags;
    uint64_t seed0;
    uint64_t seed1;
    int32_t reserved[6];
};

struct alignas(8) CktRunSpec {
    int32_t abi_version;
    int32_t num_ops;
    int32_t step_id;
    int32_t reserved[13];
};

struct alignas(8) CktInputs {
    const int32_t* op_type;  // num_ops
    const int64_t* a0;       // num_ops
    const int64_t* a1;       // num_ops
};

struct alignas(8) CktOutputs {
    int32_t* counts;          // 17 entries
    uint64_t* op_event_hash;  // 1
    uint64_t* read_hash;      // 1
    uint64_t* slot_state_hash;// 1
    uint64_t* stash_hash;     // 1
    uint64_t* page_hash;      // 1
};

#define CKT_NUM_COUNTS 17

static_assert(sizeof(CktProblemSpec) == 80, "CktProblemSpec layout drift");
static_assert(sizeof(CktRunSpec) == 64, "CktRunSpec layout drift");
static_assert(sizeof(CktInputs) == 24, "CktInputs layout drift");
static_assert(sizeof(CktOutputs) == 48, "CktOutputs layout drift");

static inline int ckt_n_pages(const CktProblemSpec* s) {
    return (s->slot_count + s->page_size - 1) / s->page_size;
}

static inline int ckt_validate_problem_spec(const CktProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != CKT_ABI_VERSION) return 0;
    if (spec->slot_count < CKT_MIN_SLOT_COUNT || spec->slot_count > CKT_MAX_SLOT_COUNT) return 0;
    if (spec->page_size < CKT_MIN_PAGE_SIZE || spec->page_size > CKT_MAX_PAGE_SIZE) return 0;
    if (spec->neighborhood < CKT_MIN_NEIGHBORHOOD || spec->neighborhood > spec->slot_count) return 0;
    if (spec->max_displacements_per_home < 0) return 0;
    if (spec->stash_capacity < 0 || spec->stash_capacity > CKT_MAX_STASH) return 0;
    if (spec->max_tombstones < 0 || spec->max_tombstones > CKT_MAX_TOMBSTONES) return 0;
    if (spec->max_ops < 0 || spec->max_ops > CKT_MAX_OPS) return 0;
    if (spec->max_steps < 1 || spec->max_steps > CKT_MAX_STEPS) return 0;
    return 1;
}

static inline int ckt_validate_run_spec(const CktRunSpec* run, const CktProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != CKT_ABI_VERSION) return 0;
    if (run->num_ops < 0 || run->num_ops > spec->max_ops) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const CktProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const CktProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const CktRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // CUCKOO_TOMBSTONE_TABLE_COMMON_H_
