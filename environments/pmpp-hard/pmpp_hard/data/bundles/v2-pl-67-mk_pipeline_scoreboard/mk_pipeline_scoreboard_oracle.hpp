// file: mk_pipeline_scoreboard_oracle.hpp

#ifndef MK_PIPELINE_SCOREBOARD_ORACLE_HPP_
#define MK_PIPELINE_SCOREBOARD_ORACLE_HPP_

#include "mk_pipeline_scoreboard_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <climits>
#include <sstream>
#include <string>
#include <vector>

/*
 * Host oracle: the canonical, exact MK8 software-pipeline scoreboard semantics.
 * The reference and naive CUDA kernels reproduce every output bit-for-bit.
 *
 * ============================ DETERMINISTIC READING ============================
 * The MK8 spec leaves several couplings to "the most specific deterministic
 * interpretation". They are pinned here and are identical across all three
 * implementations.
 *
 * INSTRUCTION ORDER. Instructions are stored in a slot array; "instr_seq order"
 *   means ascending instr_seq. Only ENQUEUE creates instr_seq (1,2,3,...), so
 *   instr_seq order == enqueue order == slot order. "Nonterminal" = status not
 *   in {DONE, CANCELLED}. "Earlier" = smaller instr_seq.
 *
 * ENQUEUE validity (any failure => INVALID, nothing else mutated):
 *   - instr_id == 0 is reserved (0 means "no instruction") => invalid.
 *   - id already exists among non-CANCELLED-and-non-DONE-or-any instructions:
 *     we forbid reuse of an id that exists in ANY slot (terminal or not) => invalid.
 *   - queue full (used slots == max_instrs) => invalid.
 *   - read_count > max_reads_per_instr or write_count > max_writes_per_instr.
 *   - any read or write tile id out of [0,tile_count) => invalid.
 *   - duplicate write tile (a write tile id repeated) => invalid. Duplicate read
 *     tiles ARE allowed (each increments pending_reader_count); a tile may be
 *     both read and written (read counted, then write).
 *   - scratch_pages + (distinct read tiles) + (distinct write tiles) > buffer_count.
 *     "distinct read tiles" counts unique read tile ids; a tile that is both read
 *     and written counts once for reads and once for writes here (capacity check
 *     is conservative, matching spec text "scratch pages plus distinct read/write
 *     tiles"). out_counter validity is NOT checked at enqueue (UINT32_MAX = none;
 *     any other value is stored verbatim and only used at COUNTER_INC where an
 *     out-of-range counter simply emits no COUNTER_INC).
 *
 * ISSUE_LOADS(limit). limit==0 is a valid no-op (no scan, no events).
 *   Scan QUEUED instructions in instr_seq order, restricted to the first
 *   `issue_window` NONTERMINAL NONCANCELLED instructions (the window is a prefix
 *   of {status != DONE, != CANCELLED} by instr_seq). We advance a window counter
 *   while scanning; once `issue_window` such instructions have been visited we
 *   stop. Issue up to `limit` loads (a "load" here counts one issued instruction;
 *   `limit` bounds issued instructions, matching "issue up to limit loads" where
 *   each load-issuable instruction consumes one unit).
 *   For each candidate QUEUED instruction in window:
 *     Compute hazards relative to EARLIER nonterminal instructions:
 *       WAW: some earlier nonterminal instr writes one of my write tiles.
 *       RAW: for a read tile, there is an earlier nonterminal writer pending
 *            (tile.writer_instr != 0 AND that writer is earlier & nonterminal)
 *            AND the tile's current stored version is not yet readable, i.e. the
 *            tile is not resident with current_version. Concretely RAW is clear
 *            for a read tile iff writer_instr == 0, OR (the tile is resident in a
 *            buffer whose version == current_version). [Spec: "no earlier writer
 *            pending, or the tile's current version is already stored and
 *            resident/readable".]
 *       WAR: for a write tile, some earlier nonterminal instr has an unreleased
 *            read of that tile. We track this as: tile.pending_reader_count
 *            attributable to earlier nonterminal instrs. Implemented as: WAR
 *            present iff there exists an earlier nonterminal instr that lists the
 *            tile as a read AND has not released its reads (last_use_released==0).
 *     Page capacity: count buffers needed = (nonresident read tiles) +
 *            (write tiles) + scratch_pages, where a read tile is "resident" iff
 *            tile.resident_buffer != NO and that buffer.version==current_version
 *            and buffer.state==RESIDENT. Available = number of FREE buffers.
 *            (We additionally may reuse the resident buffer for a read tile.)
 *     Failure handling: evaluate hazards in order WAW, RAW, WAR, then capacity.
 *       - On WAW/RAW/WAR failure: emit LOAD_HAZARD_STALL for THIS instruction and
 *         STOP the scan entirely (no later instructions considered this call).
 *       - On capacity-only failure: emit LOAD_HAZARD_STALL and CONTINUE scanning
 *         later instructions (still within window & limit).
 *     If issuable:
 *       Buffer allocation order (deterministic):
 *         (1) For each read tile in listed order: if the tile is resident in a
 *             buffer with version==current_version (state RESIDENT), REUSE it
 *             (pin it, no new load); else allocate the lowest-id FREE buffer,
 *             set it LOADING, tile=read tile, version=current_version,
 *             owner=this instr, and schedule a LOAD_DONE.
 *             read_version recorded = current_version of the tile at issue time.
 *         (2) For each write tile in listed order: allocate the lowest-id FREE
 *             buffer not already chosen this issue; set it LOADING, tile=write
 *             tile, owner=this instr; assign fresh write_version =
 *             version_seq_next++ ; set tile.writer_instr = this instr_id.
 *             A LOAD_DONE is scheduled for it (it must "load" its scratch page).
 *         (3) For each scratch page: allocate lowest-id FREE buffer not already
 *             chosen; set LOADING, tile=NO, owner=this instr, no version; schedule
 *             a LOAD_DONE.
 *       assigned_buffers[] lists, in order, all buffers chosen (reads, then
 *       writes, then scratch). pin_count incremented on each chosen buffer.
 *       If ANY buffer was newly set LOADING (i.e. at least one actual load was
 *       scheduled), status=LOAD_ISSUED and emit one LOAD_ISSUE per buffer that
 *       is LOADING (in assigned_buffers order, reused-resident buffers excluded).
 *       Schedule LOAD_DONE ops at clock+load_latency, one per LOADING buffer, in
 *       assigned_buffers order (so op_seq increases in that order).
 *       If NO actual load was needed (all reads resident, no writes, no scratch),
 *       status=LOAD_READY and emit LOAD_READY_INLINE (single event), and set every
 *       reused buffer's state to READY_READ.
 *     Each issued (or inline) instruction consumes one unit of `limit`.
 *
 * ISSUE_COMPUTE(limit). Scan by instr_seq, up to `limit` issued.
 *   For an instruction in LOAD_ISSUED: it is "still loading" if any assigned
 *   buffer has state LOADING. If still loading, emit COMPUTE_LOAD_WAIT (for that
 *   instr) and STOP the scan.
 *   For an instruction in LOAD_READY: set all its assigned buffers to COMPUTING,
 *   schedule one COMPUTE_DONE at clock+compute_latency, status=COMPUTE_ISSUED,
 *   emit COMPUTE_ISSUE. Consumes one unit of limit.
 *   Instructions not in {LOAD_ISSUED, LOAD_READY} are skipped (no event, no
 *   limit consumed, scan continues). limit==0 => no-op.
 *
 * ISSUE_STORES(limit). Scan by instr_seq, up to `limit` issued.
 *   For an instruction in COMPUTE_ISSUED whose compute has NOT completed
 *   (compute_done_flag==0): it is not ready; skip it (no event, scan continues).
 *   For an instruction in COMPUTE_ISSUED with compute_done_flag==1: set its WRITE
 *   buffers to STORING, schedule one STORE_DONE at clock+store_latency,
 *   status=STORE_ISSUED, emit STORE_ISSUE. Consumes one unit of limit. Other
 *   statuses skipped. limit==0 => no-op.
 *
 * ADVANCE(delta, max_ops). delta==0 valid. clock += delta. Then process up to
 *   max_ops ACTIVE pending ops whose due_clock <= clock, ordered by
 *   (due_clock, op_seq) ascending. Each processed op is removed from the queue.
 *   LOAD_DONE: if the op's buffer still belongs to the same instruction
 *     (buffer.owner_instr == op.instr_id) and the instr is nonterminal, set the
 *     buffer state to READY_WRITE if its tile is a write tile of the instr (i.e.
 *     buffer.version was a freshly reserved write version, tracked by role) else
 *     READY_READ; set last_touch_seq = ++event-independent touch (we use op_seq
 *     ordering, see below); emit LOAD_DONE. Otherwise the op is STALE: emit
 *     OP_STALE_DROP (and do nothing else). Buffer role (read vs write vs scratch)
 *     is recorded per (instr, buffer) at issue. Scratch buffers go to READY_READ.
 *     After applying, if ALL of the instr's LOADING buffers are now non-LOADING
 *     (none of its assigned buffers is LOADING), set status LOAD_READY (only if
 *     currently LOAD_ISSUED) and emit INSTR_LOAD_READY.
 *   COMPUTE_DONE: if instr status==COMPUTE_ISSUED, compute
 *       result = FNV1a64(instr_id, instr_seq, payload_seed, read tile versions in
 *                read order, write versions in write order, assigned buffers in
 *                order). Set compute_done_flag=1 (status stays COMPUTE_ISSUED).
 *       emit COMPUTE_DONE with aux_u64 = result. Else STALE => OP_STALE_DROP.
 *   STORE_DONE: if instr status==STORE_ISSUED: for each WRITE tile in listed
 *       order: set tile.current_version = its write_version; writer_instr = 0;
 *       last_store_seq = the event_seq stamped on its TILE_STORE event; dirty=0;
 *       set the instr's write buffer for that tile to RESIDENT with tile/version;
 *       tile.resident_buffer = that buffer; emit TILE_STORE. THEN decrement
 *       pending_reader_count for all READ tiles in listed order; for each read
 *       buffer of this instr, if its tile now has pending_reader_count==0 AND the
 *       buffer is not the resident current-version buffer of that tile, release it
 *       (set FREE, clear) and emit READ_RELEASE. Also release scratch buffers
 *       (always, they have no readers) via READ_RELEASE in assigned order after
 *       reads. mark last_use_released=1. THEN increment output counter if valid
 *       (out_counter in [0,counter_count)) and emit COUNTER_INC. THEN status=DONE,
 *       emit INSTR_DONE. Else STALE => OP_STALE_DROP.
 *     RELEASE ORDER for READ_RELEASE within a STORE_DONE: assigned_buffers order,
 *     restricted to read buffers then scratch buffers (i.e. the order they appear
 *     in assigned_buffers), each that qualifies. A read buffer that was a reused
 *     resident buffer (pinned, not owned exclusively) is released the same way if
 *     its tile reaches 0 readers and it is not the resident current-version buffer.
 *
 * CANCEL(instr_id). Invalid if id absent or instr terminal (DONE/CANCELLED).
 *   Mark CANCELLED. Logically deactivate its pending ops (set active=0; they are
 *   simply removed, NOT emitting OP_STALE_DROP now — stale drop only happens for
 *   ops processed in ADVANCE; cancelled-instr ops are removed here silently per
 *   "remove pending ops logically"). Release its owned NONRESIDENT buffers
 *   (buffer.owner_instr==id AND state != RESIDENT) in buffer-id order, set FREE,
 *   emit BUFFER_CANCEL_RELEASE per released buffer. For every read tile whose read
 *   was not yet released (instr.last_use_released==0), decrement
 *   pending_reader_count once per read-tile occurrence. For every write tile whose
 *   tile.writer_instr==id, clear writer_instr=0. Emit INSTR_CANCEL.
 *
 * HOST_COUNTER_INC(counter_id, amount). Invalid if counter_id out of range.
 *   counter[counter_id] += amount, emit HOST_COUNTER_INC.
 *
 * EVENT TUPLE fields (unused take sentinel): tile->NO_U32, buffer->NO_U32,
 *   version->0, counter->NO_U32, instr_id->0 when not applicable, aux->0.
 *
 * COUNTERS array `counter[]` is the user dependency-counter array (size
 *   counter_count), incremented by COUNTER_INC and HOST_COUNTER_INC. Separate
 *   from `counts[]` which are per-event-class statistics.
 *
 * TOUCH ORDERING. We need a deterministic last_touch_seq for buffers. We use a
 *   persistent touch_seq_next starting 1, post-incremented whenever a buffer's
 *   last_touch_seq is updated (at allocation/LOAD_DONE/state transitions that the
 *   spec marks "set last_touch_seq"). Only LOAD_DONE sets last_touch in spec; we
 *   also set it at buffer allocation so buffer_hash is well-defined. This is an
 *   internal detail folded into buffer_hash & state_checksum identically in all 3.
 */

struct MkpsHostInputsView {
    const int32_t* op;
    const uint32_t* a0; const uint32_t* a1; const uint32_t* a2; const uint32_t* a3;
    const uint32_t* a4; const uint32_t* a5; const uint32_t* a6; const uint32_t* a7;
    const uint64_t* a8;
    const uint32_t* tiles;
};

struct MkpsHostOutputsView {
    const int64_t* counts;
    const uint64_t* pipeline_event_hash;
    const uint64_t* instr_hash;
    const uint64_t* tile_hash;
    const uint64_t* buffer_hash;
    const uint64_t* pending_hash;
    const uint64_t* counter_hash;
    const uint64_t* event_seq_out;
    const uint64_t* state_checksum;
};

struct MkpsExpected {
    std::vector<int64_t> counts;
    uint64_t pipeline_event_hash = 0;
    uint64_t instr_hash = 0;
    uint64_t tile_hash = 0;
    uint64_t buffer_hash = 0;
    uint64_t pending_hash = 0;
    uint64_t counter_hash = 0;
    uint64_t event_seq_out = 0;
    uint64_t state_checksum = 0;
};

static inline uint64_t mkps_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= 1099511628211ULL;
    return h;
}
static inline void mkps_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = (const uint8_t*)ptr;
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = mkps_fnv_byte(v, p[i]);
    *h = v;
}
static constexpr uint64_t MKPS_FNV_OFF = 1469598103934665603ULL;

struct OrInstr {
    bool used = false;
    uint32_t instr_id = 0;
    uint64_t instr_seq = 0;
    int read_count = 0;
    int write_count = 0;
    uint32_t reads[MKPS_MAX_READS] = {};
    uint32_t writes[MKPS_MAX_WRITES] = {};
    int scratch_pages = 0;
    uint32_t load_latency = 0, compute_latency = 0, store_latency = 0;
    uint32_t out_counter = MKPS_NO_U32;
    uint64_t payload_seed = 0;
    uint8_t status = MKPS_ST_QUEUED;
    // assigned buffers (ordered reads, writes, scratch)
    int nbuf = 0;
    uint32_t buf_id[MKPS_MAX_READS + MKPS_MAX_WRITES + MKPS_MAX_BUFFERS] = {};
    uint8_t  buf_role[MKPS_MAX_READS + MKPS_MAX_WRITES + MKPS_MAX_BUFFERS] = {};
    uint8_t  buf_reused[MKPS_MAX_READS + MKPS_MAX_WRITES + MKPS_MAX_BUFFERS] = {};
    uint64_t read_versions[MKPS_MAX_READS] = {};
    uint64_t write_versions[MKPS_MAX_WRITES] = {};
    uint8_t compute_done_flag = 0;
    uint8_t last_use_released = 0;
};

struct OrTile {
    uint64_t current_version = 0;
    uint32_t writer_instr = 0;   // instr_id, 0=none
    uint64_t pending_reader_count = 0;
    uint64_t last_store_seq = 0;
    uint32_t resident_buffer = MKPS_NO_U32;
    uint8_t dirty = 0;
};

struct OrBuf {
    uint8_t state = MKPS_BUF_FREE;
    uint32_t tile = MKPS_NO_U32;
    uint64_t version = 0;
    uint32_t owner_instr = 0;
    uint64_t pin_count = 0;
    uint64_t last_touch_seq = 0;
};

struct OrPend {
    bool active = false;
    uint8_t kind = 0;
    uint64_t due_clock = 0;
    uint64_t op_seq = 0;
    uint32_t instr_id = 0;
    uint32_t buffer_id = MKPS_NO_U32;
};

struct MkpsOracleState {
    MkpsProblemSpec spec{};

    uint64_t clock = 0;
    uint64_t event_seq = 0;
    uint64_t instr_seq_next = 1;
    uint64_t op_seq_next = 1;
    uint64_t version_seq_next = 1;
    uint64_t touch_seq_next = 1;

    std::vector<OrInstr> instrs;     // [max_instrs] slot array
    std::vector<OrTile> tiles;       // [tile_count]
    std::vector<OrBuf> bufs;         // [buffer_count]
    std::vector<OrPend> pend;        // [max_pending_ops] queue (compacted on remove)
    int pend_n = 0;
    std::vector<uint64_t> counter;   // [counter_count]

    std::vector<int64_t> counts;     // cumulative stats
    uint64_t running_event_hash = MKPS_FNV_OFF;

    int B, T, NI, MR, MW, WIN, MP, NC;

    void init(const MkpsProblemSpec& s) {
        spec = s;
        B = s.buffer_count; T = s.tile_count; NI = s.max_instrs;
        MR = s.max_reads_per_instr; MW = s.max_writes_per_instr;
        WIN = s.issue_window; MP = s.max_pending_ops; NC = s.counter_count;
        instrs.assign((size_t)NI, OrInstr{});
        tiles.assign((size_t)T, OrTile{});
        bufs.assign((size_t)B, OrBuf{});
        pend.assign((size_t)MP, OrPend{});
        counter.assign((size_t)NC, 0);
        counts.assign(MKPS_COUNT_FIELDS, 0);
        reset();
    }

    void reset() {
        clock = 0; event_seq = 0; instr_seq_next = 1; op_seq_next = 1;
        version_seq_next = 1; touch_seq_next = 1;
        for (auto& x : instrs) x = OrInstr{};
        for (auto& x : tiles) x = OrTile{};
        for (auto& x : bufs) x = OrBuf{};
        for (auto& x : pend) x = OrPend{};
        pend_n = 0;
        std::fill(counter.begin(), counter.end(), (uint64_t)0);
        std::fill(counts.begin(), counts.end(), (int64_t)0);
        running_event_hash = MKPS_FNV_OFF;
    }

    // ---- event emission ----
    void emit(uint8_t kind, uint32_t op_index, uint64_t instr_id, uint32_t tile,
              uint32_t buffer, uint64_t version, uint32_t counter_id, uint64_t aux) {
        uint64_t h = running_event_hash;
        uint64_t seq = event_seq;
        uint64_t clk = clock;
        mkps_fnv_bytes(&h, &kind, sizeof(uint8_t));
        mkps_fnv_bytes(&h, &seq, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &op_index, sizeof(uint32_t));
        mkps_fnv_bytes(&h, &clk, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &instr_id, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &tile, sizeof(uint32_t));
        mkps_fnv_bytes(&h, &buffer, sizeof(uint32_t));
        mkps_fnv_bytes(&h, &version, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &counter_id, sizeof(uint32_t));
        mkps_fnv_bytes(&h, &aux, sizeof(uint64_t));
        running_event_hash = h;
        event_seq += 1;
    }

    int find_instr(uint32_t id) const {
        if (id == 0) return -1;
        for (int i = 0; i < NI; ++i)
            if (instrs[(size_t)i].used && instrs[(size_t)i].instr_id == id) return i;
        return -1;
    }
    int free_instr_slot() const {
        for (int i = 0; i < NI; ++i) if (!instrs[(size_t)i].used) return i;
        return -1;
    }
    bool nonterminal(const OrInstr& in) const {
        return in.used && in.status != MKPS_ST_DONE && in.status != MKPS_ST_CANCELLED;
    }
    // order index by instr_seq -> we return slot indices sorted by instr_seq.
    std::vector<int> order_by_seq() const {
        std::vector<int> idx;
        for (int i = 0; i < NI; ++i) if (instrs[(size_t)i].used) idx.push_back(i);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return instrs[(size_t)a].instr_seq < instrs[(size_t)b].instr_seq;
        });
        return idx;
    }

    int lowest_free_buffer(const std::vector<char>& chosen) const {
        for (int b = 0; b < B; ++b)
            if (bufs[(size_t)b].state == MKPS_BUF_FREE && !chosen[(size_t)b]) return b;
        return -1;
    }
    int free_buffer_count() const {
        int n = 0;
        for (int b = 0; b < B; ++b) if (bufs[(size_t)b].state == MKPS_BUF_FREE) ++n;
        return n;
    }

    void push_pend(uint8_t kind, uint64_t due, uint32_t instr_id, uint32_t buffer_id) {
        if (pend_n >= MP) return; // queue full: silently drop scheduling (capacity guarded elsewhere)
        OrPend& p = pend[(size_t)pend_n++];
        p.active = true; p.kind = kind; p.due_clock = due;
        p.op_seq = op_seq_next++; p.instr_id = instr_id; p.buffer_id = buffer_id;
    }

    // ---- ENQUEUE ----
    void do_enqueue(uint32_t op_index, uint32_t id, uint32_t read_count, uint32_t write_count,
                    uint32_t scratch_pages, uint32_t load_lat, uint32_t comp_lat,
                    uint32_t store_lat, uint32_t out_counter, uint64_t payload_seed,
                    const uint32_t* reads, const uint32_t* writes) {
        bool ok = true;
        if (id == 0) ok = false;
        else if (find_instr(id) >= 0) ok = false;
        else if (free_instr_slot() < 0) ok = false;
        else if ((int)read_count > MR || (int)write_count > MW) ok = false;
        if (ok) {
            for (uint32_t r = 0; r < read_count; ++r)
                if (reads[r] >= (uint32_t)T) { ok = false; break; }
        }
        if (ok) {
            for (uint32_t w = 0; w < write_count; ++w)
                if (writes[w] >= (uint32_t)T) { ok = false; break; }
        }
        if (ok) {
            for (uint32_t a = 0; a < write_count && ok; ++a)
                for (uint32_t b = a + 1; b < write_count; ++b)
                    if (writes[a] == writes[b]) { ok = false; break; }
        }
        if (ok) {
            // distinct read tiles
            int distinct_r = 0;
            for (uint32_t a = 0; a < read_count; ++a) {
                bool seen = false;
                for (uint32_t b = 0; b < a; ++b) if (reads[b] == reads[a]) { seen = true; break; }
                if (!seen) ++distinct_r;
            }
            int need = (int)scratch_pages + distinct_r + (int)write_count;
            if (need > B) ok = false;
        }
        if (!ok) {
            counts[MKC_invalid_count] += 1;
            emit(MKEV_INVALID, op_index, (uint64_t)id, MKPS_NO_U32, MKPS_NO_U32, 0,
                 MKPS_NO_U32, 0);
            return;
        }
        int slot = free_instr_slot();
        OrInstr& in = instrs[(size_t)slot];
        in = OrInstr{};
        in.used = true;
        in.instr_id = id;
        in.instr_seq = instr_seq_next++;
        in.read_count = (int)read_count;
        in.write_count = (int)write_count;
        for (uint32_t r = 0; r < read_count; ++r) in.reads[r] = reads[r];
        for (uint32_t w = 0; w < write_count; ++w) in.writes[w] = writes[w];
        in.scratch_pages = (int)scratch_pages;
        in.load_latency = load_lat;
        in.compute_latency = comp_lat;
        in.store_latency = store_lat;
        in.out_counter = out_counter;
        in.payload_seed = payload_seed;
        in.status = MKPS_ST_QUEUED;
        for (uint32_t r = 0; r < read_count; ++r)
            tiles[(size_t)reads[r]].pending_reader_count += 1;
        counts[MKC_instr_enqueued] += 1;
        emit(MKEV_INSTR_ENQUEUE, op_index, (uint64_t)id, MKPS_NO_U32, MKPS_NO_U32, 0,
             MKPS_NO_U32, (uint64_t)in.instr_seq);
    }

    bool read_tile_resident_current(uint32_t tile) const {
        const OrTile& t = tiles[(size_t)tile];
        if (t.resident_buffer == MKPS_NO_U32) return false;
        const OrBuf& bf = bufs[(size_t)t.resident_buffer];
        return bf.state == MKPS_BUF_RESIDENT && bf.tile == tile &&
               bf.version == t.current_version;
    }

    // ---- ISSUE_LOADS ----
    void do_issue_loads(uint32_t op_index, uint32_t limit) {
        if (limit == 0) return;
        std::vector<int> order = order_by_seq();
        int issued = 0;
        int window_seen = 0;
        for (int oi = 0; oi < (int)order.size(); ++oi) {
            int si = order[oi];
            OrInstr& in = instrs[(size_t)si];
            if (!nonterminal(in)) continue; // window counts nonterminal noncancelled
            if (window_seen >= WIN) break;
            window_seen += 1;
            if ((int)issued >= (int)limit) break;
            if (in.status != MKPS_ST_QUEUED) continue;

            // hazards relative to earlier nonterminal instrs
            bool waw = false, raw = false, war = false;
            // WAW
            for (int oj = 0; oj < oi && !waw; ++oj) {
                OrInstr& e = instrs[(size_t)order[oj]];
                if (!nonterminal(e)) continue;
                for (int w = 0; w < in.write_count && !waw; ++w)
                    for (int ew = 0; ew < e.write_count; ++ew)
                        if (e.writes[ew] == in.writes[w]) { waw = true; break; }
            }
            // RAW
            if (!waw) {
                for (int r = 0; r < in.read_count && !raw; ++r) {
                    uint32_t tl = in.reads[r];
                    if (tiles[(size_t)tl].writer_instr != 0) {
                        // is the writer an earlier nonterminal instr?
                        int wsi = find_instr(tiles[(size_t)tl].writer_instr);
                        bool earlier_nonterm = false;
                        if (wsi >= 0) {
                            OrInstr& we = instrs[(size_t)wsi];
                            if (nonterminal(we) && we.instr_seq < in.instr_seq)
                                earlier_nonterm = true;
                        }
                        if (earlier_nonterm && !read_tile_resident_current(tl)) raw = true;
                    }
                }
            }
            // WAR
            if (!waw && !raw) {
                for (int w = 0; w < in.write_count && !war; ++w) {
                    uint32_t tl = in.writes[w];
                    for (int oj = 0; oj < oi && !war; ++oj) {
                        OrInstr& e = instrs[(size_t)order[oj]];
                        if (!nonterminal(e)) continue;
                        if (e.last_use_released) continue;
                        for (int er = 0; er < e.read_count; ++er)
                            if (e.reads[er] == tl) { war = true; break; }
                    }
                }
            }

            if (waw || raw || war) {
                counts[MKC_load_hazard_stall] += 1;
                emit(MKEV_LOAD_HAZARD_STALL, op_index, (uint64_t)in.instr_id, MKPS_NO_U32,
                     MKPS_NO_U32, 0, MKPS_NO_U32, 0);
                break; // hazard stops scan
            }

            // capacity: count needed buffers
            int need = 0;
            for (int r = 0; r < in.read_count; ++r) {
                // distinct read tiles, nonresident
                bool dup = false;
                for (int q = 0; q < r; ++q) if (in.reads[q] == in.reads[r]) { dup = true; break; }
                if (dup) continue;
                if (!read_tile_resident_current(in.reads[r])) need += 1;
            }
            need += in.write_count;
            need += in.scratch_pages;
            int avail = free_buffer_count();
            if (need > avail) {
                counts[MKC_load_hazard_stall] += 1;
                emit(MKEV_LOAD_HAZARD_STALL, op_index, (uint64_t)in.instr_id, MKPS_NO_U32,
                     MKPS_NO_U32, 0, MKPS_NO_U32, 0);
                continue; // capacity stall permits scanning later
            }

            // issuable: allocate
            std::vector<char> chosen((size_t)B, 0);
            in.nbuf = 0;
            bool any_load = false;
            // (1) reads
            for (int r = 0; r < in.read_count; ++r) {
                uint32_t tl = in.reads[r];
                in.read_versions[r] = tiles[(size_t)tl].current_version;
                if (read_tile_resident_current(tl)) {
                    uint32_t bid = tiles[(size_t)tl].resident_buffer;
                    OrBuf& bf = bufs[(size_t)bid];
                    bf.pin_count += 1;
                    bf.last_touch_seq = touch_seq_next++;
                    chosen[(size_t)bid] = 1;
                    in.buf_id[in.nbuf] = bid;
                    in.buf_role[in.nbuf] = MKPS_ROLE_READ;
                    in.buf_reused[in.nbuf] = 1;
                    in.nbuf += 1;
                } else {
                    int bid = lowest_free_buffer(chosen);
                    chosen[(size_t)bid] = 1;
                    OrBuf& bf = bufs[(size_t)bid];
                    bf.state = MKPS_BUF_LOADING;
                    bf.tile = tl;
                    bf.version = tiles[(size_t)tl].current_version;
                    bf.owner_instr = in.instr_id;
                    bf.pin_count += 1;
                    bf.last_touch_seq = touch_seq_next++;
                    in.buf_id[in.nbuf] = (uint32_t)bid;
                    in.buf_role[in.nbuf] = MKPS_ROLE_READ;
                    in.buf_reused[in.nbuf] = 0;
                    in.nbuf += 1;
                    any_load = true;
                }
            }
            // (2) writes
            for (int w = 0; w < in.write_count; ++w) {
                uint32_t tl = in.writes[w];
                int bid = lowest_free_buffer(chosen);
                chosen[(size_t)bid] = 1;
                uint64_t wv = version_seq_next++;
                in.write_versions[w] = wv;
                OrBuf& bf = bufs[(size_t)bid];
                bf.state = MKPS_BUF_LOADING;
                bf.tile = tl;
                bf.version = wv;
                bf.owner_instr = in.instr_id;
                bf.pin_count += 1;
                bf.last_touch_seq = touch_seq_next++;
                tiles[(size_t)tl].writer_instr = in.instr_id;
                in.buf_id[in.nbuf] = (uint32_t)bid;
                in.buf_role[in.nbuf] = MKPS_ROLE_WRITE;
                in.buf_reused[in.nbuf] = 0;
                in.nbuf += 1;
                any_load = true;
            }
            // (3) scratch
            for (int s = 0; s < in.scratch_pages; ++s) {
                int bid = lowest_free_buffer(chosen);
                chosen[(size_t)bid] = 1;
                OrBuf& bf = bufs[(size_t)bid];
                bf.state = MKPS_BUF_LOADING;
                bf.tile = MKPS_NO_U32;
                bf.version = 0;
                bf.owner_instr = in.instr_id;
                bf.pin_count += 1;
                bf.last_touch_seq = touch_seq_next++;
                in.buf_id[in.nbuf] = (uint32_t)bid;
                in.buf_role[in.nbuf] = MKPS_ROLE_SCRATCH;
                in.buf_reused[in.nbuf] = 0;
                in.nbuf += 1;
                any_load = true;
            }

            if (any_load) {
                in.status = MKPS_ST_LOAD_ISSUED;
                counts[MKC_load_issue] += 1; // one per instruction issued
                for (int k = 0; k < in.nbuf; ++k) {
                    if (in.buf_reused[k]) continue;
                    emit(MKEV_LOAD_ISSUE, op_index, (uint64_t)in.instr_id, in.buf_role[k] == MKPS_ROLE_SCRATCH ? MKPS_NO_U32 : bufs[(size_t)in.buf_id[k]].tile,
                         in.buf_id[k], bufs[(size_t)in.buf_id[k]].version, MKPS_NO_U32, 0);
                }
                for (int k = 0; k < in.nbuf; ++k) {
                    if (in.buf_reused[k]) continue;
                    push_pend(MKPS_PEND_LOAD_DONE, clock + (uint64_t)in.load_latency,
                              in.instr_id, in.buf_id[k]);
                }
            } else {
                in.status = MKPS_ST_LOAD_READY;
                for (int k = 0; k < in.nbuf; ++k) {
                    OrBuf& bf = bufs[(size_t)in.buf_id[k]];
                    bf.state = MKPS_BUF_READY_READ;
                }
                counts[MKC_load_ready_inline] += 1;
                emit(MKEV_LOAD_READY_INLINE, op_index, (uint64_t)in.instr_id, MKPS_NO_U32,
                     MKPS_NO_U32, 0, MKPS_NO_U32, 0);
            }
            issued += 1;
        }
    }

    bool instr_any_loading(const OrInstr& in) const {
        for (int k = 0; k < in.nbuf; ++k)
            if (bufs[(size_t)in.buf_id[k]].state == MKPS_BUF_LOADING) return true;
        return false;
    }

    // ---- ISSUE_COMPUTE ----
    void do_issue_compute(uint32_t op_index, uint32_t limit) {
        if (limit == 0) return;
        std::vector<int> order = order_by_seq();
        int issued = 0;
        for (int oi = 0; oi < (int)order.size(); ++oi) {
            if ((int)issued >= (int)limit) break;
            OrInstr& in = instrs[(size_t)order[oi]];
            if (in.status == MKPS_ST_LOAD_ISSUED) {
                if (instr_any_loading(in)) {
                    counts[MKC_compute_load_wait] += 1;
                    emit(MKEV_COMPUTE_LOAD_WAIT, op_index, (uint64_t)in.instr_id, MKPS_NO_U32,
                         MKPS_NO_U32, 0, MKPS_NO_U32, 0);
                    break;
                }
                continue; // not LOAD_READY yet; skip without consuming
            }
            if (in.status != MKPS_ST_LOAD_READY) continue;
            for (int k = 0; k < in.nbuf; ++k)
                bufs[(size_t)in.buf_id[k]].state = MKPS_BUF_COMPUTING;
            push_pend(MKPS_PEND_COMPUTE_DONE, clock + (uint64_t)in.compute_latency,
                      in.instr_id, MKPS_NO_U32);
            in.status = MKPS_ST_COMPUTE_ISSUED;
            counts[MKC_compute_issue] += 1;
            emit(MKEV_COMPUTE_ISSUE, op_index, (uint64_t)in.instr_id, MKPS_NO_U32,
                 MKPS_NO_U32, 0, MKPS_NO_U32, 0);
            issued += 1;
        }
    }

    // ---- ISSUE_STORES ----
    void do_issue_stores(uint32_t op_index, uint32_t limit) {
        if (limit == 0) return;
        std::vector<int> order = order_by_seq();
        int issued = 0;
        for (int oi = 0; oi < (int)order.size(); ++oi) {
            if ((int)issued >= (int)limit) break;
            OrInstr& in = instrs[(size_t)order[oi]];
            if (in.status != MKPS_ST_COMPUTE_ISSUED) continue;
            if (!in.compute_done_flag) continue; // not ready
            for (int k = 0; k < in.nbuf; ++k)
                if (in.buf_role[k] == MKPS_ROLE_WRITE)
                    bufs[(size_t)in.buf_id[k]].state = MKPS_BUF_STORING;
            push_pend(MKPS_PEND_STORE_DONE, clock + (uint64_t)in.store_latency,
                      in.instr_id, MKPS_NO_U32);
            in.status = MKPS_ST_STORE_ISSUED;
            counts[MKC_store_issue] += 1;
            emit(MKEV_STORE_ISSUE, op_index, (uint64_t)in.instr_id, MKPS_NO_U32,
                 MKPS_NO_U32, 0, MKPS_NO_U32, 0);
            issued += 1;
        }
    }

    uint64_t compute_fnv_result(const OrInstr& in) const {
        uint64_t h = MKPS_FNV_OFF;
        uint32_t id = in.instr_id;
        uint64_t seq = in.instr_seq;
        uint64_t ps = in.payload_seed;
        mkps_fnv_bytes(&h, &id, sizeof(uint32_t));
        mkps_fnv_bytes(&h, &seq, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &ps, sizeof(uint64_t));
        for (int r = 0; r < in.read_count; ++r) {
            uint64_t v = in.read_versions[r];
            mkps_fnv_bytes(&h, &v, sizeof(uint64_t));
        }
        for (int w = 0; w < in.write_count; ++w) {
            uint64_t v = in.write_versions[w];
            mkps_fnv_bytes(&h, &v, sizeof(uint64_t));
        }
        for (int k = 0; k < in.nbuf; ++k) {
            uint32_t b = in.buf_id[k];
            mkps_fnv_bytes(&h, &b, sizeof(uint32_t));
        }
        return h;
    }

    // ---- ADVANCE ----
    void process_load_done(uint32_t op_index, const OrPend& p) {
        int si = find_instr(p.instr_id);
        if (si < 0) {
            counts[MKC_op_stale_drop] += 1;
            emit(MKEV_OP_STALE_DROP, op_index, (uint64_t)p.instr_id, MKPS_NO_U32,
                 p.buffer_id, 0, MKPS_NO_U32, (uint64_t)MKPS_PEND_LOAD_DONE);
            return;
        }
        OrInstr& in = instrs[(size_t)si];
        OrBuf& bf = bufs[(size_t)p.buffer_id];
        if (!nonterminal(in) || bf.owner_instr != p.instr_id || bf.state != MKPS_BUF_LOADING) {
            counts[MKC_op_stale_drop] += 1;
            emit(MKEV_OP_STALE_DROP, op_index, (uint64_t)p.instr_id, MKPS_NO_U32,
                 p.buffer_id, 0, MKPS_NO_U32, (uint64_t)MKPS_PEND_LOAD_DONE);
            return;
        }
        // determine role
        uint8_t role = MKPS_ROLE_SCRATCH;
        for (int k = 0; k < in.nbuf; ++k)
            if (in.buf_id[k] == p.buffer_id) { role = in.buf_role[k]; break; }
        bf.state = (role == MKPS_ROLE_WRITE) ? MKPS_BUF_READY_WRITE : MKPS_BUF_READY_READ;
        bf.last_touch_seq = touch_seq_next++;
        counts[MKC_load_done] += 1;
        emit(MKEV_LOAD_DONE, op_index, (uint64_t)p.instr_id, bf.tile, p.buffer_id,
             bf.version, MKPS_NO_U32, 0);
        if (in.status == MKPS_ST_LOAD_ISSUED && !instr_any_loading(in)) {
            in.status = MKPS_ST_LOAD_READY;
            counts[MKC_instr_load_ready] += 1;
            emit(MKEV_INSTR_LOAD_READY, op_index, (uint64_t)p.instr_id, MKPS_NO_U32,
                 MKPS_NO_U32, 0, MKPS_NO_U32, 0);
        }
    }

    void process_compute_done(uint32_t op_index, const OrPend& p) {
        int si = find_instr(p.instr_id);
        if (si < 0 || instrs[(size_t)si].status != MKPS_ST_COMPUTE_ISSUED) {
            counts[MKC_op_stale_drop] += 1;
            emit(MKEV_OP_STALE_DROP, op_index, (uint64_t)p.instr_id, MKPS_NO_U32,
                 MKPS_NO_U32, 0, MKPS_NO_U32, (uint64_t)MKPS_PEND_COMPUTE_DONE);
            return;
        }
        OrInstr& in = instrs[(size_t)si];
        uint64_t result = compute_fnv_result(in);
        in.compute_done_flag = 1;
        counts[MKC_compute_done] += 1;
        emit(MKEV_COMPUTE_DONE, op_index, (uint64_t)in.instr_id, MKPS_NO_U32,
             MKPS_NO_U32, 0, MKPS_NO_U32, result);
    }

    void process_store_done(uint32_t op_index, const OrPend& p) {
        int si = find_instr(p.instr_id);
        if (si < 0 || instrs[(size_t)si].status != MKPS_ST_STORE_ISSUED) {
            counts[MKC_op_stale_drop] += 1;
            emit(MKEV_OP_STALE_DROP, op_index, (uint64_t)p.instr_id, MKPS_NO_U32,
                 MKPS_NO_U32, 0, MKPS_NO_U32, (uint64_t)MKPS_PEND_STORE_DONE);
            return;
        }
        OrInstr& in = instrs[(size_t)si];
        // write tiles in listed order
        for (int w = 0; w < in.write_count; ++w) {
            uint32_t tl = in.writes[w];
            uint64_t wv = in.write_versions[w];
            // find the write buffer assigned to this tile
            uint32_t wbuf = MKPS_NO_U32;
            for (int k = 0; k < in.nbuf; ++k)
                if (in.buf_role[k] == MKPS_ROLE_WRITE && bufs[(size_t)in.buf_id[k]].tile == tl) {
                    wbuf = in.buf_id[k]; break;
                }
            OrTile& t = tiles[(size_t)tl];
            t.current_version = wv;
            t.writer_instr = 0;
            t.last_store_seq = event_seq; // seq stamped on the TILE_STORE we are about to emit
            t.dirty = 0;
            if (wbuf != MKPS_NO_U32) {
                OrBuf& bf = bufs[(size_t)wbuf];
                bf.state = MKPS_BUF_RESIDENT;
                bf.tile = tl;
                bf.version = wv;
                bf.last_touch_seq = touch_seq_next++;
                t.resident_buffer = wbuf;
            }
            counts[MKC_tile_store] += 1;
            emit(MKEV_TILE_STORE, op_index, (uint64_t)in.instr_id, tl, wbuf, wv,
                 MKPS_NO_U32, 0);
        }
        // decrement pending readers for all read tiles in listed order
        for (int r = 0; r < in.read_count; ++r) {
            OrTile& t = tiles[(size_t)in.reads[r]];
            if (t.pending_reader_count > 0) t.pending_reader_count -= 1;
        }
        in.last_use_released = 1;
        // release read & scratch buffers in assigned order
        for (int k = 0; k < in.nbuf; ++k) {
            if (in.buf_role[k] == MKPS_ROLE_WRITE) continue;
            uint32_t bid = in.buf_id[k];
            OrBuf& bf = bufs[(size_t)bid];
            if (in.buf_role[k] == MKPS_ROLE_READ) {
                uint32_t tl = bf.tile;
                bool zero_readers = (tl < (uint32_t)T) ? (tiles[(size_t)tl].pending_reader_count == 0) : true;
                bool is_resident_cur = (tl < (uint32_t)T) && tiles[(size_t)tl].resident_buffer == bid &&
                                       bufs[(size_t)bid].version == tiles[(size_t)tl].current_version &&
                                       bufs[(size_t)bid].state == MKPS_BUF_RESIDENT;
                if (zero_readers && !is_resident_cur) {
                    if (tl < (uint32_t)T && tiles[(size_t)tl].resident_buffer == bid)
                        tiles[(size_t)tl].resident_buffer = MKPS_NO_U32;
                    bf = OrBuf{};
                    counts[MKC_read_release] += 1;
                    emit(MKEV_READ_RELEASE, op_index, (uint64_t)in.instr_id, tl, bid, 0,
                         MKPS_NO_U32, 0);
                }
            } else { // scratch
                bf = OrBuf{};
                counts[MKC_read_release] += 1;
                emit(MKEV_READ_RELEASE, op_index, (uint64_t)in.instr_id, MKPS_NO_U32, bid, 0,
                     MKPS_NO_U32, 0);
            }
        }
        // counter increment
        if (in.out_counter != MKPS_NO_U32 && in.out_counter < (uint32_t)NC) {
            counter[(size_t)in.out_counter] += 1;
            counts[MKC_counter_inc] += 1;
            emit(MKEV_COUNTER_INC, op_index, (uint64_t)in.instr_id, MKPS_NO_U32, MKPS_NO_U32,
                 0, in.out_counter, counter[(size_t)in.out_counter]);
        }
        in.status = MKPS_ST_DONE;
        counts[MKC_instr_done] += 1;
        emit(MKEV_INSTR_DONE, op_index, (uint64_t)in.instr_id, MKPS_NO_U32, MKPS_NO_U32,
             0, MKPS_NO_U32, 0);
    }

    void do_advance(uint32_t op_index, uint32_t delta, uint32_t max_ops) {
        clock += (uint64_t)delta;
        uint32_t processed = 0;
        while (processed < max_ops) {
            // find min (due_clock, op_seq) among active with due_clock <= clock
            int best = -1;
            for (int i = 0; i < pend_n; ++i) {
                if (!pend[(size_t)i].active) continue;
                if (pend[(size_t)i].due_clock > clock) continue;
                if (best < 0 ||
                    pend[(size_t)i].due_clock < pend[(size_t)best].due_clock ||
                    (pend[(size_t)i].due_clock == pend[(size_t)best].due_clock &&
                     pend[(size_t)i].op_seq < pend[(size_t)best].op_seq)) {
                    best = i;
                }
            }
            if (best < 0) break;
            OrPend p = pend[(size_t)best];
            pend[(size_t)best].active = false; // remove
            if (p.kind == MKPS_PEND_LOAD_DONE) process_load_done(op_index, p);
            else if (p.kind == MKPS_PEND_COMPUTE_DONE) process_compute_done(op_index, p);
            else process_store_done(op_index, p);
            ++processed;
        }
        compact_pending();
    }

    void compact_pending() {
        int w = 0;
        for (int i = 0; i < pend_n; ++i)
            if (pend[(size_t)i].active) pend[(size_t)w++] = pend[(size_t)i];
        for (int i = w; i < pend_n; ++i) pend[(size_t)i] = OrPend{};
        pend_n = w;
    }

    // ---- CANCEL ----
    void do_cancel(uint32_t op_index, uint32_t id) {
        int si = find_instr(id);
        if (si < 0 || !nonterminal(instrs[(size_t)si])) {
            counts[MKC_invalid_count] += 1;
            emit(MKEV_INVALID, op_index, (uint64_t)id, MKPS_NO_U32, MKPS_NO_U32, 0,
                 MKPS_NO_U32, 0);
            return;
        }
        OrInstr& in = instrs[(size_t)si];
        in.status = MKPS_ST_CANCELLED;
        // remove pending ops logically
        for (int i = 0; i < pend_n; ++i)
            if (pend[(size_t)i].active && pend[(size_t)i].instr_id == id)
                pend[(size_t)i].active = false;
        // release owned nonresident buffers in buffer-id order
        for (int b = 0; b < B; ++b) {
            OrBuf& bf = bufs[(size_t)b];
            if (bf.owner_instr == id && bf.state != MKPS_BUF_FREE && bf.state != MKPS_BUF_RESIDENT) {
                uint32_t tl = bf.tile;
                if (tl < (uint32_t)T && tiles[(size_t)tl].resident_buffer == (uint32_t)b)
                    tiles[(size_t)tl].resident_buffer = MKPS_NO_U32;
                bf = OrBuf{};
                counts[MKC_buffer_cancel_release] += 1;
                emit(MKEV_BUFFER_CANCEL_RELEASE, op_index, (uint64_t)id, tl, (uint32_t)b, 0,
                     MKPS_NO_U32, 0);
            }
        }
        // decrement pending reader counts if reads not yet released
        if (!in.last_use_released) {
            for (int r = 0; r < in.read_count; ++r) {
                OrTile& t = tiles[(size_t)in.reads[r]];
                if (t.pending_reader_count > 0) t.pending_reader_count -= 1;
            }
        }
        // clear writer for write tiles whose writer is this instr
        for (int w = 0; w < in.write_count; ++w) {
            OrTile& t = tiles[(size_t)in.writes[w]];
            if (t.writer_instr == id) t.writer_instr = 0;
        }
        counts[MKC_instr_cancel] += 1;
        emit(MKEV_INSTR_CANCEL, op_index, (uint64_t)id, MKPS_NO_U32, MKPS_NO_U32, 0,
             MKPS_NO_U32, 0);
        compact_pending();
    }

    // ---- HOST_COUNTER_INC ----
    void do_host_counter(uint32_t op_index, uint32_t counter_id, uint32_t amount) {
        if (counter_id >= (uint32_t)NC) {
            counts[MKC_invalid_count] += 1;
            emit(MKEV_INVALID, op_index, 0, MKPS_NO_U32, MKPS_NO_U32, 0,
                 (counter_id < (uint32_t)NC) ? counter_id : MKPS_NO_U32, 0);
            return;
        }
        counter[(size_t)counter_id] += (uint64_t)amount;
        counts[MKC_host_counter_inc] += 1;
        emit(MKEV_HOST_COUNTER_INC, op_index, 0, MKPS_NO_U32, MKPS_NO_U32, 0,
             counter_id, counter[(size_t)counter_id]);
    }

    // ---- hashes ----
    uint64_t instr_hash_compute() const {
        uint64_t h = MKPS_FNV_OFF;
        std::vector<int> idx;
        for (int i = 0; i < NI; ++i)
            if (instrs[(size_t)i].used &&
                instrs[(size_t)i].status != MKPS_ST_DONE &&
                instrs[(size_t)i].status != MKPS_ST_CANCELLED)
                idx.push_back(i);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return instrs[(size_t)a].instr_seq < instrs[(size_t)b].instr_seq;
        });
        for (int si : idx) {
            const OrInstr& in = instrs[(size_t)si];
            uint32_t id = in.instr_id;
            uint64_t seq = in.instr_seq;
            uint8_t st = in.status;
            uint32_t rc = (uint32_t)in.read_count, wc = (uint32_t)in.write_count;
            uint32_t sp = (uint32_t)in.scratch_pages;
            uint8_t cdf = in.compute_done_flag, lur = in.last_use_released;
            mkps_fnv_bytes(&h, &id, sizeof(uint32_t));
            mkps_fnv_bytes(&h, &seq, sizeof(uint64_t));
            mkps_fnv_bytes(&h, &st, sizeof(uint8_t));
            mkps_fnv_bytes(&h, &rc, sizeof(uint32_t));
            mkps_fnv_bytes(&h, &wc, sizeof(uint32_t));
            mkps_fnv_bytes(&h, &sp, sizeof(uint32_t));
            mkps_fnv_bytes(&h, &cdf, sizeof(uint8_t));
            mkps_fnv_bytes(&h, &lur, sizeof(uint8_t));
            for (int r = 0; r < in.read_count; ++r) {
                uint32_t tl = in.reads[r]; uint64_t v = in.read_versions[r];
                mkps_fnv_bytes(&h, &tl, sizeof(uint32_t));
                mkps_fnv_bytes(&h, &v, sizeof(uint64_t));
            }
            for (int w = 0; w < in.write_count; ++w) {
                uint32_t tl = in.writes[w]; uint64_t v = in.write_versions[w];
                mkps_fnv_bytes(&h, &tl, sizeof(uint32_t));
                mkps_fnv_bytes(&h, &v, sizeof(uint64_t));
            }
            uint32_t nb = (uint32_t)in.nbuf;
            mkps_fnv_bytes(&h, &nb, sizeof(uint32_t));
            for (int k = 0; k < in.nbuf; ++k) {
                uint32_t b = in.buf_id[k]; uint8_t role = in.buf_role[k];
                mkps_fnv_bytes(&h, &b, sizeof(uint32_t));
                mkps_fnv_bytes(&h, &role, sizeof(uint8_t));
            }
        }
        return h;
    }

    uint64_t tile_hash_compute() const {
        uint64_t h = MKPS_FNV_OFF;
        for (int l = 0; l < T; ++l) {
            const OrTile& t = tiles[(size_t)l];
            uint32_t tl = (uint32_t)l;
            uint64_t cv = t.current_version;
            uint64_t wi = (uint64_t)t.writer_instr;
            uint64_t prc = t.pending_reader_count;
            uint64_t lss = t.last_store_seq;
            uint32_t rb = t.resident_buffer;
            uint8_t d = t.dirty;
            mkps_fnv_bytes(&h, &tl, sizeof(uint32_t));
            mkps_fnv_bytes(&h, &cv, sizeof(uint64_t));
            mkps_fnv_bytes(&h, &wi, sizeof(uint64_t));
            mkps_fnv_bytes(&h, &prc, sizeof(uint64_t));
            mkps_fnv_bytes(&h, &lss, sizeof(uint64_t));
            mkps_fnv_bytes(&h, &rb, sizeof(uint32_t));
            mkps_fnv_bytes(&h, &d, sizeof(uint8_t));
        }
        return h;
    }

    uint64_t buffer_hash_compute() const {
        uint64_t h = MKPS_FNV_OFF;
        for (int b = 0; b < B; ++b) {
            const OrBuf& bf = bufs[(size_t)b];
            uint32_t id = (uint32_t)b;
            uint8_t st = bf.state;
            uint32_t tl = bf.tile;
            uint64_t ver = bf.version;
            uint64_t own = (uint64_t)bf.owner_instr;
            uint64_t pin = bf.pin_count;
            uint64_t lt = bf.last_touch_seq;
            mkps_fnv_bytes(&h, &id, sizeof(uint32_t));
            mkps_fnv_bytes(&h, &st, sizeof(uint8_t));
            mkps_fnv_bytes(&h, &tl, sizeof(uint32_t));
            mkps_fnv_bytes(&h, &ver, sizeof(uint64_t));
            mkps_fnv_bytes(&h, &own, sizeof(uint64_t));
            mkps_fnv_bytes(&h, &pin, sizeof(uint64_t));
            mkps_fnv_bytes(&h, &lt, sizeof(uint64_t));
        }
        return h;
    }

    uint64_t pending_hash_compute() const {
        uint64_t h = MKPS_FNV_OFF;
        for (int i = 0; i < pend_n; ++i) {
            const OrPend& p = pend[(size_t)i];
            if (!p.active) continue;
            uint8_t kind = p.kind;
            uint64_t due = p.due_clock;
            uint64_t seq = p.op_seq;
            uint64_t iid = (uint64_t)p.instr_id;
            uint32_t bid = p.buffer_id;
            mkps_fnv_bytes(&h, &kind, sizeof(uint8_t));
            mkps_fnv_bytes(&h, &due, sizeof(uint64_t));
            mkps_fnv_bytes(&h, &seq, sizeof(uint64_t));
            mkps_fnv_bytes(&h, &iid, sizeof(uint64_t));
            mkps_fnv_bytes(&h, &bid, sizeof(uint32_t));
        }
        return h;
    }

    uint64_t counter_hash_compute() const {
        uint64_t h = MKPS_FNV_OFF;
        for (int c = 0; c < NC; ++c) {
            uint32_t id = (uint32_t)c;
            uint64_t v = counter[(size_t)c];
            mkps_fnv_bytes(&h, &id, sizeof(uint32_t));
            mkps_fnv_bytes(&h, &v, sizeof(uint64_t));
        }
        return h;
    }

    uint64_t state_checksum_compute(uint64_t ih, uint64_t th, uint64_t bh,
                                    uint64_t ph, uint64_t cnh) const {
        uint64_t h = MKPS_FNV_OFF;
        int b = B, t = T, ni = NI, win = WIN, mp = MP, nc = NC;
        mkps_fnv_bytes(&h, &b, sizeof(int32_t));
        mkps_fnv_bytes(&h, &t, sizeof(int32_t));
        mkps_fnv_bytes(&h, &ni, sizeof(int32_t));
        mkps_fnv_bytes(&h, &win, sizeof(int32_t));
        mkps_fnv_bytes(&h, &mp, sizeof(int32_t));
        mkps_fnv_bytes(&h, &nc, sizeof(int32_t));
        uint64_t c = clock, es = event_seq, isn = instr_seq_next, osn = op_seq_next, vsn = version_seq_next;
        mkps_fnv_bytes(&h, &c, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &es, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &isn, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &osn, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &vsn, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &ih, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &th, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &bh, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &ph, sizeof(uint64_t));
        mkps_fnv_bytes(&h, &cnh, sizeof(uint64_t));
        for (int i = 0; i < MKPS_COUNT_FIELDS; ++i) {
            int64_t v = counts[(size_t)i];
            mkps_fnv_bytes(&h, &v, sizeof(int64_t));
        }
        return h;
    }

    void step_once(const MkpsRunSpec& run, const MkpsHostInputsView& in,
                   MkpsExpected* expected) {
        for (int i = 0; i < run.batch_size; ++i) {
            const int op = in.op[i];
            const uint32_t a0 = in.a0[i], a1 = in.a1[i], a2 = in.a2[i], a3 = in.a3[i];
            const uint32_t a4 = in.a4[i], a5 = in.a5[i], a6 = in.a6[i], a7 = in.a7[i];
            const uint64_t a8 = in.a8[i];
            const uint32_t* reads = &in.tiles[(size_t)i * 16];
            const uint32_t* writes = &in.tiles[(size_t)i * 16 + 8];
            if (op == MKPS_OP_ENQUEUE)
                do_enqueue((uint32_t)i, a0, a1, a2, a3, a4, a5, a6, a7, a8, reads, writes);
            else if (op == MKPS_OP_ISSUE_LOADS) do_issue_loads((uint32_t)i, a0);
            else if (op == MKPS_OP_ISSUE_COMPUTE) do_issue_compute((uint32_t)i, a0);
            else if (op == MKPS_OP_ISSUE_STORES) do_issue_stores((uint32_t)i, a0);
            else if (op == MKPS_OP_ADVANCE) do_advance((uint32_t)i, a0, a1);
            else if (op == MKPS_OP_CANCEL) do_cancel((uint32_t)i, a0);
            else if (op == MKPS_OP_HOST_COUNTER) do_host_counter((uint32_t)i, a0, a1);
            else {
                counts[MKC_invalid_count] += 1;
                emit(MKEV_INVALID, (uint32_t)i, 0, MKPS_NO_U32, MKPS_NO_U32, 0,
                     MKPS_NO_U32, 0);
            }
        }
        uint64_t ih = instr_hash_compute();
        uint64_t th = tile_hash_compute();
        uint64_t bh = buffer_hash_compute();
        uint64_t ph = pending_hash_compute();
        uint64_t cnh = counter_hash_compute();
        expected->counts = counts;
        expected->pipeline_event_hash = running_event_hash;
        expected->instr_hash = ih;
        expected->tile_hash = th;
        expected->buffer_hash = bh;
        expected->pending_hash = ph;
        expected->counter_hash = cnh;
        expected->event_seq_out = event_seq;
        expected->state_checksum = state_checksum_compute(ih, th, bh, ph, cnh);
    }
};

static inline bool mkps_check_all_outputs(
    const MkpsProblemSpec& spec,
    const MkpsExpected& expected,
    const MkpsHostOutputsView& got,
    std::string* error) {
    (void)spec;
    for (int i = 0; i < MKPS_COUNT_FIELDS; ++i) {
        if (got.counts[i] != expected.counts[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "counts[" << i << "] mismatch: got " << got.counts[i]
                    << ", expected " << expected.counts[(size_t)i];
                *error = oss.str();
            }
            return false;
        }
    }
    struct HashField { const char* name; uint64_t got; uint64_t exp; };
    HashField fields[] = {
        {"pipeline_event_hash", got.pipeline_event_hash[0], expected.pipeline_event_hash},
        {"instr_hash", got.instr_hash[0], expected.instr_hash},
        {"tile_hash", got.tile_hash[0], expected.tile_hash},
        {"buffer_hash", got.buffer_hash[0], expected.buffer_hash},
        {"pending_hash", got.pending_hash[0], expected.pending_hash},
        {"counter_hash", got.counter_hash[0], expected.counter_hash},
        {"event_seq_out", got.event_seq_out[0], expected.event_seq_out},
        {"state_checksum", got.state_checksum[0], expected.state_checksum},
    };
    for (const HashField& f : fields) {
        if (f.got != f.exp) {
            if (error) {
                std::ostringstream oss;
                oss << f.name << " mismatch: got 0x" << std::hex << f.got
                    << ", expected 0x" << f.exp;
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

/*
GRADER MODEL
  oracle.init(problem_spec)
  solution_reset + oracle.reset()
  for each step:
    solution_run(...)
    oracle.step_once(...)
    mkps_check_all_outputs(...)

Required adversarial coverage:
  - software pipeline: enqueue several instrs, issue loads early while earlier
    instrs still store; double/triple buffer slot acquire/release.
  - RAW: read of a tile whose earlier writer has not stored -> hazard stall, scan
    stops; after store completes, load issues and reuses resident buffer.
  - WAW: two instrs writing same tile -> later stalls until earlier terminal.
  - WAR: write of a tile still being read by an unreleased earlier reader -> stall.
  - page-capacity stall permits scanning later instructions (out-of-order issue).
  - ADVANCE ordering by (due_clock, op_seq); stale ops after cancel -> stale drop.
  - cancel releases buffers + decrements reader counts; counters after stores.
  - reset and exact replay.
*/

#endif  // MK_PIPELINE_SCOREBOARD_ORACLE_HPP_
