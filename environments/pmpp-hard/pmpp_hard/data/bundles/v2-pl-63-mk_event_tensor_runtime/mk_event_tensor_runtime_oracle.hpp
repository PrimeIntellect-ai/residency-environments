// file: mk_event_tensor_runtime_oracle.hpp
//
// CPU reference (oracle) for MK4 Event-Tensor Pop/Push Runtime. Uses std::
// containers (std::map / std::vector) and shares no algorithm code with the GPU
// reference or naive implementations. This is the source of truth for the coupled
// semantics: event-tensor cell decrements, first-time zero-decrement unblocking
// (push dependents in task_seq order), idle pop-on-ready, dynamic-mask
// cancellation at completion, lazy stale-drop of ready/popped queues, and all
// hashes.
//
// CANONICAL FIELD CONVENTIONS for mk_event_hash (a record is emitted for every
// event; event_seq consumed per event):
//   event_kind:u8; event_seq:u64; op_index:u32; tensor_or_U32MAX:u32;
//   i_or_U32MAX:u32; j_or_U32MAX:u32; task_id_or_ZERO:u64;
//   scheduler_or_U32MAX:u32; count_or_U64MAX:u64.
// Per-event population (any field not meaningful -> its sentinel):
//   TENSOR_DEFINE:         tensor; i=MAX; j=MAX; task=0; sched=MAX;
//                          count=generation.
//   DEFINE_STALL:          tensor; i=MAX; j=MAX; task=0; sched=MAX; count=MAX.
//   TASK_READY_REGISTER:   tensor=MAX; i=MAX; j=MAX; task; sched=scheduler_hint;
//                          count=ready_seq.
//   TASK_BLOCKED_REGISTER: tensor=MAX; i=MAX; j=MAX; task; sched=scheduler_hint;
//                          count=MAX.
//   CELL_DECREMENT:        tensor; i; j; task=0; sched=MAX; count=remaining_after.
//   OUTPUT_DECREMENT:      tensor; i; j; task=completing task; sched=MAX;
//                          count=remaining_after.
//   CELL_ZERO:             tensor; i; j; task=0; sched=MAX; count=0.
//   CELL_ALREADY_ZERO:     tensor; i; j; task=0; sched=MAX; count=0.
//   TASK_READY_PUSH:       tensor=MAX; i=MAX; j=MAX; task; sched=scheduler_hint;
//                          count=ready_seq.
//   READY_STALE_DROP:      tensor=MAX; i=MAX; j=MAX; task; sched=scheduler;
//                          count=ready_seq_of_entry.
//   TASK_POP:              tensor=MAX; i=MAX; j=MAX; task; sched=scheduler;
//                          count=pop_seq.
//   TASK_START:            tensor=MAX; i=MAX; j=MAX; task; sched=scheduler;
//                          count=pop_seq.
//   POPPED_STALE_DROP:     tensor=MAX; i=MAX; j=MAX; task; sched=scheduler;
//                          count=pop_seq_of_entry.
//   TASK_COMPLETE:         tensor=MAX; i=MAX; j=MAX; task; sched=MAX; count=MAX.
//   TASK_DYNAMIC_CANCEL:   tensor=MAX; i=MAX; j=MAX; task; sched=MAX;
//                          count=observed_mask.
//   CELL_CANCEL:           tensor; i; j; task=0; sched=MAX; count=MAX.
//   TASK_CANCEL_BY_CELL:   tensor; i; j; task; sched=MAX; count=MAX.
//   TASK_CANCEL_EXPLICIT:  tensor=MAX; i=MAX; j=MAX; task; sched=MAX; count=MAX.
//   INVALID:               tensor=MAX; i=MAX; j=MAX; task=task_id_field; sched=MAX;
//                          count=MAX.

#ifndef MK_EVENT_TENSOR_RUNTIME_ORACLE_HPP_
#define MK_EVENT_TENSOR_RUNTIME_ORACLE_HPP_

#include "mk_event_tensor_runtime_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <deque>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct MkHostInputsView {
    const int32_t*  op_kind;
    const uint32_t* tensor_id;
    const uint32_t* ci;
    const uint32_t* cj;
    const uint32_t* rank;
    const uint32_t* dim0;
    const uint32_t* dim1;
    const uint64_t* task_id;
    const uint32_t* sched;
    const uint64_t* amount;
    const uint64_t* mask;
    const uint64_t* payload;
    const uint32_t* dep_off;
    const uint32_t* dep_count;
    const uint32_t* out_off;
    const uint32_t* out_count;
    const uint32_t* dep_tensor;
    const uint32_t* dep_i;
    const uint32_t* dep_j;
    const uint32_t* out_tensor;
    const uint32_t* out_i;
    const uint32_t* out_j;
    const uint64_t* out_amount;
};

struct MkHostOutputsView {
    const int32_t* tensors_defined;
    const int32_t* define_stall;
    const int32_t* tasks_ready_register;
    const int32_t* tasks_blocked_register;
    const int32_t* cell_decrements;
    const int32_t* cell_zero;
    const int32_t* cell_already_zero;
    const int32_t* tasks_ready_push;
    const int32_t* tasks_popped;
    const int32_t* tasks_started;
    const int32_t* tasks_completed;
    const int32_t* dynamic_cancelled;
    const int32_t* cell_cancelled;
    const int32_t* tasks_cancel_by_cell;
    const int32_t* tasks_cancel_explicit;
    const int32_t* ready_stale_drop;
    const int32_t* popped_stale_drop;
    const int32_t* invalid_count;
    const uint64_t* mk_event_hash;
    const uint64_t* cell_hash;
    const uint64_t* task_hash;
    const uint64_t* queue_hash;
    const uint64_t* reverse_dep_hash;
};

struct MkExpected {
    int32_t tensors_defined = 0, define_stall = 0, tasks_ready_register = 0;
    int32_t tasks_blocked_register = 0, cell_decrements = 0, cell_zero = 0;
    int32_t cell_already_zero = 0, tasks_ready_push = 0, tasks_popped = 0;
    int32_t tasks_started = 0, tasks_completed = 0, dynamic_cancelled = 0;
    int32_t cell_cancelled = 0, tasks_cancel_by_cell = 0, tasks_cancel_explicit = 0;
    int32_t ready_stale_drop = 0, popped_stale_drop = 0, invalid_count = 0;
    uint64_t mk_event_hash = MK_FNV_OFFSET;
    uint64_t cell_hash = MK_FNV_OFFSET;
    uint64_t task_hash = MK_FNV_OFFSET;
    uint64_t queue_hash = MK_FNV_OFFSET;
    uint64_t reverse_dep_hash = MK_FNV_OFFSET;
};

static inline uint64_t mk_o_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= MK_FNV_PRIME; return h;
}
static inline void mk_o_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* q = (const uint8_t*)p; uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = mk_o_fnv_byte(v, q[i]);
    *h = v;
}
static inline void mk_o_u8(uint64_t* h, uint8_t v)   { mk_o_bytes(h, &v, 1); }
static inline void mk_o_u32(uint64_t* h, uint32_t v) { mk_o_bytes(h, &v, 4); }
static inline void mk_o_u64(uint64_t* h, uint64_t v) { mk_o_bytes(h, &v, 8); }

// ---- persistent model objects ----

struct MkOCell {
    uint64_t initial_count = 0;
    uint64_t remaining_count = 0;
    uint8_t  zero_pushed = 0;
    uint64_t cell_seq = 0;
    uint64_t last_decrement_seq = 0;
    uint8_t  cancelled = 0;
    std::vector<uint64_t> deps;  // dependent task ids, ordered by task_seq
};

struct MkOTensor {
    uint8_t  defined = 0;
    uint8_t  rank = 0;
    uint32_t dim0 = 0;
    uint32_t dim1 = 0;
    uint64_t generation = 0;
    std::vector<MkOCell> cells;  // linear index i*max(dim1,1)+j
};

struct MkODepRef { uint32_t tensor, i, j; };
struct MkOOutRef { uint32_t tensor, i, j; uint64_t amount; };

struct MkOTask {
    uint64_t task_id = 0;
    uint64_t task_seq = 0;
    uint32_t scheduler_hint = 0;
    uint8_t  status = MK_ST_BLOCKED;
    uint64_t dynamic_mask = 0;
    uint64_t payload_seed = 0;
    uint64_t ready_seq = 0;  // 0 means none
    uint64_t pop_seq = 0;    // 0 means none
    std::vector<MkODepRef> deps;
    std::vector<MkOOutRef> outs;
};

struct MkOQueueEntry {
    uint64_t task_id;
    uint64_t seq;  // ready_seq (ready queue) or pop_seq (popped list)
};

struct MkOracleState {
    MkProblemSpec spec{};

    uint64_t event_seq = 0;
    uint64_t task_seq_next = 1;
    uint64_t cell_seq_next = 1;
    uint64_t ready_seq_next = 1;
    uint64_t pop_seq_next = 1;
    uint64_t op_index = 0;

    std::vector<MkOTensor> tensors;
    std::map<uint64_t, MkOTask> tasks;
    std::vector<std::deque<MkOQueueEntry>> ready_q;   // per scheduler
    std::vector<std::deque<MkOQueueEntry>> popped_q;  // per scheduler

    void init(const MkProblemSpec& s) { spec = s; reset(); }

    void reset() {
        event_seq = 0; task_seq_next = 1; cell_seq_next = 1; ready_seq_next = 1;
        pop_seq_next = 1; op_index = 0;
        tensors.assign((size_t)spec.tensor_count, MkOTensor());
        tasks.clear();
        ready_q.assign((size_t)spec.scheduler_count, std::deque<MkOQueueEntry>());
        popped_q.assign((size_t)spec.scheduler_count, std::deque<MkOQueueEntry>());
    }

    // ---- cell access ----
    bool tensor_in_range(uint32_t t) const { return t < (uint32_t)spec.tensor_count; }

    int cell_index(const MkOTensor& tn, uint32_t i, uint32_t j) const {
        uint32_t d1 = tn.dim1 > 0 ? tn.dim1 : 1u;
        if (i >= tn.dim0) return -1;
        if (j >= d1) return -1;
        return (int)((uint64_t)i * (uint64_t)d1 + (uint64_t)j);
    }

    // Returns pointer to existing cell or nullptr.
    MkOCell* find_cell(uint32_t t, uint32_t i, uint32_t j) {
        if (!tensor_in_range(t)) return nullptr;
        MkOTensor& tn = tensors[t];
        if (!tn.defined) return nullptr;
        int idx = cell_index(tn, i, j);
        if (idx < 0) return nullptr;
        return &tn.cells[(size_t)idx];
    }

    // ---- event emission ----
    void emit(MkExpected* e, uint8_t kind, uint32_t op_idx, uint32_t tensor,
              uint32_t i, uint32_t j, uint64_t task_id, uint32_t sched,
              uint64_t count) {
        uint64_t* h = &e->mk_event_hash;
        mk_o_u8(h, kind);
        mk_o_u64(h, event_seq);
        mk_o_u32(h, op_idx);
        mk_o_u32(h, tensor);
        mk_o_u32(h, i);
        mk_o_u32(h, j);
        mk_o_u64(h, task_id);
        mk_o_u32(h, sched);
        mk_o_u64(h, count);
        event_seq += 1;
    }

    void emit_invalid(MkExpected* e, uint32_t op_idx, uint64_t task_id) {
        emit(e, MK_EVT_INVALID, op_idx, MK_U32_MAX, MK_U32_MAX, MK_U32_MAX,
             task_id, MK_U32_MAX, MK_U64_MAX);
        e->invalid_count += 1;
    }

    // Are all dependency cells of task t at zero and not cancelled?
    bool all_deps_zero(const MkOTask& t) {
        for (const MkODepRef& d : t.deps) {
            MkOCell* c = find_cell(d.tensor, d.i, d.j);
            if (!c) return false;  // absent cell can never be "zero" -> stays blocked
            if (c->cancelled) return false;
            if (c->remaining_count != 0) return false;
        }
        return true;
    }

    // Scan a freshly-zeroed cell's dependents (task_seq order); push newly-ready.
    void scan_and_push(MkExpected* e, uint32_t op_idx, MkOCell* c) {
        // c->deps already ordered by task_seq (insertion order = task_seq order).
        std::vector<uint64_t> snapshot = c->deps;
        for (uint64_t tid : snapshot) {
            auto it = tasks.find(tid);
            if (it == tasks.end()) continue;
            MkOTask& t = it->second;
            if (t.status != MK_ST_BLOCKED) continue;
            if (!all_deps_zero(t)) continue;
            t.status = MK_ST_READY;
            t.ready_seq = ready_seq_next++;
            t.pop_seq = 0;
            MkOQueueEntry qe{t.task_id, t.ready_seq};
            ready_q[t.scheduler_hint].push_back(qe);
            emit(e, MK_EVT_TASK_READY_PUSH, op_idx, MK_U32_MAX, MK_U32_MAX,
                 MK_U32_MAX, t.task_id, t.scheduler_hint, t.ready_seq);
            e->tasks_ready_push += 1;
        }
    }

    // Apply one decrement to an existing, non-cancelled cell. event_kind selects
    // CELL_DECREMENT vs OUTPUT_DECREMENT. task_for_event is the task id for the
    // event record (0 for SIGNAL_CELL). Returns nothing; updates counts/events.
    void apply_decrement(MkExpected* e, uint32_t op_idx, uint32_t t, uint32_t i,
                         uint32_t j, uint64_t amount, uint8_t event_kind,
                         uint64_t task_for_event) {
        MkOCell* c = find_cell(t, i, j);
        // caller guarantees c != nullptr and !cancelled and amount != 0.
        bool was_zero = (c->remaining_count == 0);
        if (amount >= c->remaining_count) c->remaining_count = 0;
        else c->remaining_count -= amount;
        // emit decrement; set last_decrement_seq to this event's seq.
        c->last_decrement_seq = event_seq;
        emit(e, event_kind, op_idx, t, i, j, task_for_event, MK_U32_MAX,
             c->remaining_count);
        e->cell_decrements += 1;

        if (c->remaining_count == 0 && !was_zero && c->zero_pushed == 0) {
            c->zero_pushed = 1;
            emit(e, MK_EVT_CELL_ZERO, op_idx, t, i, j, 0, MK_U32_MAX, 0);
            e->cell_zero += 1;
            scan_and_push(e, op_idx, c);
        } else if (was_zero) {
            // cell was already zero before this decrement.
            emit(e, MK_EVT_CELL_ALREADY_ZERO, op_idx, t, i, j, 0, MK_U32_MAX, 0);
            e->cell_already_zero += 1;
        }
        // (If remaining_count just hit 0 but zero_pushed was already 1 -- only
        //  possible if it was zero before, handled by the was_zero branch.)
    }

    // ---- op handlers ----

    void op_define_tensor(MkExpected* e, uint32_t op_idx, uint32_t t, uint32_t rank,
                          uint32_t dim0, uint32_t dim1, uint64_t initial_count) {
        if (!tensor_in_range(t)) { emit_invalid(e, op_idx, 0); return; }
        if (rank != 1 && rank != 2) { emit_invalid(e, op_idx, 0); return; }
        if (dim0 == 0) { emit_invalid(e, op_idx, 0); return; }
        if (rank == 2 && dim1 == 0) { emit_invalid(e, op_idx, 0); return; }
        uint32_t d1 = (rank == 2) ? dim1 : 1u;
        uint64_t total = (uint64_t)dim0 * (uint64_t)d1;
        if (total > (uint64_t)spec.max_cells_per_tensor) { emit_invalid(e, op_idx, 0); return; }

        MkOTensor& tn = tensors[t];
        if (tn.defined) {
            // Stall if any cell has a dependent nonterminal task.
            bool stall = false;
            for (const MkOCell& c : tn.cells) {
                for (uint64_t tid : c.deps) {
                    auto it = tasks.find(tid);
                    if (it == tasks.end()) continue;
                    uint8_t st = it->second.status;
                    if (st == MK_ST_BLOCKED || st == MK_ST_READY ||
                        st == MK_ST_POPPED || st == MK_ST_RUNNING) { stall = true; break; }
                }
                if (stall) break;
            }
            if (stall) {
                emit(e, MK_EVT_DEFINE_STALL, op_idx, t, MK_U32_MAX, MK_U32_MAX, 0,
                     MK_U32_MAX, MK_U64_MAX);
                e->define_stall += 1;
                return;
            }
        }

        // (Re)define.
        tn.defined = 1;
        tn.rank = (uint8_t)rank;
        tn.dim0 = dim0;
        tn.dim1 = (rank == 2) ? dim1 : 0u;  // store 0 for rank-1
        tn.generation += 1;
        uint32_t d1s = (rank == 2) ? dim1 : 1u;
        size_t ncells = (size_t)dim0 * (size_t)d1s;
        tn.cells.assign(ncells, MkOCell());
        for (size_t k = 0; k < ncells; ++k) {
            MkOCell& c = tn.cells[k];
            c.initial_count = initial_count;
            c.remaining_count = initial_count;
            c.zero_pushed = (initial_count == 0) ? 1 : 0;
            c.cell_seq = cell_seq_next++;
            c.last_decrement_seq = 0;
            c.cancelled = 0;
            c.deps.clear();
        }
        emit(e, MK_EVT_TENSOR_DEFINE, op_idx, t, MK_U32_MAX, MK_U32_MAX, 0,
             MK_U32_MAX, tn.generation);
        e->tensors_defined += 1;
    }

    void op_register_task(MkExpected* e, uint32_t op_idx, uint64_t task_id,
                          uint32_t sched_hint, const std::vector<MkODepRef>& deps,
                          const std::vector<MkOOutRef>& outs, uint64_t dynamic_mask,
                          uint64_t payload_seed) {
        if (tasks.count(task_id) != 0) { emit_invalid(e, op_idx, task_id); return; }
        if ((int)tasks.size() >= spec.max_tasks) { emit_invalid(e, op_idx, task_id); return; }
        if (sched_hint >= (uint32_t)spec.scheduler_count) { emit_invalid(e, op_idx, task_id); return; }
        if ((int)deps.size() > spec.max_task_deps) { emit_invalid(e, op_idx, task_id); return; }
        if ((int)outs.size() > spec.max_task_outputs) { emit_invalid(e, op_idx, task_id); return; }
        // All dep/output cells must exist and not be cancelled.
        for (const MkODepRef& d : deps) {
            MkOCell* c = find_cell(d.tensor, d.i, d.j);
            if (!c || c->cancelled) { emit_invalid(e, op_idx, task_id); return; }
        }
        for (const MkOOutRef& o : outs) {
            MkOCell* c = find_cell(o.tensor, o.i, o.j);
            if (!c || c->cancelled) { emit_invalid(e, op_idx, task_id); return; }
        }

        MkOTask t;
        t.task_id = task_id;
        t.task_seq = task_seq_next++;
        t.scheduler_hint = sched_hint;
        t.dynamic_mask = dynamic_mask;
        t.payload_seed = payload_seed;
        t.ready_seq = 0;
        t.pop_seq = 0;
        t.deps = deps;
        t.outs = outs;

        // Insert into reverse-dep index of every dependency cell (ordered by
        // task_seq; task_seq monotone so append keeps order).
        for (const MkODepRef& d : deps) {
            MkOCell* c = find_cell(d.tensor, d.i, d.j);
            c->deps.push_back(task_id);
        }

        bool ready = all_deps_zero(t);
        if (ready) {
            t.status = MK_ST_READY;
            t.ready_seq = ready_seq_next++;
            tasks[task_id] = t;
            MkOQueueEntry qe{task_id, t.ready_seq};
            ready_q[sched_hint].push_back(qe);
            emit(e, MK_EVT_TASK_READY_REGISTER, op_idx, MK_U32_MAX, MK_U32_MAX,
                 MK_U32_MAX, task_id, sched_hint, t.ready_seq);
            e->tasks_ready_register += 1;
        } else {
            t.status = MK_ST_BLOCKED;
            tasks[task_id] = t;
            emit(e, MK_EVT_TASK_BLOCKED_REGISTER, op_idx, MK_U32_MAX, MK_U32_MAX,
                 MK_U32_MAX, task_id, sched_hint, MK_U64_MAX);
            e->tasks_blocked_register += 1;
        }
    }

    void op_signal_cell(MkExpected* e, uint32_t op_idx, uint32_t t, uint32_t i,
                        uint32_t j, uint64_t amount) {
        MkOCell* c = find_cell(t, i, j);
        if (!c || c->cancelled || amount == 0) { emit_invalid(e, op_idx, 0); return; }
        apply_decrement(e, op_idx, t, i, j, amount, MK_EVT_CELL_DECREMENT, 0);
    }

    void op_pop_tasks(MkExpected* e, uint32_t op_idx, uint32_t scheduler, uint64_t limit) {
        if (scheduler >= (uint32_t)spec.scheduler_count) { emit_invalid(e, op_idx, 0); return; }
        if (limit == 0) return;  // valid no-op
        std::deque<MkOQueueEntry>& q = ready_q[scheduler];
        uint64_t popped = 0;
        while (popped < limit && !q.empty()) {
            MkOQueueEntry ent = q.front();
            auto it = tasks.find(ent.task_id);
            bool valid = (it != tasks.end()) && (it->second.status == MK_ST_READY) &&
                         (it->second.ready_seq == ent.seq);
            if (!valid) {
                q.pop_front();
                emit(e, MK_EVT_READY_STALE_DROP, op_idx, MK_U32_MAX, MK_U32_MAX,
                     MK_U32_MAX, ent.task_id, scheduler, ent.seq);
                e->ready_stale_drop += 1;
                continue;
            }
            // valid pop
            q.pop_front();
            MkOTask& t = it->second;
            t.status = MK_ST_POPPED;
            t.pop_seq = pop_seq_next++;
            popped_q[scheduler].push_back(MkOQueueEntry{t.task_id, t.pop_seq});
            emit(e, MK_EVT_TASK_POP, op_idx, MK_U32_MAX, MK_U32_MAX, MK_U32_MAX,
                 t.task_id, scheduler, t.pop_seq);
            e->tasks_popped += 1;
            popped += 1;
        }
    }

    void op_start_popped(MkExpected* e, uint32_t op_idx, uint32_t scheduler, uint64_t limit) {
        if (scheduler >= (uint32_t)spec.scheduler_count) { emit_invalid(e, op_idx, 0); return; }
        if (limit == 0) return;  // valid no-op
        std::deque<MkOQueueEntry>& q = popped_q[scheduler];
        uint64_t started = 0;
        while (started < limit && !q.empty()) {
            MkOQueueEntry ent = q.front();
            auto it = tasks.find(ent.task_id);
            bool valid = (it != tasks.end()) && (it->second.status == MK_ST_POPPED) &&
                         (it->second.pop_seq == ent.seq);
            if (!valid) {
                q.pop_front();
                emit(e, MK_EVT_POPPED_STALE_DROP, op_idx, MK_U32_MAX, MK_U32_MAX,
                     MK_U32_MAX, ent.task_id, scheduler, ent.seq);
                e->popped_stale_drop += 1;
                continue;
            }
            q.pop_front();
            MkOTask& t = it->second;
            t.status = MK_ST_RUNNING;
            emit(e, MK_EVT_TASK_START, op_idx, MK_U32_MAX, MK_U32_MAX, MK_U32_MAX,
                 t.task_id, scheduler, t.pop_seq);
            e->tasks_started += 1;
            started += 1;
        }
    }

    void op_complete_task(MkExpected* e, uint32_t op_idx, uint64_t task_id,
                          uint64_t observed_mask) {
        auto it = tasks.find(task_id);
        if (it == tasks.end() || it->second.status != MK_ST_RUNNING) {
            emit_invalid(e, op_idx, task_id); return;
        }
        MkOTask& t = it->second;
        if ((observed_mask & t.dynamic_mask) != t.dynamic_mask) {
            t.status = MK_ST_CANCELLED;
            emit(e, MK_EVT_TASK_DYNAMIC_CANCEL, op_idx, MK_U32_MAX, MK_U32_MAX,
                 MK_U32_MAX, task_id, MK_U32_MAX, observed_mask);
            e->dynamic_cancelled += 1;
            return;
        }
        t.status = MK_ST_DONE;
        emit(e, MK_EVT_TASK_COMPLETE, op_idx, MK_U32_MAX, MK_U32_MAX, MK_U32_MAX,
             task_id, MK_U32_MAX, MK_U64_MAX);
        e->tasks_completed += 1;
        // Output decrements in listed order (snapshot the output refs: completing
        // does not change the task's own output list).
        std::vector<MkOOutRef> outs = t.outs;
        for (const MkOOutRef& o : outs) {
            MkOCell* c = find_cell(o.tensor, o.i, o.j);
            if (!c || c->cancelled || o.amount == 0) continue;  // skip silently
            apply_decrement(e, op_idx, o.tensor, o.i, o.j, o.amount,
                            MK_EVT_OUTPUT_DECREMENT, task_id);
        }
    }

    void op_cancel_cell(MkExpected* e, uint32_t op_idx, uint32_t t, uint32_t i, uint32_t j) {
        MkOCell* c = find_cell(t, i, j);
        if (!c) { emit_invalid(e, op_idx, 0); return; }
        c->cancelled = 1;
        emit(e, MK_EVT_CELL_CANCEL, op_idx, t, i, j, 0, MK_U32_MAX, MK_U64_MAX);
        e->cell_cancelled += 1;
        // Cancel BLOCKED or READY dependent tasks in task_seq order.
        std::vector<uint64_t> snapshot = c->deps;
        for (uint64_t tid : snapshot) {
            auto it = tasks.find(tid);
            if (it == tasks.end()) continue;
            uint8_t st = it->second.status;
            if (st == MK_ST_BLOCKED || st == MK_ST_READY) {
                it->second.status = MK_ST_CANCELLED;
                emit(e, MK_EVT_TASK_CANCEL_BY_CELL, op_idx, t, i, j, tid,
                     MK_U32_MAX, MK_U64_MAX);
                e->tasks_cancel_by_cell += 1;
            }
        }
    }

    void op_cancel_task(MkExpected* e, uint32_t op_idx, uint64_t task_id) {
        auto it = tasks.find(task_id);
        if (it == tasks.end()) { emit_invalid(e, op_idx, task_id); return; }
        uint8_t st = it->second.status;
        if (st == MK_ST_DONE || st == MK_ST_CANCELLED) { emit_invalid(e, op_idx, task_id); return; }
        it->second.status = MK_ST_CANCELLED;
        emit(e, MK_EVT_TASK_CANCEL_EXPLICIT, op_idx, MK_U32_MAX, MK_U32_MAX,
             MK_U32_MAX, task_id, MK_U32_MAX, MK_U64_MAX);
        e->tasks_cancel_explicit += 1;
    }

    // ---- state hashes ----

    uint64_t cell_hash() const {
        uint64_t h = MK_FNV_OFFSET;
        for (uint32_t t = 0; t < (uint32_t)spec.tensor_count; ++t) {
            const MkOTensor& tn = tensors[t];
            if (!tn.defined) continue;
            uint32_t d1 = tn.dim1 > 0 ? tn.dim1 : 1u;
            for (uint32_t i = 0; i < tn.dim0; ++i) {
                for (uint32_t j = 0; j < d1; ++j) {
                    const MkOCell& c = tn.cells[(size_t)i * d1 + j];
                    mk_o_u32(&h, t);
                    mk_o_u64(&h, tn.generation);
                    mk_o_u32(&h, i);
                    mk_o_u32(&h, j);
                    mk_o_u64(&h, c.initial_count);
                    mk_o_u64(&h, c.remaining_count);
                    mk_o_u8(&h, c.zero_pushed);
                    mk_o_u64(&h, c.cell_seq);
                    mk_o_u64(&h, c.last_decrement_seq);
                    mk_o_u8(&h, c.cancelled);
                }
            }
        }
        return h;
    }

    uint64_t task_hash() const {
        uint64_t h = MK_FNV_OFFSET;
        for (auto it = tasks.begin(); it != tasks.end(); ++it) {
            const MkOTask& t = it->second;
            mk_o_u64(&h, t.task_id);
            mk_o_u64(&h, t.task_seq);
            mk_o_u32(&h, t.scheduler_hint);
            mk_o_u8(&h, t.status);
            mk_o_u64(&h, t.dynamic_mask);
            mk_o_u64(&h, t.payload_seed);
            mk_o_u64(&h, t.ready_seq);
            mk_o_u64(&h, t.pop_seq);
            for (const MkODepRef& d : t.deps) {
                mk_o_u32(&h, d.tensor); mk_o_u32(&h, d.i); mk_o_u32(&h, d.j);
            }
            for (const MkOOutRef& o : t.outs) {
                mk_o_u32(&h, o.tensor); mk_o_u32(&h, o.i); mk_o_u32(&h, o.j);
                mk_o_u64(&h, o.amount);
            }
        }
        return h;
    }

    uint64_t queue_hash() const {
        uint64_t h = MK_FNV_OFFSET;
        for (uint32_t s = 0; s < (uint32_t)spec.scheduler_count; ++s) {
            const std::deque<MkOQueueEntry>& q = ready_q[s];
            for (size_t pos = 0; pos < q.size(); ++pos) {
                mk_o_u8(&h, 0);
                mk_o_u32(&h, s);
                mk_o_u64(&h, (uint64_t)pos);
                mk_o_u64(&h, q[pos].task_id);
                mk_o_u64(&h, q[pos].seq);
            }
        }
        for (uint32_t s = 0; s < (uint32_t)spec.scheduler_count; ++s) {
            const std::deque<MkOQueueEntry>& q = popped_q[s];
            for (size_t pos = 0; pos < q.size(); ++pos) {
                mk_o_u8(&h, 1);
                mk_o_u32(&h, s);
                mk_o_u64(&h, (uint64_t)pos);
                mk_o_u64(&h, q[pos].task_id);
                mk_o_u64(&h, q[pos].seq);
            }
        }
        return h;
    }

    uint64_t reverse_dep_hash() const {
        uint64_t h = MK_FNV_OFFSET;
        for (uint32_t t = 0; t < (uint32_t)spec.tensor_count; ++t) {
            const MkOTensor& tn = tensors[t];
            if (!tn.defined) continue;
            uint32_t d1 = tn.dim1 > 0 ? tn.dim1 : 1u;
            for (uint32_t i = 0; i < tn.dim0; ++i) {
                for (uint32_t j = 0; j < d1; ++j) {
                    const MkOCell& c = tn.cells[(size_t)i * d1 + j];
                    if (c.deps.empty()) continue;
                    mk_o_u32(&h, t);
                    mk_o_u32(&h, i);
                    mk_o_u32(&h, j);
                    for (uint64_t tid : c.deps) mk_o_u64(&h, tid);
                }
            }
        }
        return h;
    }

    void step_once(const MkRunSpec& run, const MkHostInputsView& in, MkExpected* exp) {
        *exp = MkExpected();
        for (int r = 0; r < run.batch_size; ++r) {
            const uint32_t op_idx = (uint32_t)op_index;
            const int32_t kind = in.op_kind[r];
            switch (kind) {
                case MK_OP_DEFINE_TENSOR:
                    op_define_tensor(exp, op_idx, in.tensor_id[r], in.rank[r],
                                     in.dim0[r], in.dim1[r], in.amount[r]);
                    break;
                case MK_OP_REGISTER_TASK: {
                    std::vector<MkODepRef> deps;
                    std::vector<MkOOutRef> outs;
                    uint32_t doff = in.dep_off[r], dcnt = in.dep_count[r];
                    uint32_t ooff = in.out_off[r], ocnt = in.out_count[r];
                    for (uint32_t k = 0; k < dcnt; ++k) {
                        MkODepRef d{in.dep_tensor[doff + k], in.dep_i[doff + k], in.dep_j[doff + k]};
                        deps.push_back(d);
                    }
                    for (uint32_t k = 0; k < ocnt; ++k) {
                        MkOOutRef o{in.out_tensor[ooff + k], in.out_i[ooff + k],
                                    in.out_j[ooff + k], in.out_amount[ooff + k]};
                        outs.push_back(o);
                    }
                    op_register_task(exp, op_idx, in.task_id[r], in.sched[r], deps,
                                     outs, in.mask[r], in.payload[r]);
                    break;
                }
                case MK_OP_SIGNAL_CELL:
                    op_signal_cell(exp, op_idx, in.tensor_id[r], in.ci[r], in.cj[r], in.amount[r]);
                    break;
                case MK_OP_POP_TASKS:
                    op_pop_tasks(exp, op_idx, in.sched[r], in.amount[r]);
                    break;
                case MK_OP_START_POPPED:
                    op_start_popped(exp, op_idx, in.sched[r], in.amount[r]);
                    break;
                case MK_OP_COMPLETE_TASK:
                    op_complete_task(exp, op_idx, in.task_id[r], in.mask[r]);
                    break;
                case MK_OP_CANCEL_CELL:
                    op_cancel_cell(exp, op_idx, in.tensor_id[r], in.ci[r], in.cj[r]);
                    break;
                case MK_OP_CANCEL_TASK:
                    op_cancel_task(exp, op_idx, in.task_id[r]);
                    break;
                default:
                    emit_invalid(exp, op_idx, in.task_id[r]);
                    break;
            }
            op_index += 1;
        }
        exp->cell_hash = cell_hash();
        exp->task_hash = task_hash();
        exp->queue_hash = queue_hash();
        exp->reverse_dep_hash = reverse_dep_hash();
    }
};

static inline bool mk_check_all_outputs(const MkExpected& e,
                                        const MkHostOutputsView& g,
                                        std::string* error) {
#define MK_CHECK_INT(field)                                                    \
    if (g.field[0] != e.field) {                                              \
        if (error) { std::ostringstream o; o << #field " mismatch: got "      \
            << g.field[0] << ", expected " << e.field; *error = o.str(); }     \
        return false; }
#define MK_CHECK_HASH(field)                                                   \
    if (g.field[0] != e.field) {                                              \
        if (error) { std::ostringstream o; o << #field " mismatch: got 0x"    \
            << std::hex << g.field[0] << ", expected 0x" << e.field;           \
            *error = o.str(); }                                                \
        return false; }

    MK_CHECK_INT(tensors_defined);
    MK_CHECK_INT(define_stall);
    MK_CHECK_INT(tasks_ready_register);
    MK_CHECK_INT(tasks_blocked_register);
    MK_CHECK_INT(cell_decrements);
    MK_CHECK_INT(cell_zero);
    MK_CHECK_INT(cell_already_zero);
    MK_CHECK_INT(tasks_ready_push);
    MK_CHECK_INT(tasks_popped);
    MK_CHECK_INT(tasks_started);
    MK_CHECK_INT(tasks_completed);
    MK_CHECK_INT(dynamic_cancelled);
    MK_CHECK_INT(cell_cancelled);
    MK_CHECK_INT(tasks_cancel_by_cell);
    MK_CHECK_INT(tasks_cancel_explicit);
    MK_CHECK_INT(ready_stale_drop);
    MK_CHECK_INT(popped_stale_drop);
    MK_CHECK_INT(invalid_count);
    MK_CHECK_HASH(mk_event_hash);
    MK_CHECK_HASH(cell_hash);
    MK_CHECK_HASH(task_hash);
    MK_CHECK_HASH(queue_hash);
    MK_CHECK_HASH(reverse_dep_hash);
#undef MK_CHECK_INT
#undef MK_CHECK_HASH
    return true;
}

#endif  // MK_EVENT_TENSOR_RUNTIME_ORACLE_HPP_
