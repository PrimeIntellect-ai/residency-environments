// file: buddy_allocator_cleaner_oracle.hpp
//
// Host-side canonical model for T42. This is the source of truth that BOTH
// device implementations (reference and naive) must match exactly. It is an
// INDEPENDENT third implementation (STL-backed, host code).

#ifndef BUDDY_ALLOCATOR_CLEANER_ORACLE_HPP_
#define BUDDY_ALLOCATOR_CLEANER_ORACLE_HPP_

#include "buddy_allocator_cleaner_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Event-kind enumerations (stable byte values, hashed into event streams).
// ---------------------------------------------------------------------------
enum BacAllocKind : uint8_t {
    BAC_EK_ALLOC_OK = 0,
    BAC_EK_ALLOC_OOM = 1,
    BAC_EK_RELOCATE_OBJECT = 2,
};

enum BacFinalizeKind : uint8_t {
    BAC_FK_FREE_DEFERRED = 0,
    BAC_FK_FREE_DEFERRED_DUP = 1,
    BAC_FK_OBJECT_FINALIZE = 2,
    BAC_FK_SEAL_IMPLICIT = 3,
    BAC_FK_SEAL_EXPLICIT = 4,
    BAC_FK_SEAL_EMPTY = 5,
    BAC_FK_CLEAN_BLOCKED_SEGMENT = 6,
    BAC_FK_SEGMENT_RECLAIM = 7,
};

// reason codes for finalize_hash
enum BacFinalizeReason : uint8_t {
    BAC_RSN_NONE = 0,
    BAC_RSN_FREE_IMMEDIATE = 1,
    BAC_RSN_UNPIN_FINALIZE = 2,
};

static const uint64_t BAC_U64_MAX = 0xFFFFFFFFFFFFFFFFULL;
static const uint8_t BAC_ORDER_NONE = 255;
static const uint64_t BAC_FNV_BASIS = 1469598103934665603ULL;
static const uint64_t BAC_FNV_PRIME = 1099511628211ULL;

// ---------------------------------------------------------------------------
// FNV-1a-64 helpers.
// ---------------------------------------------------------------------------
static inline uint64_t bac_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= BAC_FNV_PRIME;
    return h;
}
static inline void bac_oracle_fnv_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = bac_oracle_fnv_byte(v, b[i]);
    *h = v;
}
static inline void bac_oracle_fnv_u8(uint64_t* h, uint8_t v) { bac_oracle_fnv_bytes(h, &v, 1); }
static inline void bac_oracle_fnv_u32(uint64_t* h, uint32_t v) { bac_oracle_fnv_bytes(h, &v, 4); }
static inline void bac_oracle_fnv_u64(uint64_t* h, uint64_t v) { bac_oracle_fnv_bytes(h, &v, 8); }

// ---------------------------------------------------------------------------
// Expected output snapshot.
// ---------------------------------------------------------------------------
struct BacExpected {
    uint64_t alloc_ok = 0;
    uint64_t alloc_oom = 0;
    uint64_t free_finalized = 0;
    uint64_t free_deferred = 0;
    uint64_t pin_ok = 0;
    uint64_t unpin_ok = 0;
    uint64_t seal_explicit = 0;
    uint64_t seal_implicit = 0;
    uint64_t seal_empty = 0;
    uint64_t relocated_objects = 0;
    uint64_t clean_blocked_segments = 0;
    uint64_t segments_reclaimed = 0;
    uint64_t buddy_splits = 0;
    uint64_t buddy_merges = 0;
    uint64_t padding_pages_added = 0;
    uint64_t invalid_count = 0;

    uint64_t alloc_event_hash = BAC_FNV_BASIS;
    uint64_t finalize_hash = BAC_FNV_BASIS;

    uint64_t buddy_hash = BAC_FNV_BASIS;
    uint64_t segment_hash = BAC_FNV_BASIS;
    uint64_t object_hash = BAC_FNV_BASIS;

    uint64_t live_object_count = 0;
    uint64_t live_segment_count = 0;
};

struct BacHostOutputsView {
    const uint64_t* alloc_ok;
    const uint64_t* alloc_oom;
    const uint64_t* free_finalized;
    const uint64_t* free_deferred;
    const uint64_t* pin_ok;
    const uint64_t* unpin_ok;
    const uint64_t* seal_explicit;
    const uint64_t* seal_implicit;
    const uint64_t* seal_empty;
    const uint64_t* relocated_objects;
    const uint64_t* clean_blocked_segments;
    const uint64_t* segments_reclaimed;
    const uint64_t* buddy_splits;
    const uint64_t* buddy_merges;
    const uint64_t* padding_pages_added;
    const uint64_t* invalid_count;
    const uint64_t* alloc_event_hash;
    const uint64_t* finalize_hash;
    const uint64_t* buddy_hash;
    const uint64_t* segment_hash;
    const uint64_t* object_hash;
    const uint64_t* live_object_count;
    const uint64_t* live_segment_count;
};

// ---------------------------------------------------------------------------
// Internal tables.
// ---------------------------------------------------------------------------
struct BacOracleSegment {
    uint64_t segment_id;
    uint32_t class_id;
    uint64_t base_page;
    uint64_t append_offset_pages;
    uint64_t live_pages;
    uint64_t dead_pages;
    uint8_t sealed;
};

struct BacOracleObject {
    uint64_t obj_id;
    uint32_t class_id;
    uint64_t segment_id;
    uint64_t base_page;
    uint8_t order;
    uint64_t block_pages;
    uint64_t requested_pages;
    uint64_t pin_count;
    uint8_t free_pending;
    uint64_t alloc_seq;
    uint64_t move_seq;
};

struct BacOracleState {
    BacProblemSpec spec{};
    uint64_t total_pages = 0;
    uint64_t segment_pages = 0;
    int max_order = 0;
    int segment_order = 0;

    // free_lists[order] = sorted (ascending) list of base_page.
    std::vector<std::vector<uint64_t>> free_lists;

    uint64_t next_segment_id = 1;
    uint64_t event_seq = 0;
    uint32_t op_index = 0;

    // active_segment[class] : 0 => none, else segment_id.
    std::vector<uint64_t> active_segment;

    // segment_id -> segment
    std::map<uint64_t, BacOracleSegment> segments;
    // obj_id -> object
    std::map<uint64_t, BacOracleObject> objects;

    // accumulating counters / hashes (persist across steps)
    BacExpected acc;

    void init(const BacProblemSpec& s) {
        spec = s;
        max_order = s.max_order;
        segment_order = s.segment_order;
        total_pages = (uint64_t)1 << max_order;
        segment_pages = (uint64_t)1 << segment_order;
        reset();
    }

    void reset() {
        free_lists.assign((size_t)max_order + 1, std::vector<uint64_t>());
        free_lists[(size_t)max_order].push_back(0);  // one block at max_order base 0
        next_segment_id = 1;
        event_seq = 0;
        op_index = 0;
        active_segment.assign((size_t)spec.num_classes, 0);
        segments.clear();
        objects.clear();
        acc = BacExpected();
    }

    // ---- buddy helpers ----
    static int ceil_log2(uint64_t n) {
        int o = 0;
        uint64_t p = 1;
        while (p < n) { p <<= 1; ++o; }
        return o;
    }

    bool free_list_contains(int order, uint64_t base, size_t* idx_out) const {
        const std::vector<uint64_t>& v = free_lists[(size_t)order];
        // sorted ascending; binary search
        size_t lo = 0, hi = v.size();
        while (lo < hi) {
            size_t mid = (lo + hi) / 2;
            if (v[mid] < base) lo = mid + 1;
            else hi = mid;
        }
        if (lo < v.size() && v[lo] == base) { if (idx_out) *idx_out = lo; return true; }
        return false;
    }

    void free_list_insert(int order, uint64_t base) {
        std::vector<uint64_t>& v = free_lists[(size_t)order];
        size_t lo = 0, hi = v.size();
        while (lo < hi) {
            size_t mid = (lo + hi) / 2;
            if (v[mid] < base) lo = mid + 1;
            else hi = mid;
        }
        v.insert(v.begin() + (ptrdiff_t)lo, base);
    }

    void free_list_remove(int order, uint64_t base) {
        size_t idx = 0;
        if (free_list_contains(order, base, &idx)) {
            free_lists[(size_t)order].erase(free_lists[(size_t)order].begin() + (ptrdiff_t)idx);
        }
    }

    // Allocate one block at order want_order from buddy. Returns true on
    // success with base in *base_out. Emits BUDDY_SPLIT per split.
    bool buddy_alloc(int want_order, uint64_t* base_out) {
        int k = -1;
        for (int o = want_order; o <= max_order; ++o) {
            if (!free_lists[(size_t)o].empty()) { k = o; break; }
        }
        if (k < 0) return false;
        // lowest base at order k
        uint64_t base = free_lists[(size_t)k][0];
        free_lists[(size_t)k].erase(free_lists[(size_t)k].begin());
        while (k > want_order) {
            // split: left half (base) continues, right half inserted at k-1
            int child = k - 1;
            uint64_t right = base + ((uint64_t)1 << child);
            free_list_insert(child, right);
            acc.buddy_splits += 1;
            k = child;
        }
        *base_out = base;
        return true;
    }

    // Free a block (base, order) back to buddy with iterative merge.
    void buddy_free(uint64_t base, int order) {
        int o = order;
        uint64_t b = base;
        while (o < max_order) {
            uint64_t buddy = b ^ ((uint64_t)1 << o);
            size_t idx = 0;
            if (free_list_contains(o, buddy, &idx)) {
                free_lists[(size_t)o].erase(free_lists[(size_t)o].begin() + (ptrdiff_t)idx);
                b = (b < buddy) ? b : buddy;
                o += 1;
                acc.buddy_merges += 1;
            } else {
                break;
            }
        }
        free_list_insert(o, b);
    }

    // ---- event hashing ----
    void emit_alloc_event(uint8_t kind, uint64_t seq, uint32_t opidx,
                          uint64_t obj_id, uint32_t class_id,
                          uint64_t seg_id, uint64_t base_page, uint8_t order,
                          uint64_t requested_pages, uint64_t block_pages) {
        uint64_t* h = &acc.alloc_event_hash;
        bac_oracle_fnv_u8(h, kind);
        bac_oracle_fnv_u64(h, seq);
        bac_oracle_fnv_u32(h, opidx);
        bac_oracle_fnv_u64(h, obj_id);
        bac_oracle_fnv_u32(h, class_id);
        bac_oracle_fnv_u64(h, seg_id);
        bac_oracle_fnv_u64(h, base_page);
        bac_oracle_fnv_u8(h, order);
        bac_oracle_fnv_u64(h, requested_pages);
        bac_oracle_fnv_u64(h, block_pages);
    }

    void emit_finalize_event(uint8_t kind, uint64_t seq, uint32_t opidx,
                             uint64_t obj_id, uint64_t seg_id,
                             uint64_t base_page, uint64_t pages, uint8_t reason) {
        uint64_t* h = &acc.finalize_hash;
        bac_oracle_fnv_u8(h, kind);
        bac_oracle_fnv_u64(h, seq);
        bac_oracle_fnv_u32(h, opidx);
        bac_oracle_fnv_u64(h, obj_id);
        bac_oracle_fnv_u64(h, seg_id);
        bac_oracle_fnv_u64(h, base_page);
        bac_oracle_fnv_u64(h, pages);
        bac_oracle_fnv_u8(h, reason);
    }

    // Seal a segment (implicit or explicit). Adds tail to dead pages.
    void seal_segment(BacOracleSegment& seg, bool implicit, uint64_t seq, uint32_t opidx) {
        uint64_t tail = segment_pages - seg.append_offset_pages;
        seg.dead_pages += tail;
        seg.append_offset_pages = segment_pages;
        seg.sealed = 1;
        if (implicit) {
            acc.seal_implicit += 1;
            emit_finalize_event(BAC_FK_SEAL_IMPLICIT, seq, opidx, BAC_U64_MAX,
                                seg.segment_id, seg.base_page, tail, BAC_RSN_NONE);
        } else {
            acc.seal_explicit += 1;
            emit_finalize_event(BAC_FK_SEAL_EXPLICIT, seq, opidx, BAC_U64_MAX,
                                seg.segment_id, seg.base_page, tail, BAC_RSN_NONE);
        }
    }

    // Append-allocate a block for (class_id, order, block_pages). This is the
    // shared placement engine for both user ALLOC and relocation. On success
    // sets *seg_id_out, *base_out and returns true. On OOM returns false.
    // It mutates segment state, may seal implicitly + buddy split, and updates
    // padding counters. It does NOT create the object record (caller does).
    bool append_allocate(uint32_t class_id, int order, uint64_t block_pages,
                         uint64_t seq, uint32_t opidx,
                         uint64_t* seg_id_out, uint64_t* base_out) {
        uint64_t active_id = active_segment[class_id];
        bool need_new = false;
        uint64_t aligned_offset = 0;

        if (active_id == 0) {
            need_new = true;
        } else {
            BacOracleSegment& seg = segments[active_id];
            uint64_t off = seg.append_offset_pages;
            aligned_offset = (off + block_pages - 1) & ~(block_pages - 1);
            if (aligned_offset + block_pages > segment_pages) {
                need_new = true;
            }
        }

        if (need_new) {
            // Seal the existing active segment first (if any).
            if (active_id != 0) {
                seal_segment(segments[active_id], /*implicit=*/true, seq, opidx);
                active_segment[class_id] = 0;
                active_id = 0;
            }
            // Capacity check for the segment table.
            if (segments.size() >= (size_t)spec.max_segments) {
                return false;  // OOM (no room for new segment record)
            }
            // Allocate a new segment from buddy at segment_order.
            uint64_t new_base = 0;
            if (!buddy_alloc(segment_order, &new_base)) {
                return false;  // OOM (buddy exhausted); prior seal persists.
            }
            BacOracleSegment seg;
            seg.segment_id = next_segment_id++;
            seg.class_id = class_id;
            seg.base_page = new_base;
            seg.append_offset_pages = 0;
            seg.live_pages = 0;
            seg.dead_pages = 0;
            seg.sealed = 0;
            segments[seg.segment_id] = seg;
            active_segment[class_id] = seg.segment_id;
            active_id = seg.segment_id;
            aligned_offset = 0;  // fresh segment, offset 0 already aligned
        }

        BacOracleSegment& seg = segments[active_id];
        uint64_t padding = aligned_offset - seg.append_offset_pages;
        uint64_t base_page = seg.base_page + aligned_offset;
        seg.append_offset_pages = aligned_offset + block_pages;
        seg.live_pages += block_pages;
        seg.dead_pages += padding;
        acc.padding_pages_added += padding;

        *seg_id_out = seg.segment_id;
        *base_out = base_page;
        return true;
    }

    // Finalize an object physically: remove from table, update segment counts,
    // emit OBJECT_FINALIZE.
    void finalize_object(BacOracleObject& obj, uint64_t seq, uint32_t opidx, uint8_t reason) {
        BacOracleSegment& seg = segments[obj.segment_id];
        seg.live_pages -= obj.block_pages;
        seg.dead_pages += obj.block_pages;
        acc.free_finalized += 1;
        emit_finalize_event(BAC_FK_OBJECT_FINALIZE, seq, opidx, obj.obj_id,
                            obj.segment_id, obj.base_page, obj.block_pages, reason);
        objects.erase(obj.obj_id);
    }

    // Reclaim an empty victim segment: remove, buddy-free, emit SEGMENT_RECLAIM.
    void reclaim_segment(uint64_t seg_id, uint64_t seq, uint32_t opidx) {
        BacOracleSegment seg = segments[seg_id];  // copy
        // If it was active for its class, clear active (it should be sealed,
        // but be defensive — only sealed segments are victims).
        if (active_segment[seg.class_id] == seg_id) active_segment[seg.class_id] = 0;
        segments.erase(seg_id);
        buddy_free(seg.base_page, segment_order);
        acc.segments_reclaimed += 1;
        emit_finalize_event(BAC_FK_SEGMENT_RECLAIM, seq, opidx, BAC_U64_MAX,
                            seg_id, seg.base_page, segment_pages, BAC_RSN_NONE);
    }

    // ---- per-op handlers ----
    void op_alloc(const BacOp& op, uint64_t seq, uint32_t opidx) {
        uint32_t class_id = (uint32_t)op.class_id;
        uint64_t requested = op.a;
        // validity
        if ((int)class_id >= spec.num_classes || requested == 0 ||
            requested > segment_pages || objects.count(op.obj_id) != 0) {
            acc.invalid_count += 1;
            return;
        }
        if (objects.size() >= (size_t)spec.max_objects) {
            acc.alloc_oom += 1;
            emit_alloc_event(BAC_EK_ALLOC_OOM, seq, opidx, op.obj_id, class_id,
                             BAC_U64_MAX, BAC_U64_MAX, BAC_ORDER_NONE, requested, 0);
            return;
        }
        int order = ceil_log2(requested);
        uint64_t block_pages = (uint64_t)1 << order;

        uint64_t seg_id = 0, base_page = 0;
        if (!append_allocate(class_id, order, block_pages, seq, opidx, &seg_id, &base_page)) {
            acc.alloc_oom += 1;
            emit_alloc_event(BAC_EK_ALLOC_OOM, seq, opidx, op.obj_id, class_id,
                             BAC_U64_MAX, BAC_U64_MAX, BAC_ORDER_NONE, requested, block_pages);
            return;
        }
        BacOracleObject obj;
        obj.obj_id = op.obj_id;
        obj.class_id = class_id;
        obj.segment_id = seg_id;
        obj.base_page = base_page;
        obj.order = (uint8_t)order;
        obj.block_pages = block_pages;
        obj.requested_pages = requested;
        obj.pin_count = 0;
        obj.free_pending = 0;
        obj.alloc_seq = seq;
        obj.move_seq = seq;
        objects[obj.obj_id] = obj;

        acc.alloc_ok += 1;
        emit_alloc_event(BAC_EK_ALLOC_OK, seq, opidx, op.obj_id, class_id,
                         seg_id, base_page, (uint8_t)order, requested, block_pages);
    }

    void op_free(const BacOp& op, uint64_t seq, uint32_t opidx) {
        auto it = objects.find(op.obj_id);
        if (it == objects.end()) { acc.invalid_count += 1; return; }
        BacOracleObject& obj = it->second;
        if (obj.pin_count > 0) {
            if (obj.free_pending == 1) {
                acc.free_deferred += 1;  // counter counts every FREE_DEFERRED event (incl dup)
                emit_finalize_event(BAC_FK_FREE_DEFERRED_DUP, seq, opidx, obj.obj_id,
                                    obj.segment_id, obj.base_page, obj.block_pages, BAC_RSN_NONE);
                return;
            }
            obj.free_pending = 1;
            acc.free_deferred += 1;
            emit_finalize_event(BAC_FK_FREE_DEFERRED, seq, opidx, obj.obj_id,
                                obj.segment_id, obj.base_page, obj.block_pages, BAC_RSN_NONE);
            return;
        }
        // pin_count == 0: finalize immediately
        finalize_object(obj, seq, opidx, BAC_RSN_FREE_IMMEDIATE);
    }

    void op_pin(const BacOp& op, uint64_t seq, uint32_t opidx) {
        auto it = objects.find(op.obj_id);
        if (it == objects.end()) { acc.invalid_count += 1; return; }
        BacOracleObject& obj = it->second;
        if (obj.free_pending == 1) { acc.invalid_count += 1; return; }
        // increment mod 2^64 unless it would wrap to zero
        if (obj.pin_count == BAC_U64_MAX) { acc.invalid_count += 1; return; }
        obj.pin_count += 1;
        acc.pin_ok += 1;
    }

    void op_unpin(const BacOp& op, uint64_t seq, uint32_t opidx) {
        auto it = objects.find(op.obj_id);
        if (it == objects.end()) { acc.invalid_count += 1; return; }
        BacOracleObject& obj = it->second;
        if (obj.pin_count == 0) { acc.invalid_count += 1; return; }
        obj.pin_count -= 1;
        acc.unpin_ok += 1;
        if (obj.pin_count == 0 && obj.free_pending == 1) {
            finalize_object(obj, seq, opidx, BAC_RSN_UNPIN_FINALIZE);
        }
    }

    void op_seal(const BacOp& op, uint64_t seq, uint32_t opidx) {
        uint32_t class_id = (uint32_t)op.class_id;
        if ((int)class_id >= spec.num_classes) { acc.invalid_count += 1; return; }
        uint64_t active_id = active_segment[class_id];
        if (active_id == 0) {
            acc.seal_empty += 1;
            emit_finalize_event(BAC_FK_SEAL_EMPTY, seq, opidx, BAC_U64_MAX,
                                BAC_U64_MAX, 0, 0, BAC_RSN_NONE);
            return;
        }
        seal_segment(segments[active_id], /*implicit=*/false, seq, opidx);
        active_segment[class_id] = 0;
    }

    // Victim selection: among sealed segments with dead_pages > 0:
    //   largest dead_pages; tie smallest live_pages; tie smallest base_page;
    //   tie smallest segment_id. Returns 0 if none.
    uint64_t select_victim() const {
        uint64_t best_id = 0;
        const BacOracleSegment* best = nullptr;
        for (const auto& kv : segments) {
            const BacOracleSegment& s = kv.second;
            if (s.sealed == 0 || s.dead_pages == 0) continue;
            bool better;
            if (best == nullptr) {
                better = true;
            } else if (s.dead_pages != best->dead_pages) {
                better = s.dead_pages > best->dead_pages;
            } else if (s.live_pages != best->live_pages) {
                better = s.live_pages < best->live_pages;
            } else if (s.base_page != best->base_page) {
                better = s.base_page < best->base_page;
            } else {
                better = s.segment_id < best->segment_id;
            }
            if (better) { best = &s; best_id = s.segment_id; }
        }
        return best_id;
    }

    void op_clean(const BacOp& op, uint64_t seq, uint32_t opidx) {
        uint64_t max_segs = op.a;
        uint64_t budget = op.b;
        if (max_segs == 0 || budget == 0) return;  // valid no-op

        while (max_segs > 0 && budget > 0) {
            uint64_t vic_id = select_victim();
            if (vic_id == 0) break;

            // If victim already has no live pages at selection time, reclaim now.
            if (segments[vic_id].live_pages == 0) {
                reclaim_segment(vic_id, seq, opidx);
                max_segs -= 1;
                continue;
            }

            // Gather movable objects in this segment sorted by (base_page, obj_id).
            std::vector<uint64_t> movable;  // obj_ids
            for (const auto& kv : objects) {
                const BacOracleObject& o = kv.second;
                if (o.segment_id != vic_id) continue;
                if (o.pin_count != 0 || o.free_pending != 0) continue;
                movable.push_back(o.obj_id);
            }
            std::sort(movable.begin(), movable.end(),
                      [this](uint64_t a, uint64_t b) {
                          const BacOracleObject& oa = objects.at(a);
                          const BacOracleObject& ob = objects.at(b);
                          if (oa.base_page != ob.base_page) return oa.base_page < ob.base_page;
                          return oa.obj_id < ob.obj_id;
                      });

            if (movable.empty()) {
                // live_pages > 0 here (already handled ==0 above) and no movable.
                acc.clean_blocked_segments += 1;
                emit_finalize_event(BAC_FK_CLEAN_BLOCKED_SEGMENT, seq, opidx, BAC_U64_MAX,
                                    vic_id, segments[vic_id].base_page,
                                    segments[vic_id].live_pages, BAC_RSN_NONE);
                max_segs -= 1;
                continue;
            }

            bool stop_clean = false;
            for (uint64_t oid : movable) {
                // The object may have been finalized? No — relocation doesn't
                // finalize. It still exists. But the victim segment record may
                // be gone only after reclaim, which ends the loop.
                BacOracleObject obj = objects.at(oid);  // copy of current fields

                if (obj.block_pages > budget) { stop_clean = true; break; }

                int order = (int)obj.order;
                uint64_t block_pages = obj.block_pages;
                uint64_t dst_seg = 0, dst_base = 0;
                if (!append_allocate(obj.class_id, order, block_pages, seq, opidx,
                                     &dst_seg, &dst_base)) {
                    stop_clean = true;  // OOM
                    break;
                }
                // old segment loses live, gains dead
                BacOracleSegment& old_seg = segments[vic_id];
                old_seg.live_pages -= block_pages;
                old_seg.dead_pages += block_pages;
                // update object
                BacOracleObject& real = objects.at(oid);
                real.segment_id = dst_seg;
                real.base_page = dst_base;
                real.move_seq = seq;

                acc.relocated_objects += 1;
                emit_alloc_event(BAC_EK_RELOCATE_OBJECT, seq, opidx, oid, real.class_id,
                                 dst_seg, dst_base, (uint8_t)order, real.requested_pages,
                                 block_pages);
                budget -= block_pages;

                if (segments[vic_id].live_pages == 0) {
                    reclaim_segment(vic_id, seq, opidx);
                    max_segs -= 1;
                    break;
                }
            }

            if (stop_clean) break;
            // Loop back and RE-SELECT a victim. If this victim still has live
            // pages (pinned/pending residue) it will be re-selected on a later
            // pass and reported as CLEAN_BLOCKED_SEGMENT (it now has no movable
            // objects). We intentionally do NOT shortcut that here so that
            // selection order across segments stays faithful to the contract.
        }
    }

    void apply_op(const BacOp& op) {
        event_seq += 1;  // wraps mod 2^64
        uint64_t seq = event_seq;
        uint32_t opidx = op_index;
        op_index += 1;  // wraps mod 2^32
        switch (op.op_type) {
            case BAC_OP_ALLOC: op_alloc(op, seq, opidx); break;
            case BAC_OP_FREE:  op_free(op, seq, opidx); break;
            case BAC_OP_PIN:   op_pin(op, seq, opidx); break;
            case BAC_OP_UNPIN: op_unpin(op, seq, opidx); break;
            case BAC_OP_SEAL:  op_seal(op, seq, opidx); break;
            case BAC_OP_CLEAN: op_clean(op, seq, opidx); break;
            default: acc.invalid_count += 1; break;
        }
    }

    // ---- structural hashes ----
    uint64_t compute_buddy_hash() const {
        uint64_t h = BAC_FNV_BASIS;
        for (int order = 0; order <= max_order; ++order) {
            const std::vector<uint64_t>& v = free_lists[(size_t)order];
            for (uint64_t base : v) {  // ascending base (list is sorted)
                bac_oracle_fnv_u8(&h, (uint8_t)order);
                bac_oracle_fnv_u64(&h, base);
            }
        }
        return h;
    }

    uint64_t compute_segment_hash() const {
        // order by base_page, then segment_id
        std::vector<const BacOracleSegment*> v;
        v.reserve(segments.size());
        for (const auto& kv : segments) v.push_back(&kv.second);
        std::sort(v.begin(), v.end(), [](const BacOracleSegment* a, const BacOracleSegment* b) {
            if (a->base_page != b->base_page) return a->base_page < b->base_page;
            return a->segment_id < b->segment_id;
        });
        uint64_t h = BAC_FNV_BASIS;
        for (const BacOracleSegment* s : v) {
            uint8_t is_active = (active_segment[s->class_id] == s->segment_id) ? 1 : 0;
            bac_oracle_fnv_u64(&h, s->segment_id);
            bac_oracle_fnv_u32(&h, s->class_id);
            bac_oracle_fnv_u64(&h, s->base_page);
            bac_oracle_fnv_u64(&h, s->append_offset_pages);
            bac_oracle_fnv_u64(&h, s->live_pages);
            bac_oracle_fnv_u64(&h, s->dead_pages);
            bac_oracle_fnv_u8(&h, s->sealed);
            bac_oracle_fnv_u8(&h, is_active);
        }
        return h;
    }

    uint64_t compute_object_hash() const {
        uint64_t h = BAC_FNV_BASIS;
        // std::map iterates obj_id ascending
        for (const auto& kv : objects) {
            const BacOracleObject& o = kv.second;
            bac_oracle_fnv_u64(&h, o.obj_id);
            bac_oracle_fnv_u32(&h, o.class_id);
            bac_oracle_fnv_u64(&h, o.segment_id);
            bac_oracle_fnv_u64(&h, o.base_page);
            bac_oracle_fnv_u8(&h, o.order);
            bac_oracle_fnv_u64(&h, o.requested_pages);
            bac_oracle_fnv_u64(&h, o.block_pages);
            bac_oracle_fnv_u64(&h, o.pin_count);
            bac_oracle_fnv_u8(&h, o.free_pending);
            bac_oracle_fnv_u64(&h, o.alloc_seq);
            bac_oracle_fnv_u64(&h, o.move_seq);
        }
        return h;
    }

    void step_once(const BacRunSpec& run, const BacOp* ops, BacExpected* expected) {
        for (int i = 0; i < run.num_ops; ++i) {
            apply_op(ops[i]);
        }
        *expected = acc;
        expected->buddy_hash = compute_buddy_hash();
        expected->segment_hash = compute_segment_hash();
        expected->object_hash = compute_object_hash();
        expected->live_object_count = (uint64_t)objects.size();
        expected->live_segment_count = (uint64_t)segments.size();
    }
};

// ---------------------------------------------------------------------------
// Output comparison.
// ---------------------------------------------------------------------------
static inline bool bac_check_all_outputs(
    const BacExpected& e,
    const BacHostOutputsView& g,
    std::string* err) {
#define BAC_CHECK(field) \
    if (g.field[0] != e.field) { \
        if (err) { std::ostringstream o; o << #field " mismatch: got " \
            << g.field[0] << " expected " << e.field; *err = o.str(); } \
        return false; }
#define BAC_CHECK_HEX(field) \
    if (g.field[0] != e.field) { \
        if (err) { std::ostringstream o; o << #field " mismatch: got 0x" \
            << std::hex << g.field[0] << " expected 0x" << e.field; *err = o.str(); } \
        return false; }

    BAC_CHECK(alloc_ok);
    BAC_CHECK(alloc_oom);
    BAC_CHECK(free_finalized);
    BAC_CHECK(free_deferred);
    BAC_CHECK(pin_ok);
    BAC_CHECK(unpin_ok);
    BAC_CHECK(seal_explicit);
    BAC_CHECK(seal_implicit);
    BAC_CHECK(seal_empty);
    BAC_CHECK(relocated_objects);
    BAC_CHECK(clean_blocked_segments);
    BAC_CHECK(segments_reclaimed);
    BAC_CHECK(buddy_splits);
    BAC_CHECK(buddy_merges);
    BAC_CHECK(padding_pages_added);
    BAC_CHECK(invalid_count);
    BAC_CHECK_HEX(alloc_event_hash);
    BAC_CHECK_HEX(finalize_hash);
    BAC_CHECK_HEX(buddy_hash);
    BAC_CHECK_HEX(segment_hash);
    BAC_CHECK_HEX(object_hash);
    BAC_CHECK(live_object_count);
    BAC_CHECK(live_segment_count);
#undef BAC_CHECK
#undef BAC_CHECK_HEX
    return true;
}

/*
GRADER MODEL

  oracle.init(problem_spec)
  solution_reset + oracle.reset()
  for each step:
    solution_run(...)
    oracle.step_once(...)
    bac_check_all_outputs(...)

Required harness coverage:
  - basic alloc fill + implicit seal + buddy split
  - pin / unpin / pin-delayed free / finalize ordering
  - clean with full reclaim (merge chains)
  - clean blocked by pins (CLEAN_BLOCKED_SEGMENT)
  - clean partial relocation with budget exhaustion
  - alloc OOM (object table full) and OOM (buddy exhausted) with prior implicit seal persisting
  - num_ops = 0 step
  - reset and replay
*/

#endif  // BUDDY_ALLOCATOR_CLEANER_ORACLE_HPP_
