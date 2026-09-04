// file: mk_instr_decoder_hazard_oracle.hpp
//
// Host golden model for MK10.  Implements the exact coupled semantics of the
// megakernel instruction decoder + tile hazard scoreboard and produces all
// FNV-1a-64 checksums after every coarse op.  STL-based, deliberately written
// with a different data layout than either device implementation.

#ifndef MK_INSTR_DECODER_HAZARD_ORACLE_HPP_
#define MK_INSTR_DECODER_HAZARD_ORACLE_HPP_

#include "mk_instr_decoder_hazard_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <deque>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ------------------------------------------------------------------ FNV-1a-64
struct MkFnv {
    uint64_t h = 1469598103934665603ULL;
    void u8(uint8_t v)  { h ^= (uint64_t)v; h *= 1099511628211ULL; }
    void u32(uint32_t v){ const uint8_t* p=(const uint8_t*)&v; for(int i=0;i<4;++i) u8(p[i]); }
    void u64(uint64_t v){ const uint8_t* p=(const uint8_t*)&v; for(int i=0;i<8;++i) u8(p[i]); }
};

struct MkExpected {
    std::vector<int64_t> counts;
    int32_t op_index = 0;
    uint64_t clock = 0, event_seq = 0;
    uint64_t decoder_event_hash = 0, sm_stream_hash = 0, decoded_hash = 0;
    uint64_t tile_scoreboard_hash = 0, page_hash = 0, pending_unit_hash = 0;
    uint64_t counter_hash = 0, state_checksum = 0;
};

struct MkHostOutputsView {
    const int64_t* counts;
    const int32_t* op_index_out;
    const uint64_t* clock_out;
    const uint64_t* event_seq_out;
    const uint64_t* decoder_event_hash;
    const uint64_t* sm_stream_hash;
    const uint64_t* decoded_hash;
    const uint64_t* tile_scoreboard_hash;
    const uint64_t* page_hash;
    const uint64_t* pending_unit_hash;
    const uint64_t* counter_hash;
    const uint64_t* state_checksum;
};

struct MkOracle {
    static const uint32_t U32MAX = 0xFFFFFFFFu;

    MkProblemSpec spec{};
    const MkInstr* stream = nullptr;   // [sm_count * stream_len]

    // scalars
    uint64_t clock = 0, event_seq = 0;
    uint64_t decode_seq_next = 1, issue_seq_next = 1, unit_seq_next = 1;
    uint64_t retire_seq_next = 1, epoch_seq_next = 1;
    int32_t op_index = 0;
    uint64_t event_hash = 1469598103934665603ULL;

    struct SmState {
        uint32_t pc = 0;
        uint64_t epoch = 0;
        uint8_t halted = 0;
        std::deque<uint64_t> decode_q;   // decode_seqs
        std::deque<uint64_t> issue_q;
        std::deque<uint64_t> replay_q;
    };
    std::vector<SmState> sms;

    struct Rec {
        int32_t sm = 0;
        uint32_t pc = 0;
        uint64_t epoch = 0;
        uint64_t decode_seq = 0;
        uint64_t instr_uid = 0;
        int32_t opcode = 0;
        int32_t status = MK_ST_DECODED;
        int32_t hazard_code = MK_HZ_NONE;
        uint64_t result_hash = 0;
        std::vector<int32_t> read_tiles;
        std::vector<int32_t> write_tiles;
        std::vector<int32_t> pages;      // assigned pages (page id within sm)
        // carried instr fields needed at completion:
        int32_t free_tile = -1;
        int32_t counter_id = -1;
        uint64_t target = 0, amount = 0, compute_seed = 0;
        uint64_t page_key = 0;
        int32_t dst_tile = -1;
        int32_t taken_pc = 0, fallthrough_pc = 0;
        uint64_t latency = 0;
    };
    // keyed by decode_seq
    std::map<uint64_t, Rec> recs;

    struct Tile {
        uint64_t version = 0;
        uint64_t writer = 0;            // writer_decode_seq or 0
        uint64_t oldest_reader = 0;     // or 0
        uint64_t reader_count = 0;
        uint32_t resident_page = U32MAX;
        uint8_t dirty = 0;
        uint8_t freed = 0;
    };
    std::vector<Tile> tiles;

    struct Page {
        int32_t state = MK_PG_FREE;
        uint64_t page_key = 0;
        uint32_t tile = U32MAX;
        uint64_t version = 0;
        uint64_t owner = 0;
    };
    // pages[sm * page_count + p]
    std::vector<Page> pages;

    std::vector<uint64_t> counters;

    struct Unit {
        int32_t kind = 0;
        int32_t sm = 0;
        uint64_t decode_seq = 0;
        uint64_t epoch = 0;
        uint64_t due_clock = 0;
        uint64_t unit_seq = 0;
    };
    std::vector<Unit> pending;

    std::vector<int64_t> counts;

    int SM=0, SL=0, DW=0, IW=0, RW=0, TC=0, PG=0, CC=0;

    void init(const MkProblemSpec& s, const MkInstr* str) {
        spec = s; stream = str;
        SM = s.sm_count; SL = s.stream_len_per_sm; DW = s.decode_width;
        IW = s.issue_width; RW = s.retire_width; TC = s.tile_count;
        PG = s.page_count_per_sm; CC = s.counter_count;
        counts.assign(MK_COUNT_N, 0);
        sms.assign(SM, SmState{});
        tiles.assign(TC, Tile{});
        pages.assign((size_t)SM * PG, Page{});
        counters.assign(CC, 0);
        reset();
    }

    void reset() {
        clock = 0; event_seq = 0; op_index = 0;
        decode_seq_next = issue_seq_next = unit_seq_next = retire_seq_next = epoch_seq_next = 1;
        event_hash = 1469598103934665603ULL;
        for (auto& c : counts) c = 0;
        for (auto& sm : sms) { sm.pc=0; sm.epoch=0; sm.halted=0; sm.decode_q.clear(); sm.issue_q.clear(); sm.replay_q.clear(); }
        recs.clear();
        for (auto& t : tiles) t = Tile{};
        for (auto& p : pages) p = Page{};
        for (auto& c : counters) c = 0;
        pending.clear();
    }

    const MkInstr& instr_at(int sm, uint32_t pc) const { return stream[(size_t)sm * SL + pc]; }
    Page& page(int sm, int p) { return pages[(size_t)sm * PG + p]; }
    Rec* find(uint64_t ds) { auto it = recs.find(ds); return it==recs.end()?nullptr:&it->second; }

    // ---------------------------------------------------------------- emit
    void emit(uint8_t kind, uint32_t sm, uint32_t pc, uint64_t decode_seq,
              uint64_t instr_uid, uint32_t tile, uint64_t aux) {
        MkFnv f; f.h = event_hash;
        f.u8(kind);
        f.u64(event_seq);
        f.u32((uint32_t)op_index);
        f.u64(clock);
        f.u32(sm);
        f.u32(pc);
        f.u64(decode_seq);
        f.u64(instr_uid);
        f.u32(tile);
        f.u64(aux);
        event_hash = f.h;
        event_seq += 1;
    }

    void invalid() {
        counts[MK_C_INVALID] += 1;
        emit(MK_EV_INVALID, U32MAX, U32MAX, 0, 0, U32MAX, 0);
    }

    static uint64_t fnv_epoch(uint64_t seed, int sm, uint64_t ctr) {
        MkFnv f; f.u64(seed); f.u32((uint32_t)sm); f.u64(ctr); return f.h;
    }

    // ---------------------------------------------------------------- scoreboard helpers
    // recompute oldest unretired reader of a tile, scanning records on a given sm.
    void recompute_oldest_reader(int tile, int sm) {
        uint64_t best = 0;
        for (auto& kv : recs) {
            Rec& r = kv.second;
            if (r.sm != sm) continue;
            if (r.status==MK_ST_RETIRED || r.status==MK_ST_CANCELLED) continue;
            // a reader has reserved only if issued (ISSUED/COMPLETED/REPLAY)
            if (!(r.status==MK_ST_ISSUED || r.status==MK_ST_COMPLETED || r.status==MK_ST_REPLAY)) continue;
            bool reads=false; for (int t : r.read_tiles) if (t==tile) { reads=true; break; }
            if (!reads) continue;
            if (best==0 || r.decode_seq < best) best = r.decode_seq;
        }
        tiles[tile].oldest_reader = best;
    }

    // remove a decode_seq from all three queues of its sm.
    void remove_from_queues(int sm, uint64_t ds) {
        auto rm = [&](std::deque<uint64_t>& q){ for(auto it=q.begin();it!=q.end();++it) if(*it==ds){q.erase(it);return;} };
        rm(sms[sm].decode_q); rm(sms[sm].issue_q); rm(sms[sm].replay_q);
    }

    // release all pages owned by a record.
    void release_record_pages(Rec& r) {
        for (int pid : r.pages) {
            Page& pg = page(r.sm, pid);
            // if a tile is resident here, detach
            pg.state = MK_PG_FREE; pg.page_key=0; pg.tile=U32MAX; pg.version=0; pg.owner=0;
        }
        r.pages.clear();
    }

    // unwind scoreboard reservations of an ISSUED record being cancelled.
    void unwind_reservations(Rec& r) {
        for (int t : r.write_tiles) {
            if (tiles[t].writer == r.decode_seq) tiles[t].writer = 0;
        }
        for (int t : r.read_tiles) {
            if (tiles[t].reader_count > 0) tiles[t].reader_count -= 1;
        }
        // recompute oldest readers for read tiles after this record stops counting.
        // (mark CANCELLED first by caller before calling recompute) -> caller sets status.
    }

    // cancel records on sm with epoch==ep and decode_seq>after, in descending order.
    void cancel_younger(int sm, uint64_t ep, uint64_t after) {
        std::vector<uint64_t> victims;
        for (auto& kv : recs) {
            Rec& r = kv.second;
            if (r.sm!=sm) continue;
            if (r.epoch!=ep) continue;
            if (r.decode_seq<=after) continue;
            if (r.status==MK_ST_RETIRED || r.status==MK_ST_CANCELLED) continue;
            victims.push_back(r.decode_seq);
        }
        std::sort(victims.begin(), victims.end(), std::greater<uint64_t>());
        for (uint64_t ds : victims) {
            Rec& r = recs[ds];
            bool was_issued = (r.status==MK_ST_ISSUED || r.status==MK_ST_COMPLETED || r.status==MK_ST_REPLAY);
            release_record_pages(r);
            if (was_issued) unwind_reservations(r);
            r.status = MK_ST_CANCELLED;
            remove_from_queues(sm, ds);
            counts[MK_C_EPOCH_CANCEL] += 1;
            emit(MK_EV_EPOCH_CANCEL, sm, r.pc, ds, r.instr_uid, U32MAX, 0);
        }
        // after cancelling issued readers, recompute oldest readers for affected tiles.
        for (uint64_t ds : victims) {
            Rec& r = recs[ds];
            for (int t : r.read_tiles) recompute_oldest_reader(t, sm);
        }
    }

    // ---------------------------------------------------------------- ops
    bool any_outstanding() const {
        for (auto& kv : recs) {
            const Rec& r = kv.second;
            if (r.status!=MK_ST_RETIRED && r.status!=MK_ST_CANCELLED) return true;
        }
        if (!pending.empty()) return true;
        return false;
    }

    void op_begin(uint64_t seed) {
        if (any_outstanding()) { invalid(); return; }
        for (int s=0;s<SM;++s) {
            sms[s].pc = 0;
            sms[s].epoch = fnv_epoch(seed, s, epoch_seq_next);
            epoch_seq_next += 1;
            sms[s].halted = 0;
            sms[s].decode_q.clear(); sms[s].issue_q.clear(); sms[s].replay_q.clear();
        }
        for (auto& c : counters) c = 0;
        for (auto& t : tiles) t = Tile{};
        for (auto& p : pages) p = Page{};
        recs.clear();
        counts[MK_C_KERNEL_BEGIN] += 1;
        emit(MK_EV_KERNEL_BEGIN, U32MAX, U32MAX, 0, 0, U32MAX, seed);
    }

    void make_rec_fields(Rec& r, const MkInstr& in) {
        r.opcode = in.opcode;
        r.instr_uid = in.instr_uid;
        r.latency = in.latency;
        switch (in.opcode) {
            case MK_LD:
                r.write_tiles.push_back(in.dst_tile);
                r.dst_tile = in.dst_tile; r.page_key = in.page_key;
                break;
            case MK_ALU:
                for (int i=0;i<in.n_read;++i) r.read_tiles.push_back(in.read_tiles[i]);
                for (int i=0;i<in.n_write;++i) r.write_tiles.push_back(in.write_tiles[i]);
                r.compute_seed = in.compute_seed;
                break;
            case MK_ST:
                r.read_tiles.push_back(in.st_read_tile);
                break;
            case MK_WAIT:
                r.counter_id = in.counter_id; r.target = in.target;
                break;
            case MK_INC:
                r.counter_id = in.counter_id; r.amount = in.amount;
                break;
            case MK_FREE:
                r.free_tile = in.free_tile;
                break;
            case MK_BRANCH:
                r.counter_id = in.counter_id; r.target = in.target;
                r.taken_pc = in.taken_pc; r.fallthrough_pc = in.fallthrough_pc;
                break;
            case MK_NOP: default: break;
        }
    }

    void op_decode(int sm, int limit) {
        if (sm<0 || sm>=SM || limit==0) { invalid(); return; }
        int budget = std::min(limit, DW);
        for (int i=0;i<budget;++i) {
            SmState& S = sms[sm];
            if (S.halted) { counts[MK_C_DECODE_HALTED]+=1; emit(MK_EV_DECODE_HALTED, sm, S.pc, 0,0,U32MAX,0); break; }
            if ((int)S.decode_q.size() >= spec.max_decode_queue) {
                counts[MK_C_DECODE_QUEUE_FULL]+=1; emit(MK_EV_DECODE_QUEUE_FULL, sm, S.pc, 0,0,U32MAX,0); break;
            }
            if (S.pc >= (uint32_t)SL) { S.halted=1; counts[MK_C_DECODE_HALTED]+=1; emit(MK_EV_DECODE_HALTED, sm, S.pc, 0,0,U32MAX,0); break; }
            const MkInstr& in = instr_at(sm, S.pc);
            if ((in.predicate_mask & S.epoch) == 0) {
                counts[MK_C_DECODE_PRED_SKIP]+=1; emit(MK_EV_DECODE_PRED_SKIP, sm, S.pc, 0, in.instr_uid, U32MAX, 0);
                S.pc += 1; continue;
            }
            if ((int)recs.size() >= spec.max_decode_records) {
                // pool exhausted: treat like queue full (deterministic backpressure).
                counts[MK_C_DECODE_QUEUE_FULL]+=1; emit(MK_EV_DECODE_QUEUE_FULL, sm, S.pc, 0,0,U32MAX,0); break;
            }
            Rec r;
            r.sm = sm; r.pc = S.pc; r.epoch = S.epoch;
            r.decode_seq = decode_seq_next++;
            r.status = MK_ST_DECODED; r.hazard_code = MK_HZ_NONE; r.result_hash = 0;
            make_rec_fields(r, in);
            recs[r.decode_seq] = r;
            S.decode_q.push_back(r.decode_seq);
            S.issue_q.push_back(r.decode_seq);
            counts[MK_C_INSTR_DECODED]+=1;
            emit(MK_EV_INSTR_DECODE, sm, r.pc, r.decode_seq, r.instr_uid, U32MAX, 0);
            if (in.opcode == MK_BRANCH) S.pc = (uint32_t)in.fallthrough_pc;
            else S.pc += 1;
        }
    }

    int count_free_pages(int sm) {
        int n=0; for (int p=0;p<PG;++p) if (page(sm,p).state==MK_PG_FREE) ++n; return n;
    }
    int lowest_free_page(int sm) {
        for (int p=0;p<PG;++p) if (page(sm,p).state==MK_PG_FREE) return p; return -1;
    }

    // returns hazard code (0 none) for raw/war/waw.
    int hazard_check(Rec& r) {
        // RAW
        for (int t : r.read_tiles) {
            if (tiles[t].freed) return MK_HZ_RAW;
            uint64_t w = tiles[t].writer;
            if (w!=0 && w < r.decode_seq) return MK_HZ_RAW;
        }
        // WAR
        for (int t : r.write_tiles) {
            uint64_t orr = tiles[t].oldest_reader;
            if (orr!=0 && orr < r.decode_seq) return MK_HZ_WAR;
        }
        // WAW
        for (int t : r.write_tiles) {
            uint64_t w = tiles[t].writer;
            if (w!=0 && w < r.decode_seq) return MK_HZ_WAW;
        }
        return MK_HZ_NONE;
    }

    // returns true if page resources available.
    bool page_check(Rec& r) {
        switch (r.opcode) {
            case MK_LD: return count_free_pages(r.sm) >= 1;
            case MK_ALU: {
                for (int t : r.read_tiles) if (tiles[t].resident_page==U32MAX) return false;
                return count_free_pages(r.sm) >= (int)r.write_tiles.size();
            }
            case MK_ST: {
                int t = r.read_tiles.empty()? -1 : r.read_tiles[0];
                if (t<0) return true;
                return tiles[t].resident_page != U32MAX;
            }
            default: return true;
        }
    }

    void reserve_and_issue(Rec& r) {
        // reads
        for (int t : r.read_tiles) {
            tiles[t].reader_count += 1;
            if (tiles[t].oldest_reader==0 || tiles[t].oldest_reader > r.decode_seq)
                tiles[t].oldest_reader = r.decode_seq;
        }
        // writes
        for (int t : r.write_tiles) {
            tiles[t].writer = r.decode_seq;
            tiles[t].version = tiles[t].version + 1;  // mod 2^64
            tiles[t].dirty = 1;
        }
        // pages
        if (r.opcode==MK_LD) {
            int p = lowest_free_page(r.sm);
            Page& pg = page(r.sm, p);
            pg.state = MK_PG_LOADING; pg.owner = r.decode_seq; pg.page_key = r.page_key;
            pg.tile = (uint32_t)r.dst_tile; pg.version = tiles[r.dst_tile].version;
            r.pages.push_back(p);
        } else if (r.opcode==MK_ALU) {
            for (int t : r.write_tiles) {
                int p = lowest_free_page(r.sm);
                Page& pg = page(r.sm, p);
                pg.state = MK_PG_COMPUTING; pg.owner = r.decode_seq; pg.page_key = 0;
                pg.tile = (uint32_t)t; pg.version = tiles[t].version;
                r.pages.push_back(p);
            }
        }
        // pending unit
        int ukind;
        switch (r.opcode) {
            case MK_LD: ukind=MK_U_LD_DONE; break;
            case MK_ALU: ukind=MK_U_ALU_DONE; break;
            case MK_ST: ukind=MK_U_ST_DONE; break;
            case MK_WAIT: ukind=MK_U_WAIT_DONE; break;
            case MK_INC: ukind=MK_U_INC_DONE; break;
            case MK_FREE: ukind=MK_U_FREE_DONE; break;
            case MK_BRANCH: ukind=MK_U_BRANCH_DONE; break;
            default: ukind=MK_U_ST_DONE; break;  // NOP behaves like a no-op completion
        }
        Unit u; u.kind=ukind; u.sm=r.sm; u.decode_seq=r.decode_seq; u.epoch=r.epoch;
        u.due_clock = clock + r.latency; u.unit_seq = unit_seq_next++;
        pending.push_back(u);
        r.status = MK_ST_ISSUED;
        counts[MK_C_INSTR_ISSUE]+=1;
        emit(MK_EV_INSTR_ISSUE, r.sm, r.pc, r.decode_seq, r.instr_uid, U32MAX, 0);
    }

    void op_issue(int sm, int limit) {
        if (sm<0 || sm>=SM || limit==0) { invalid(); return; }
        int budget = std::min(limit, IW);
        int issued = 0;
        // iterate issue queue in decode_seq ascending order
        for (;;) {
            if (issued >= budget) break;
            // pick smallest decode_seq still present in issue queue
            SmState& S = sms[sm];
            if (S.issue_q.empty()) break;
            // build sorted snapshot of candidates not yet examined; but we must respect
            // head-of-line stop. Process candidates in ascending decode_seq, advancing a
            // cursor; page-stalled stay, hazard stops, issued/stale removed.
            std::vector<uint64_t> sorted(S.issue_q.begin(), S.issue_q.end());
            std::sort(sorted.begin(), sorted.end());
            bool progressed = false;
            bool stop = false;
            for (uint64_t ds : sorted) {
                if (issued >= budget) { stop=true; break; }
                // entry might have been removed during this loop; check presence
                bool present=false; for(uint64_t q : S.issue_q) if(q==ds){present=true;break;}
                if (!present) continue;
                Rec* rp = find(ds);
                if (!rp) { remove_from_queues(sm, ds); progressed=true; continue; }
                Rec& r = *rp;
                if (!(r.status==MK_ST_DECODED || r.status==MK_ST_REPLAY)) {
                    // drop stale queue entry
                    for(auto it=S.issue_q.begin();it!=S.issue_q.end();++it) if(*it==ds){S.issue_q.erase(it);break;}
                    progressed=true; continue;
                }
                int hz = hazard_check(r);
                if (hz!=MK_HZ_NONE) {
                    r.hazard_code = hz;
                    counts[MK_C_ISSUE_HAZARD]+=1;
                    emit(MK_EV_ISSUE_HAZARD, sm, r.pc, ds, r.instr_uid, U32MAX, (uint64_t)hz);
                    stop=true; break;  // head-of-line
                }
                if (!page_check(r)) {
                    counts[MK_C_ISSUE_PAGE_STALL]+=1;
                    emit(MK_EV_ISSUE_PAGE_STALL, sm, r.pc, ds, r.instr_uid, U32MAX, 0);
                    continue;  // skip page stalls, leave in queue
                }
                // issuable
                reserve_and_issue(r);
                for(auto it=S.issue_q.begin();it!=S.issue_q.end();++it) if(*it==ds){S.issue_q.erase(it);break;}
                issued++;
                progressed=true;
            }
            if (stop) break;
            if (!progressed) break;  // nothing more to do this op
        }
    }

    uint64_t alu_result_hash(Rec& r) {
        MkFnv f; f.u64(r.compute_seed);
        for (int t : r.read_tiles) { f.u32((uint32_t)t); f.u64(tiles[t].version); }
        for (int t : r.write_tiles) { f.u32((uint32_t)t); f.u64(tiles[t].version); }
        return f.h;
    }

    // returns index in pending of min (due_clock,unit_seq) with due<=clock, else -1.
    int pick_ready_unit() {
        int best=-1; uint64_t bd=0, bs=0;
        for (size_t i=0;i<pending.size();++i) {
            if (pending[i].due_clock > clock) continue;
            if (best<0 || pending[i].due_clock<bd || (pending[i].due_clock==bd && pending[i].unit_seq<bs)) {
                best=(int)i; bd=pending[i].due_clock; bs=pending[i].unit_seq;
            }
        }
        return best;
    }

    void complete_record(Rec& r) {
        r.status = MK_ST_COMPLETED;
        counts[MK_C_INSTR_COMPLETE]+=1;
        emit(MK_EV_INSTR_COMPLETE, r.sm, r.pc, r.decode_seq, r.instr_uid, U32MAX, 0);
    }

    void op_advance(uint64_t delta, int max_units) {
        clock = clock + delta;  // wraps
        int processed=0;
        while (processed < max_units) {
            int idx = pick_ready_unit();
            if (idx<0) break;
            Unit u = pending[idx];
            pending.erase(pending.begin()+idx);
            Rec* rp = find(u.decode_seq);
            if (!rp) { counts[MK_C_UNIT_STALE_DROP]+=1; emit(MK_EV_UNIT_STALE_DROP, u.sm, U32MAX, u.decode_seq, 0, U32MAX, 0); processed++; continue; }
            Rec& r = *rp;
            if (sms[u.sm].epoch != u.epoch) {
                counts[MK_C_UNIT_STALE_DROP]+=1;
                emit(MK_EV_UNIT_STALE_DROP, u.sm, r.pc, u.decode_seq, r.instr_uid, U32MAX, 0);
                processed++; continue;
            }
            bool completed = true;
            switch (u.kind) {
                case MK_U_LD_DONE: {
                    int p = r.pages.empty()? -1 : r.pages[0];
                    if (p>=0) { Page& pg=page(r.sm,p); pg.state=MK_PG_RESIDENT; pg.tile=(uint32_t)r.dst_tile; pg.version=tiles[r.dst_tile].version; tiles[r.dst_tile].resident_page=(uint32_t)p; }
                    counts[MK_C_LD_DONE]+=1; emit(MK_EV_LD_DONE, r.sm, r.pc, r.decode_seq, r.instr_uid, (uint32_t)r.dst_tile, 0);
                    break;
                }
                case MK_U_ALU_DONE: {
                    r.result_hash = alu_result_hash(r);
                    for (size_t k=0;k<r.pages.size();++k) {
                        int p = r.pages[k]; Page& pg=page(r.sm,p); pg.state=MK_PG_RESIDENT;
                        int t = r.write_tiles[k]; tiles[t].resident_page=(uint32_t)p; pg.tile=(uint32_t)t; pg.version=tiles[t].version;
                    }
                    counts[MK_C_ALU_DONE]+=1; emit(MK_EV_ALU_DONE, r.sm, r.pc, r.decode_seq, r.instr_uid, U32MAX, r.result_hash);
                    break;
                }
                case MK_U_ST_DONE: {
                    counts[MK_C_ST_DONE]+=1; emit(MK_EV_ST_DONE, r.sm, r.pc, r.decode_seq, r.instr_uid, U32MAX, 0);
                    break;
                }
                case MK_U_WAIT_DONE: {
                    if (counters[r.counter_id] >= r.target) {
                        counts[MK_C_WAIT_DONE]+=1; emit(MK_EV_WAIT_DONE, r.sm, r.pc, r.decode_seq, r.instr_uid, U32MAX, counters[r.counter_id]);
                    } else {
                        Unit nu; nu.kind=MK_U_WAIT_DONE; nu.sm=u.sm; nu.decode_seq=u.decode_seq; nu.epoch=u.epoch;
                        nu.due_clock = clock + 1; nu.unit_seq = unit_seq_next++;
                        pending.push_back(nu);
                        counts[MK_C_WAIT_REARM]+=1; emit(MK_EV_WAIT_REARM, r.sm, r.pc, r.decode_seq, r.instr_uid, U32MAX, r.target);
                        completed=false;
                    }
                    break;
                }
                case MK_U_INC_DONE: {
                    counters[r.counter_id] += r.amount;
                    counts[MK_C_COUNTER_INC]+=1; emit(MK_EV_COUNTER_INC, r.sm, r.pc, r.decode_seq, r.instr_uid, U32MAX, counters[r.counter_id]);
                    break;
                }
                case MK_U_FREE_DONE: {
                    int t = r.free_tile;
                    bool older = false;
                    if (t>=0) {
                        if (tiles[t].writer!=0 && tiles[t].writer < r.decode_seq) older=true;
                        if (tiles[t].oldest_reader!=0 && tiles[t].oldest_reader < r.decode_seq) older=true;
                    }
                    if (!older && t>=0) {
                        tiles[t].freed = 1;
                        uint32_t rp2 = tiles[t].resident_page;
                        if (rp2 != U32MAX) {
                            Page& pg = page(r.sm, (int)rp2);
                            pg.state=MK_PG_FREE; pg.page_key=0; pg.tile=U32MAX; pg.version=0; pg.owner=0;
                            tiles[t].resident_page=U32MAX;
                        }
                        counts[MK_C_TILE_FREE]+=1; emit(MK_EV_TILE_FREE, r.sm, r.pc, r.decode_seq, r.instr_uid, (uint32_t)t, 0);
                    } else {
                        r.status = MK_ST_REPLAY;
                        sms[r.sm].replay_q.push_back(r.decode_seq);
                        counts[MK_C_FREE_REPLAY]+=1; emit(MK_EV_FREE_REPLAY, r.sm, r.pc, r.decode_seq, r.instr_uid, (uint32_t)(t<0?U32MAX:t), 0);
                        completed=false;
                    }
                    break;
                }
                case MK_U_BRANCH_DONE: {
                    uint64_t resolved = (counters[r.counter_id] >= r.target) ? (uint64_t)r.taken_pc : (uint64_t)r.fallthrough_pc;
                    if (resolved != (uint64_t)r.fallthrough_pc) {
                        cancel_younger(r.sm, r.epoch, r.decode_seq);
                        sms[r.sm].pc = (uint32_t)resolved;
                        sms[r.sm].epoch = epoch_seq_next++;
                        counts[MK_C_BRANCH_REDIRECT]+=1; emit(MK_EV_BRANCH_REDIRECT, r.sm, sms[r.sm].pc, r.decode_seq, r.instr_uid, U32MAX, resolved);
                    } else {
                        counts[MK_C_BRANCH_NOOP]+=1; emit(MK_EV_BRANCH_NOOP, r.sm, r.pc, r.decode_seq, r.instr_uid, U32MAX, resolved);
                    }
                    break;
                }
                default: break;
            }
            if (completed) complete_record(r);
            processed++;
        }
    }

    // head = smallest decode_seq on sm not RETIRED/CANCELLED.
    uint64_t head_record(int sm) {
        uint64_t best=0;
        for (auto& kv : recs) {
            Rec& r = kv.second;
            if (r.sm!=sm) continue;
            if (r.status==MK_ST_RETIRED || r.status==MK_ST_CANCELLED) continue;
            if (best==0 || r.decode_seq<best) best=r.decode_seq;
        }
        return best;
    }

    void op_retire(int sm, int limit) {
        if (sm<0 || sm>=SM || limit==0) { invalid(); return; }
        int budget = std::min(limit, RW);
        int retired=0;
        while (retired < budget) {
            uint64_t head = head_record(sm);
            if (head==0) break;
            Rec& r = recs[head];
            if (r.status != MK_ST_COMPLETED) {
                counts[MK_C_RETIRE_BLOCKED]+=1; emit(MK_EV_RETIRE_BLOCKED, sm, r.pc, head, r.instr_uid, U32MAX, 0);
                break;
            }
            // read tiles
            for (int t : r.read_tiles) {
                if (tiles[t].reader_count>0) tiles[t].reader_count -= 1;
                // mark retired BEFORE recompute so it is excluded
            }
            r.status = MK_ST_RETIRED;  // exclude from recompute and head
            for (int t : r.read_tiles) {
                recompute_oldest_reader(t, sm);
                counts[MK_C_READ_RETIRE]+=1; emit(MK_EV_READ_RETIRE, sm, r.pc, head, r.instr_uid, (uint32_t)t, 0);
            }
            for (int t : r.write_tiles) {
                if (tiles[t].writer == head) tiles[t].writer = 0;
                counts[MK_C_WRITE_RETIRE]+=1; emit(MK_EV_WRITE_RETIRE, sm, r.pc, head, r.instr_uid, (uint32_t)t, 0);
            }
            uint64_t rseq = retire_seq_next++;
            counts[MK_C_INSTR_RETIRE]+=1; emit(MK_EV_INSTR_RETIRE, sm, r.pc, head, r.instr_uid, U32MAX, rseq);
            // remove from queues
            remove_from_queues(sm, head);
            retired++;
        }
    }

    void op_replay(int sm, int limit) {
        if (sm<0 || sm>=SM || limit==0) { invalid(); return; }
        SmState& S = sms[sm];
        int moved=0;
        std::vector<uint64_t> batch;
        while (moved < limit && !S.replay_q.empty()) {
            uint64_t ds = S.replay_q.front(); S.replay_q.pop_front();
            batch.push_back(ds);
            moved++;
        }
        // prepend to issue queue front preserving replay order
        for (auto it = batch.rbegin(); it != batch.rend(); ++it) S.issue_q.push_front(*it);
        for (uint64_t ds : batch) {
            Rec* rp = find(ds);
            counts[MK_C_REPLAY_ENQUEUE]+=1;
            emit(MK_EV_REPLAY_ENQUEUE, sm, rp?rp->pc:U32MAX, ds, rp?rp->instr_uid:0, U32MAX, 0);
        }
    }

    void op_host_inc(int counter_id, uint64_t amount) {
        if (counter_id<0 || counter_id>=CC) { invalid(); return; }
        counters[counter_id] += amount;
        counts[MK_C_HOST_COUNTER_INC]+=1;
        emit(MK_EV_HOST_COUNTER_INC, U32MAX, U32MAX, 0, 0, U32MAX, counters[counter_id]);
    }

    void op_flush(int sm) {
        if (sm<0 || sm>=SM) { invalid(); return; }
        // cancel all nonretired noncancelled records on sm descending.
        std::vector<uint64_t> victims;
        for (auto& kv : recs) {
            Rec& r = kv.second;
            if (r.sm!=sm) continue;
            if (r.status==MK_ST_RETIRED || r.status==MK_ST_CANCELLED) continue;
            victims.push_back(r.decode_seq);
        }
        std::sort(victims.begin(), victims.end(), std::greater<uint64_t>());
        for (uint64_t ds : victims) {
            Rec& r = recs[ds];
            bool was_issued = (r.status==MK_ST_ISSUED || r.status==MK_ST_COMPLETED || r.status==MK_ST_REPLAY);
            release_record_pages(r);
            if (was_issued) unwind_reservations(r);
            r.status = MK_ST_CANCELLED;
            counts[MK_C_EPOCH_CANCEL]+=1;
            emit(MK_EV_EPOCH_CANCEL, sm, r.pc, ds, r.instr_uid, U32MAX, 0);
        }
        for (uint64_t ds : victims) { Rec& r=recs[ds]; for(int t:r.read_tiles) recompute_oldest_reader(t,sm); }
        sms[sm].decode_q.clear(); sms[sm].issue_q.clear(); sms[sm].replay_q.clear();
        sms[sm].epoch = epoch_seq_next++;
        counts[MK_C_SM_FLUSH]+=1;
        emit(MK_EV_SM_FLUSH, sm, sms[sm].pc, 0, 0, U32MAX, 0);
    }

    // ---------------------------------------------------------------- hashes
    uint64_t hash_sm_stream() {
        MkFnv f;
        for (int s=0;s<SM;++s) {
            SmState& S = sms[s];
            f.u32((uint32_t)s); f.u32(S.pc); f.u64(S.epoch); f.u8(S.halted);
            f.u32((uint32_t)S.decode_q.size()); for (uint64_t id : S.decode_q) f.u64(id);
            f.u32((uint32_t)S.issue_q.size());  for (uint64_t id : S.issue_q) f.u64(id);
            f.u32((uint32_t)S.replay_q.size()); for (uint64_t id : S.replay_q) f.u64(id);
        }
        return f.h;
    }

    uint64_t hash_decoded() {
        MkFnv f;
        for (int s=0;s<SM;++s) {
            for (auto& kv : recs) {
                Rec& r = kv.second;
                if (r.sm!=s) continue;
                if (r.status==MK_ST_RETIRED || r.status==MK_ST_CANCELLED) continue;
                f.u32((uint32_t)r.sm); f.u32(r.pc); f.u64(r.epoch); f.u64(r.decode_seq);
                f.u64(r.instr_uid); f.u8((uint8_t)r.opcode); f.u8((uint8_t)r.status); f.u8((uint8_t)r.hazard_code);
                f.u64(r.result_hash);
                f.u32((uint32_t)r.read_tiles.size()); for(int t:r.read_tiles) f.u32((uint32_t)t);
                f.u32((uint32_t)r.write_tiles.size()); for(int t:r.write_tiles) f.u32((uint32_t)t);
                f.u32((uint32_t)r.pages.size()); for(int p:r.pages) f.u32((uint32_t)p);
            }
        }
        return f.h;
    }

    uint64_t hash_tiles() {
        MkFnv f;
        for (int t=0;t<TC;++t) {
            Tile& T = tiles[t];
            f.u32((uint32_t)t); f.u64(T.version); f.u64(T.writer); f.u64(T.oldest_reader);
            f.u64(T.reader_count); f.u32(T.resident_page); f.u8(T.dirty); f.u8(T.freed);
        }
        return f.h;
    }

    uint64_t hash_pages() {
        MkFnv f;
        for (int s=0;s<SM;++s) for (int p=0;p<PG;++p) {
            Page& pg = page(s,p);
            f.u32((uint32_t)s); f.u32((uint32_t)p); f.u8((uint8_t)pg.state);
            f.u64(pg.page_key); f.u32(pg.tile); f.u64(pg.version); f.u64(pg.owner);
        }
        return f.h;
    }

    uint64_t hash_pending() {
        MkFnv f;
        std::vector<Unit> v = pending;
        std::sort(v.begin(), v.end(), [](const Unit&a, const Unit&b){
            if (a.due_clock!=b.due_clock) return a.due_clock<b.due_clock;
            return a.unit_seq<b.unit_seq; });
        for (Unit& u : v) {
            f.u8((uint8_t)u.kind); f.u32((uint32_t)u.sm); f.u64(u.decode_seq);
            f.u64(u.epoch); f.u64(u.due_clock); f.u64(u.unit_seq);
        }
        return f.h;
    }

    uint64_t hash_counters() {
        MkFnv f;
        for (int c=0;c<CC;++c) { f.u32((uint32_t)c); f.u64(counters[c]); }
        return f.h;
    }

    uint64_t compute_state(uint64_t smh, uint64_t dh, uint64_t th, uint64_t ph, uint64_t puh, uint64_t ch) {
        MkFnv f;
        f.u64(clock); f.u64(event_seq); f.u32((uint32_t)op_index);
        f.u64(decode_seq_next); f.u64(issue_seq_next); f.u64(unit_seq_next);
        f.u64(retire_seq_next); f.u64(epoch_seq_next);
        f.u64(event_hash); f.u64(smh); f.u64(dh); f.u64(th); f.u64(ph); f.u64(puh); f.u64(ch);
        for (int i=0;i<MK_COUNT_N;++i) f.u64((uint64_t)counts[i]);
        return f.h;
    }

    void step_once(const MkRunSpec& run, MkExpected* exp) {
        switch (run.op_kind) {
            case MK_OP_BEGIN_KERNEL: op_begin(run.a_epoch_seed); break;
            case MK_OP_DECODE_SM: op_decode(run.a_sm, run.a_limit); break;
            case MK_OP_ISSUE_SM: op_issue(run.a_sm, run.a_limit); break;
            case MK_OP_ADVANCE: op_advance(run.a_delta, run.a_max_units); break;
            case MK_OP_RETIRE_SM: op_retire(run.a_sm, run.a_limit); break;
            case MK_OP_REPLAY_SM: op_replay(run.a_sm, run.a_limit); break;
            case MK_OP_HOST_INC_COUNTER: op_host_inc(run.a_counter_id, run.a_amount); break;
            case MK_OP_FLUSH_SM: op_flush(run.a_sm); break;
            default: break;
        }
        const int32_t this_op = op_index;
        op_index += 1;

        uint64_t smh = hash_sm_stream();
        uint64_t dh = hash_decoded();
        uint64_t th = hash_tiles();
        uint64_t ph = hash_pages();
        uint64_t puh = hash_pending();
        uint64_t ch = hash_counters();

        exp->counts = counts;
        exp->op_index = this_op;
        exp->clock = clock;
        exp->event_seq = event_seq;
        exp->decoder_event_hash = event_hash;
        exp->sm_stream_hash = smh;
        exp->decoded_hash = dh;
        exp->tile_scoreboard_hash = th;
        exp->page_hash = ph;
        exp->pending_unit_hash = puh;
        exp->counter_hash = ch;
        exp->state_checksum = compute_state(smh, dh, th, ph, puh, ch);
    }
};

static inline bool mk_check_outputs(const MkExpected& e, const MkHostOutputsView& g, std::string* err) {
    for (int i=0;i<MK_COUNT_N;++i) {
        if (g.counts[i]!=e.counts[(size_t)i]) {
            if (err){ std::ostringstream o; o<<"count["<<i<<"] got "<<g.counts[i]<<" exp "<<e.counts[(size_t)i]; *err=o.str(); }
            return false;
        }
    }
    auto c64=[&](const char* nm,uint64_t got,uint64_t exp)->bool{
        if (got!=exp){ if(err){std::ostringstream o;o<<nm<<" got 0x"<<std::hex<<got<<" exp 0x"<<exp;*err=o.str();} return false;} return true; };
    if (g.op_index_out[0]!=e.op_index){ if(err){std::ostringstream o;o<<"op_index got "<<g.op_index_out[0]<<" exp "<<e.op_index;*err=o.str();} return false; }
    if (!c64("clock",g.clock_out[0],e.clock)) return false;
    if (!c64("event_seq",g.event_seq_out[0],e.event_seq)) return false;
    if (!c64("decoder_event_hash",g.decoder_event_hash[0],e.decoder_event_hash)) return false;
    if (!c64("sm_stream_hash",g.sm_stream_hash[0],e.sm_stream_hash)) return false;
    if (!c64("decoded_hash",g.decoded_hash[0],e.decoded_hash)) return false;
    if (!c64("tile_scoreboard_hash",g.tile_scoreboard_hash[0],e.tile_scoreboard_hash)) return false;
    if (!c64("page_hash",g.page_hash[0],e.page_hash)) return false;
    if (!c64("pending_unit_hash",g.pending_unit_hash[0],e.pending_unit_hash)) return false;
    if (!c64("counter_hash",g.counter_hash[0],e.counter_hash)) return false;
    if (!c64("state_checksum",g.state_checksum[0],e.state_checksum)) return false;
    return true;
}

#endif  // MK_INSTR_DECODER_HAZARD_ORACLE_HPP_
