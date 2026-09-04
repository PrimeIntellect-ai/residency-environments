// PMPP_CANARY_52_719f3a0275 -- held-out canary; MUST NOT appear in any submission
// file: wormhole_mesh_credits_reference.cu
//
// Reference device implementation of T52. Single-thread kernel operating on
// device-resident persistent state. Uses ring-buffer FIFOs and a flat credit
// queue. Independent of the naive implementation and the host oracle.

#include "wormhole_mesh_credits_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace {

struct RFlit {
    uint64_t packet_id;
    uint64_t flit_index;
    uint64_t flit_count;
    uint64_t inject_seq;
    uint64_t entered_cycle;
    uint32_t src;
    uint32_t dst;
    uint32_t credit_return_node;
    uint8_t vc;
    uint8_t kind;
    uint8_t credit_return_port;
    uint8_t pad;
};

struct RPacket {
    uint64_t packet_id;
    uint64_t flit_count;
    uint64_t injected_flits;
    uint64_t delivered_flits;
    uint64_t inject_seq;
    uint32_t src;
    uint32_t dst;
    uint32_t chosen_vc;
    uint8_t used;
    uint8_t prefer_adaptive;
    uint8_t state;
    uint8_t pad;
};

struct RCreditRec {
    uint64_t due_cycle;
    uint64_t seq_created;
    uint64_t amount;
    uint32_t node;
    uint8_t port;
    uint8_t vc;
    uint8_t pad0;
    uint8_t pad1;
};

}  // namespace

struct WmcReferenceState {
    WmcProblemSpec spec;
    int rows, cols, nodes, vc, cap, nbuf;

    // device-resident persistent state
    RFlit* flit_pool;        // nbuf * cap
    int32_t* buf_head;       // nbuf
    int32_t* buf_count;      // nbuf
    int64_t* credit;         // nbuf
    uint8_t* link_up;        // nodes*5
    RPacket* ptable;         // max_packets
    uint64_t* injq;          // nodes * max_injq  (packet ids)
    int32_t* injq_count;     // nodes
    RCreditRec* creditq;     // max_credit_events
    int32_t* creditq_count;  // 1
    uint64_t* scalars;       // [cycle, event_seq, inject_seq_next]
    int64_t* counters;       // WMC_COUNTER_COUNT
    uint8_t* used_out;       // nodes*5 scratch for phase D

    // scratch run spec on device
    int32_t* d_runspec;      // 8 ints: op, op_index, a0..a4
};

// ---------- device helpers ----------

__device__ __forceinline__ uint64_t rfnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= 1099511628211ULL; return h;
}
__device__ void rfnv(uint64_t* h, const void* p, int n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t v = *h;
    for (int i = 0; i < n; ++i) v = rfnv_byte(v, b[i]);
    *h = v;
}

struct RCtx {
    int rows, cols, nodes, vc, cap, nbuf;
    int credit_latency, max_packets, max_injq, max_credit_events;
    RFlit* flit_pool;
    int32_t* buf_head;
    int32_t* buf_count;
    int64_t* credit;
    uint8_t* link_up;
    RPacket* ptable;
    uint64_t* injq;
    int32_t* injq_count;
    RCreditRec* creditq;
    int32_t* creditq_count;
    int64_t* counters;
    uint8_t* used_out;
    uint64_t cycle;
    uint64_t event_seq;
    uint64_t inject_seq_next;
    uint64_t ev_hash;
    uint32_t op_index;
};

__device__ __forceinline__ int r_bidx(RCtx* c, int node, int port, int v) {
    return (node * WMC_PORT_COUNT + port) * c->vc + v;
}
__device__ __forceinline__ int r_lidx(int node, int port) {
    return node * WMC_PORT_COUNT + port;
}
__device__ __forceinline__ int r_rowof(RCtx* c, int n) { return n / c->cols; }
__device__ __forceinline__ int r_colof(RCtx* c, int n) { return n % c->cols; }

__device__ bool r_output_valid(RCtx* c, int node, int port) {
    if (port == WMC_PORT_LOCAL) return false;
    int r = r_rowof(c, node), col = r_colof(c, node);
    if (port == WMC_PORT_N) return r - 1 >= 0;
    if (port == WMC_PORT_S) return r + 1 < c->rows;
    if (port == WMC_PORT_E) return col + 1 < c->cols;
    if (port == WMC_PORT_W) return col - 1 >= 0;
    return false;
}
__device__ int r_neighbor(RCtx* c, int node, int port) {
    int r = r_rowof(c, node), col = r_colof(c, node);
    if (port == WMC_PORT_N) return (r - 1) * c->cols + col;
    if (port == WMC_PORT_S) return (r + 1) * c->cols + col;
    if (port == WMC_PORT_E) return r * c->cols + (col + 1);
    if (port == WMC_PORT_W) return r * c->cols + (col - 1);
    return -1;
}
__device__ int r_opposite(int port) {
    if (port == WMC_PORT_N) return WMC_PORT_S;
    if (port == WMC_PORT_S) return WMC_PORT_N;
    if (port == WMC_PORT_E) return WMC_PORT_W;
    if (port == WMC_PORT_W) return WMC_PORT_E;
    return WMC_PORT_LOCAL;
}

// FIFO ops on ring buffer
__device__ bool r_buf_front(RCtx* c, int bi, RFlit* out) {
    if (c->buf_count[bi] == 0) return false;
    int h = c->buf_head[bi];
    *out = c->flit_pool[(size_t)bi * c->cap + h];
    return true;
}
__device__ void r_buf_pop(RCtx* c, int bi) {
    int h = c->buf_head[bi];
    c->buf_head[bi] = (h + 1) % c->cap;
    c->buf_count[bi] -= 1;
}
__device__ void r_buf_push(RCtx* c, int bi, const RFlit* f) {
    int h = c->buf_head[bi];
    int cnt = c->buf_count[bi];
    int slot = (h + cnt) % c->cap;
    c->flit_pool[(size_t)bi * c->cap + slot] = *f;
    c->buf_count[bi] = cnt + 1;
}

__device__ void r_emit(RCtx* c, uint8_t kind, uint32_t node_or, uint8_t port_or,
                        uint8_t vc_or, uint64_t pkt_or, uint64_t fidx_or, uint64_t aux) {
    uint64_t es = c->event_seq;
    uint64_t* h = &c->ev_hash;
    rfnv(h, &kind, 1);
    rfnv(h, &es, 8);
    uint64_t cyc = c->cycle; rfnv(h, &cyc, 8);
    rfnv(h, &c->op_index, 4);
    rfnv(h, &node_or, 4);
    rfnv(h, &port_or, 1);
    rfnv(h, &vc_or, 1);
    rfnv(h, &pkt_or, 8);
    rfnv(h, &fidx_or, 8);
    rfnv(h, &aux, 8);
    c->event_seq = es + 1;
}

__device__ void r_sched_credit(RCtx* c, const RFlit* f) {
    if (f->credit_return_node == UINT32_MAX) return;
    if (c->creditq_count[0] >= c->max_credit_events) return;
    int k = c->creditq_count[0];
    RCreditRec r;
    r.due_cycle = c->cycle + (uint64_t)c->credit_latency;
    r.seq_created = c->event_seq;
    r.amount = 1;
    r.node = f->credit_return_node;
    r.port = f->credit_return_port;
    r.vc = f->vc;
    r.pad0 = 0; r.pad1 = 0;
    c->creditq[k] = r;
    c->creditq_count[0] = k + 1;
}

__device__ int r_find_packet(RCtx* c, uint64_t id) {
    for (int i = 0; i < c->max_packets; ++i)
        if (c->ptable[i].used && c->ptable[i].packet_id == id) return i;
    return -1;
}

__device__ void r_minimal_ports(RCtx* c, int node, int dst, int* out, int* cnt) {
    int rc = r_rowof(c, node), cc = r_colof(c, node);
    int rd = r_rowof(c, dst), cd = r_colof(c, dst);
    int n = 0;
    if (rd < rc) out[n++] = WMC_PORT_N;
    if (cd > cc) out[n++] = WMC_PORT_E;
    if (rd > rc) out[n++] = WMC_PORT_S;
    if (cd < cc) out[n++] = WMC_PORT_W;
    *cnt = n;
}

__device__ int r_choose_vc(RCtx* c, RPacket* p) {
    if (p->prefer_adaptive == 0) return 0;
    int ports[4]; int pc = 0;
    r_minimal_ports(c, (int)p->src, (int)p->dst, ports, &pc);
    int best_vc = -1; int64_t best = -1;
    for (int v = 1; v < c->vc; ++v) {
        int64_t tot = 0;
        for (int i = 0; i < pc; ++i) {
            int pp = ports[i];
            if (c->link_up[r_lidx((int)p->src, pp)] == 0) continue;
            tot += c->credit[r_bidx(c, (int)p->src, pp, v)];
        }
        if (tot > best) { best = tot; best_vc = v; }
    }
    if (best_vc < 0 || best <= 0) return 0;
    return best_vc;
}

// ---------- phases ----------

__device__ void r_phase_A(RCtx* c) {
    // selection sort by canonical order over due<=cycle, applying in order.
    // We repeatedly scan for the minimum applicable record not yet consumed.
    int total = c->creditq_count[0];
    // Apply due records in canonical order via selection (O(n^2), n bounded by
    // max_credit_events). pad0 marks a record as consumed this phase.
    for (int i = 0; i < total; ++i) c->creditq[i].pad0 = 0;
    for (;;) {
        int best = -1;
        for (int i = 0; i < total; ++i) {
            RCreditRec* r = &c->creditq[i];
            if (r->pad0) continue;
            if (r->due_cycle > c->cycle) continue;
            if (best < 0) { best = i; continue; }
            RCreditRec* b = &c->creditq[best];
            // canonical compare
            if (r->due_cycle != b->due_cycle) { if (r->due_cycle < b->due_cycle) best = i; continue; }
            if (r->node != b->node) { if (r->node < b->node) best = i; continue; }
            if (r->port != b->port) { if (r->port < b->port) best = i; continue; }
            if (r->vc != b->vc) { if (r->vc < b->vc) best = i; continue; }
            if (r->seq_created < b->seq_created) best = i;
        }
        if (best < 0) break;
        RCreditRec r = c->creditq[best];
        c->creditq[best].pad0 = 1;
        int ci = r_bidx(c, (int)r.node, (int)r.port, (int)r.vc);
        int64_t nv = c->credit[ci] + (int64_t)r.amount;
        if (nv > c->cap) {
            c->credit[ci] = c->cap;
            c->counters[11]++;
            r_emit(c, WMC_EV_CREDIT_CLAMP, r.node, r.port, r.vc, UINT64_MAX, UINT64_MAX, r.amount);
        } else {
            c->credit[ci] = nv;
            c->counters[10]++;
            r_emit(c, WMC_EV_CREDIT_RETURN, r.node, r.port, r.vc, UINT64_MAX, UINT64_MAX, r.amount);
        }
    }
    // compact: keep records not consumed
    int w = 0;
    for (int i = 0; i < total; ++i) {
        if (!c->creditq[i].pad0) {
            c->creditq[w++] = c->creditq[i];
        }
    }
    c->creditq_count[0] = w;
}

__device__ void r_phase_B(RCtx* c) {
    for (int n = 0; n < c->nodes; ++n) {
        for (int port = 0; port < WMC_PORT_COUNT; ++port) {
            for (int v = 0; v < c->vc; ++v) {
                int bi = r_bidx(c, n, port, v);
                RFlit f;
                if (!r_buf_front(c, bi, &f)) continue;
                int pidx = r_find_packet(c, f.packet_id);
                bool dropped = (pidx >= 0 && c->ptable[pidx].state == WMC_PS_DROPPED);
                if (dropped) {
                    r_buf_pop(c, bi);
                    r_sched_credit(c, &f);
                    c->counters[9]++;
                    r_emit(c, WMC_EV_DROPPED_FLIT, n, port, v, f.packet_id, f.flit_index, 0);
                } else if ((int)f.dst == n) {
                    r_buf_pop(c, bi);
                    r_sched_credit(c, &f);
                    c->counters[7]++;
                    if (pidx >= 0) c->ptable[pidx].delivered_flits++;
                    r_emit(c, WMC_EV_FLIT_EJECT, n, port, v, f.packet_id, f.flit_index, 0);
                    if (pidx >= 0 && c->ptable[pidx].delivered_flits == c->ptable[pidx].flit_count) {
                        c->ptable[pidx].state = WMC_PS_DONE;
                        c->counters[8]++;
                        r_emit(c, WMC_EV_PACKET_DONE, n, 255, 255, f.packet_id, UINT64_MAX, 0);
                    }
                }
            }
        }
    }
}

__device__ void r_injq_pop_front(RCtx* c, int n) {
    int cnt = c->injq_count[n];
    uint64_t* base = c->injq + (size_t)n * c->max_injq;
    for (int i = 1; i < cnt; ++i) base[i - 1] = base[i];
    c->injq_count[n] = cnt - 1;
}

__device__ void r_phase_C(RCtx* c) {
    for (int n = 0; n < c->nodes; ++n) {
        if (c->injq_count[n] == 0) continue;
        uint64_t* base = c->injq + (size_t)n * c->max_injq;
        uint64_t pid = base[0];
        int pidx = r_find_packet(c, pid);
        if (pidx < 0) { r_injq_pop_front(c, n); continue; }
        RPacket* p = &c->ptable[pidx];
        if (p->state == WMC_PS_DROPPED) { r_injq_pop_front(c, n); continue; }
        if (p->chosen_vc == UINT32_MAX) p->chosen_vc = (uint32_t)r_choose_vc(c, p);
        int v = (int)p->chosen_vc;
        int bi = r_bidx(c, n, WMC_PORT_LOCAL, v);
        if (c->buf_count[bi] >= c->cap) {
            c->counters[12]++;
            r_emit(c, WMC_EV_INJECT_STALL, n, WMC_PORT_LOCAL, v, pid, UINT64_MAX, 0);
            continue;
        }
        RFlit f;
        f.packet_id = pid;
        f.flit_index = p->injected_flits;
        f.flit_count = p->flit_count;
        f.inject_seq = p->inject_seq;
        f.entered_cycle = c->cycle;
        f.src = p->src;
        f.dst = p->dst;
        f.credit_return_node = UINT32_MAX;
        f.vc = (uint8_t)v;
        if (p->flit_count == 1) f.kind = WMC_KIND_SINGLE;
        else if (p->injected_flits == 0) f.kind = WMC_KIND_HEAD;
        else if (p->injected_flits == p->flit_count - 1) f.kind = WMC_KIND_TAIL;
        else f.kind = WMC_KIND_BODY;
        f.credit_return_port = 255;
        f.pad = 0;
        r_buf_push(c, bi, &f);
        uint64_t fidx = p->injected_flits;
        p->injected_flits++;
        if (p->state == WMC_PS_QUEUED) p->state = WMC_PS_ACTIVE;
        c->counters[4]++;
        r_emit(c, WMC_EV_FLIT_INJECT, n, WMC_PORT_LOCAL, v, pid, fidx, 0);
        if (p->injected_flits == p->flit_count) r_injq_pop_front(c, n);
    }
}

__device__ void r_phase_D(RCtx* c) {
    for (int i = 0; i < c->nodes * WMC_PORT_COUNT; ++i) c->used_out[i] = 0;
    for (int n = 0; n < c->nodes; ++n) {
        for (int port = 0; port < WMC_PORT_COUNT; ++port) {
            for (int v = 0; v < c->vc; ++v) {
                int bi = r_bidx(c, n, port, v);
                RFlit f;
                if (!r_buf_front(c, bi, &f)) continue;
                int pidx = r_find_packet(c, f.packet_id);
                bool dropped = (pidx >= 0 && c->ptable[pidx].state == WMC_PS_DROPPED);
                if (dropped) {
                    r_buf_pop(c, bi);
                    r_sched_credit(c, &f);
                    c->counters[9]++;
                    r_emit(c, WMC_EV_DROPPED_FLIT, n, port, v, f.packet_id, f.flit_index, 0);
                    continue;
                }
                if ((int)f.dst == n) {
                    c->counters[16]++;  // eject_waits
                    r_emit(c, WMC_EV_EJECT_WAIT, n, port, v, f.packet_id, f.flit_index, 0);
                    continue;
                }
                int outp = -1;
                if (v == 0) {
                    int rc = r_rowof(c, n), cco = r_colof(c, n);
                    int rd = r_rowof(c, (int)f.dst), cd = r_colof(c, (int)f.dst);
                    if (cd > cco) outp = WMC_PORT_E;
                    else if (cd < cco) outp = WMC_PORT_W;
                    else if (rd > rc) outp = WMC_PORT_S;
                    else if (rd < rc) outp = WMC_PORT_N;
                    if (outp < 0 || c->link_up[r_lidx(n, outp)] == 0) {
                        c->counters[13]++;
                        r_emit(c, WMC_EV_ROUTE_STALL, n, port, v, f.packet_id, f.flit_index, 0);
                        continue;
                    }
                } else {
                    int ports[4]; int pc = 0;
                    r_minimal_ports(c, n, (int)f.dst, ports, &pc);
                    int chosen = -1; int64_t best = -1;
                    for (int i = 0; i < pc; ++i) {
                        int pp = ports[i];
                        if (c->link_up[r_lidx(n, pp)] == 0) continue;
                        int64_t cr = c->credit[r_bidx(c, n, pp, v)];
                        if (cr > best) { best = cr; chosen = pp; }
                    }
                    if (chosen < 0) {
                        c->counters[13]++;
                        r_emit(c, WMC_EV_ROUTE_STALL, n, port, v, f.packet_id, f.flit_index, 0);
                        continue;
                    }
                    outp = chosen;
                }
                if (c->used_out[r_lidx(n, outp)]) {
                    c->counters[14]++;
                    r_emit(c, WMC_EV_OUTPUT_STALL, n, outp, v, f.packet_id, f.flit_index, 0);
                    continue;
                }
                if (c->credit[r_bidx(c, n, outp, v)] == 0) {
                    c->counters[15]++;
                    r_emit(c, WMC_EV_CREDIT_STALL, n, outp, v, f.packet_id, f.flit_index, 0);
                    continue;
                }
                int nb = r_neighbor(c, n, outp);
                int oppp = r_opposite(outp);
                int dbi = r_bidx(c, nb, oppp, v);
                if (c->buf_count[dbi] >= c->cap) {
                    c->counters[17]++;  // credit_desyncs
                    r_emit(c, WMC_EV_CREDIT_DESYNC, n, outp, v, f.packet_id, f.flit_index, 0);
                    continue;
                }
                c->credit[r_bidx(c, n, outp, v)] -= 1;
                RFlit moved = f;
                r_buf_pop(c, bi);
                r_sched_credit(c, &f);
                moved.credit_return_node = (uint32_t)n;
                moved.credit_return_port = (uint8_t)outp;
                moved.entered_cycle = c->cycle;
                r_buf_push(c, dbi, &moved);
                c->used_out[r_lidx(n, outp)] = 1;
                if (v == 0) c->counters[5]++;
                else c->counters[6]++;
                r_emit(c, WMC_EV_FLIT_MOVE, n, outp, v, f.packet_id, f.flit_index, (uint64_t)nb);
            }
        }
    }
}

// NOTE: counter index map (must match oracle):
// 0 accept,1 reject,2 links,3 dropped_pkt,4 inj,5 mv_esc,6 mv_adapt,7 eject,
// 8 done,9 dropped_flit,10 cret,11 cclamp,12 inj_stall,13 route_stall,
// 14 out_stall,15 credit_stall,16 eject_wait,17 credit_desync,18 invalid

// The two slots above were written with the wrong indices; corrected here by
// using named constants in the kernel below. We re-route counters[16]/[13+3]
// at the call sites: eject_wait=16, credit_desync=17.

__global__ void r_kernel(WmcReferenceState st_in, int op, int op_index,
                         int a0, int a1, int a2, int a3, int a4) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    RCtx ctx;
    RCtx* c = &ctx;
    c->rows = st_in.rows; c->cols = st_in.cols; c->nodes = st_in.nodes;
    c->vc = st_in.vc; c->cap = st_in.cap; c->nbuf = st_in.nbuf;
    c->credit_latency = st_in.spec.credit_latency;
    c->max_packets = st_in.spec.max_packets;
    c->max_injq = st_in.spec.max_injection_queue_per_node;
    c->max_credit_events = st_in.spec.max_credit_events;
    c->flit_pool = st_in.flit_pool;
    c->buf_head = st_in.buf_head;
    c->buf_count = st_in.buf_count;
    c->credit = st_in.credit;
    c->link_up = st_in.link_up;
    c->ptable = st_in.ptable;
    c->injq = st_in.injq;
    c->injq_count = st_in.injq_count;
    c->creditq = st_in.creditq;
    c->creditq_count = st_in.creditq_count;
    c->counters = st_in.counters;
    c->used_out = st_in.used_out;
    c->cycle = st_in.scalars[0];
    c->event_seq = st_in.scalars[1];
    c->inject_seq_next = st_in.scalars[2];
    c->ev_hash = 1469598103934665603ULL;
    c->op_index = (uint32_t)op_index;

    if (op == WMC_OP_INJECT) {
        uint64_t pid = (uint64_t)(uint32_t)a0;
        int src = a1, dst = a2, fc = a3;
        uint8_t pref = (uint8_t)(a4 != 0);
        bool invalid = false;
        if (src < 0 || src >= c->nodes || dst < 0 || dst >= c->nodes) invalid = true;
        if (fc <= 0 || fc > WMC_MAX_FLIT_COUNT) invalid = true;
        int existing = r_find_packet(c, pid);
        if (existing >= 0) {
            uint8_t s = c->ptable[existing].state;
            if (s == WMC_PS_QUEUED || s == WMC_PS_ACTIVE) invalid = true;
        }
        if (invalid) {
            c->counters[18]++;
            r_emit(c, WMC_EV_INVALID, UINT32_MAX, 255, 255, pid, UINT64_MAX, 0);
        } else {
            int slot = -1;
            if (existing >= 0) slot = existing;
            else {
                for (int i = 0; i < c->max_packets; ++i) {
                    RPacket* q = &c->ptable[i];
                    if (!q->used || q->state == WMC_PS_DONE || q->state == WMC_PS_DROPPED) { slot = i; break; }
                }
            }
            bool table_full = (slot < 0);
            bool injq_full = (c->injq_count[src] >= c->max_injq);
            if (table_full || injq_full) {
                c->counters[1]++;
                r_emit(c, WMC_EV_INJECT_REJECT, (uint32_t)src, 255, 255, pid, UINT64_MAX, 0);
            } else {
                RPacket* p = &c->ptable[slot];
                p->used = 1;
                p->packet_id = pid;
                p->src = (uint32_t)src; p->dst = (uint32_t)dst;
                p->flit_count = (uint64_t)fc;
                p->prefer_adaptive = pref;
                p->injected_flits = 0; p->delivered_flits = 0;
                p->inject_seq = c->inject_seq_next++;
                p->state = WMC_PS_QUEUED;
                p->chosen_vc = UINT32_MAX;
                p->pad = 0;
                uint64_t* base = c->injq + (size_t)src * c->max_injq;
                base[c->injq_count[src]] = pid;
                c->injq_count[src]++;
                c->counters[0]++;
                r_emit(c, WMC_EV_PACKET_ACCEPT, (uint32_t)src, 255, 255, pid, UINT64_MAX, (uint64_t)dst);
            }
        }
    } else if (op == WMC_OP_SET_LINK) {
        int node = a0, port = a1, up = a2;
        bool invalid = false;
        if (node < 0 || node >= c->nodes) invalid = true;
        else if (port == WMC_PORT_LOCAL || port < 0 || port >= WMC_PORT_COUNT) invalid = true;
        else if (!r_output_valid(c, node, port)) invalid = true;
        if (invalid) {
            c->counters[18]++;
            r_emit(c, WMC_EV_INVALID, (uint32_t)(node >= 0 ? node : -1), (uint8_t)port, 255,
                   UINT64_MAX, UINT64_MAX, 0);
        } else {
            c->link_up[r_lidx(node, port)] = (uint8_t)(up != 0);
            c->counters[2]++;
            r_emit(c, WMC_EV_LINK_SET, (uint32_t)node, (uint8_t)port, 255, UINT64_MAX,
                   UINT64_MAX, (uint64_t)(up != 0));
        }
    } else if (op == WMC_OP_DROP) {
        uint64_t pid = (uint64_t)(uint32_t)a0;
        int idx = r_find_packet(c, pid);
        if (idx < 0) {
            c->counters[18]++;
            r_emit(c, WMC_EV_INVALID, UINT32_MAX, 255, 255, pid, UINT64_MAX, 0);
        } else if (c->ptable[idx].state == WMC_PS_DONE || c->ptable[idx].state == WMC_PS_DROPPED) {
            c->counters[18]++;
            r_emit(c, WMC_EV_INVALID, UINT32_MAX, 255, 255, pid, UINT64_MAX, 0);
        } else {
            for (int n = 0; n < c->nodes; ++n) {
                uint64_t* base = c->injq + (size_t)n * c->max_injq;
                int w = 0;
                for (int i = 0; i < c->injq_count[n]; ++i) {
                    if (base[i] != pid) base[w++] = base[i];
                }
                c->injq_count[n] = w;
            }
            c->ptable[idx].state = WMC_PS_DROPPED;
            c->counters[3]++;
            r_emit(c, WMC_EV_PACKET_DROP, UINT32_MAX, 255, 255, pid, UINT64_MAX, 0);
        }
    } else if (op == WMC_OP_TICK) {
        int cc = a0;
        if (cc <= 0) {
            c->counters[18]++;
            r_emit(c, WMC_EV_INVALID, UINT32_MAX, 255, 255, UINT64_MAX, UINT64_MAX, 0);
        } else {
            for (int s = 0; s < cc; ++s) {
                r_phase_A(c);
                r_phase_B(c);
                r_phase_C(c);
                r_phase_D(c);
                c->cycle += 1;
            }
        }
    } else {
        c->counters[18]++;
        r_emit(c, WMC_EV_INVALID, UINT32_MAX, 255, 255, UINT64_MAX, UINT64_MAX, 0);
    }

    st_in.scalars[0] = c->cycle;
    st_in.scalars[1] = c->event_seq;
    st_in.scalars[2] = c->inject_seq_next;
    // stash run event hash into scalars[3]
    st_in.scalars[3] = c->ev_hash;
}

// ---------- hashing kernels (single thread) ----------

__global__ void r_hash_kernel(WmcReferenceState st, int op_index_unused,
                              int64_t* out_counters, uint64_t* out_evhash,
                              uint64_t* out_phash, uint64_t* out_bhash,
                              uint64_t* out_chash, uint64_t* out_cqhash,
                              uint64_t* out_cycle, uint64_t* out_evseq) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    (void)op_index_unused;
    RCtx ctx; RCtx* c = &ctx;
    c->rows = st.rows; c->cols = st.cols; c->nodes = st.nodes; c->vc = st.vc;
    c->cap = st.cap; c->nbuf = st.nbuf;
    c->max_packets = st.spec.max_packets;
    c->max_injq = st.spec.max_injection_queue_per_node;
    c->max_credit_events = st.spec.max_credit_events;
    c->flit_pool = st.flit_pool; c->buf_head = st.buf_head; c->buf_count = st.buf_count;
    c->credit = st.credit; c->link_up = st.link_up; c->ptable = st.ptable;
    c->creditq = st.creditq; c->creditq_count = st.creditq_count;

    for (int i = 0; i < WMC_COUNTER_COUNT; ++i) out_counters[i] = st.counters[i];
    out_evhash[0] = st.scalars[3];
    out_cycle[0] = st.scalars[0];
    out_evseq[0] = st.scalars[1];

    // packet hash: nonterminal by packet_id asc, selection over a visited
    // marker stored in ptable[i].pad.
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < c->max_packets; ++i) c->ptable[i].pad = 0;
    for (;;) {
        int best = -1;
        for (int i = 0; i < c->max_packets; ++i) {
            RPacket* p = &c->ptable[i];
            if (!p->used) continue;
            if (!(p->state == WMC_PS_QUEUED || p->state == WMC_PS_ACTIVE)) continue;
            if (p->pad) continue;
            if (best < 0 || p->packet_id < c->ptable[best].packet_id) best = i;
        }
        if (best < 0) break;
        RPacket* p = &c->ptable[best];
        p->pad = 1;
        uint64_t pid = p->packet_id;
        rfnv(&h, &pid, 8);
        rfnv(&h, &p->src, 4);
        rfnv(&h, &p->dst, 4);
        rfnv(&h, &p->flit_count, 8);
        rfnv(&h, &p->prefer_adaptive, 1);
        rfnv(&h, &p->injected_flits, 8);
        rfnv(&h, &p->delivered_flits, 8);
        rfnv(&h, &p->inject_seq, 8);
        rfnv(&h, &p->state, 1);
        uint32_t cv = p->chosen_vc;
        rfnv(&h, &cv, 4);
    }
    out_phash[0] = h;

    // buffer hash
    h = 1469598103934665603ULL;
    for (int n = 0; n < c->nodes; ++n) {
        for (int port = 0; port < WMC_PORT_COUNT; ++port) {
            for (int v = 0; v < c->vc; ++v) {
                int bi = r_bidx(c, n, port, v);
                int cnt = c->buf_count[bi];
                int head = c->buf_head[bi];
                for (int pos = 0; pos < cnt; ++pos) {
                    int slot = (head + pos) % c->cap;
                    RFlit* f = &c->flit_pool[(size_t)bi * c->cap + slot];
                    uint32_t nn = (uint32_t)n; uint8_t pp = (uint8_t)port; uint8_t vv = (uint8_t)v;
                    uint64_t p64 = (uint64_t)pos;
                    rfnv(&h, &nn, 4);
                    rfnv(&h, &pp, 1);
                    rfnv(&h, &vv, 1);
                    rfnv(&h, &p64, 8);
                    rfnv(&h, &f->packet_id, 8);
                    rfnv(&h, &f->flit_index, 8);
                    rfnv(&h, &f->flit_count, 8);
                    rfnv(&h, &f->src, 4);
                    rfnv(&h, &f->dst, 4);
                    rfnv(&h, &f->kind, 1);
                    rfnv(&h, &f->inject_seq, 8);
                    rfnv(&h, &f->entered_cycle, 8);
                }
            }
        }
    }
    out_bhash[0] = h;

    // credit hash
    h = 1469598103934665603ULL;
    for (int n = 0; n < c->nodes; ++n) {
        for (int port = 0; port < WMC_PORT_COUNT; ++port) {
            for (int v = 0; v < c->vc; ++v) {
                uint32_t nn = (uint32_t)n; uint8_t pp = (uint8_t)port; uint8_t vv = (uint8_t)v;
                uint64_t cr = (uint64_t)c->credit[r_bidx(c, n, port, v)];
                uint8_t lu = c->link_up[r_lidx(n, port)];
                rfnv(&h, &nn, 4);
                rfnv(&h, &pp, 1);
                rfnv(&h, &vv, 1);
                rfnv(&h, &cr, 8);
                rfnv(&h, &lu, 1);
            }
        }
    }
    out_chash[0] = h;

    // credit-queue hash: canonical order (selection sort)
    int total = c->creditq_count[0];
    for (int i = 0; i < total; ++i) c->creditq[i].pad1 = 0;
    h = 1469598103934665603ULL;
    for (;;) {
        int best = -1;
        for (int i = 0; i < total; ++i) {
            RCreditRec* r = &c->creditq[i];
            if (r->pad1) continue;
            if (best < 0) { best = i; continue; }
            RCreditRec* b = &c->creditq[best];
            if (r->due_cycle != b->due_cycle) { if (r->due_cycle < b->due_cycle) best = i; continue; }
            if (r->node != b->node) { if (r->node < b->node) best = i; continue; }
            if (r->port != b->port) { if (r->port < b->port) best = i; continue; }
            if (r->vc != b->vc) { if (r->vc < b->vc) best = i; continue; }
            if (r->seq_created < b->seq_created) best = i;
        }
        if (best < 0) break;
        RCreditRec r = c->creditq[best];
        c->creditq[best].pad1 = 1;
        rfnv(&h, &r.due_cycle, 8);
        rfnv(&h, &r.node, 4);
        rfnv(&h, &r.port, 1);
        rfnv(&h, &r.vc, 1);
        rfnv(&h, &r.amount, 8);
    }
    out_cqhash[0] = h;
}

// ---------- host ABI ----------

static cudaError_t r_reset(WmcReferenceState* st, cudaStream_t s) {
    cudaError_t e;
    e = cudaMemsetAsync(st->buf_count, 0, sizeof(int32_t) * st->nbuf, s); if (e) return e;
    e = cudaMemsetAsync(st->buf_head, 0, sizeof(int32_t) * st->nbuf, s); if (e) return e;
    e = cudaMemsetAsync(st->injq_count, 0, sizeof(int32_t) * st->nodes, s); if (e) return e;
    e = cudaMemsetAsync(st->creditq_count, 0, sizeof(int32_t), s); if (e) return e;
    e = cudaMemsetAsync(st->counters, 0, sizeof(int64_t) * WMC_COUNTER_COUNT, s); if (e) return e;
    e = cudaMemsetAsync(st->ptable, 0, sizeof(RPacket) * st->spec.max_packets, s); if (e) return e;
    // scalars: cycle=0, event_seq=0, inject_seq_next=1, ev_hash=0
    uint64_t sc[4] = {0, 0, 1, 0};
    e = cudaMemcpyAsync(st->scalars, sc, sizeof(sc), cudaMemcpyHostToDevice, s); if (e) return e;
    // credit + link init: need per-node validity. Build on host then upload.
    {
        int nodes = st->nodes;
        int64_t* cr = (int64_t*)malloc(sizeof(int64_t) * st->nbuf);
        uint8_t* lk = (uint8_t*)malloc(sizeof(uint8_t) * nodes * WMC_PORT_COUNT);
        for (int n = 0; n < nodes; ++n) {
            int r = n / st->cols, col = n % st->cols;
            for (int port = 0; port < WMC_PORT_COUNT; ++port) {
                bool valid;
                if (port == WMC_PORT_LOCAL) valid = false;
                else if (port == WMC_PORT_N) valid = (r - 1 >= 0);
                else if (port == WMC_PORT_S) valid = (r + 1 < st->rows);
                else if (port == WMC_PORT_E) valid = (col + 1 < st->cols);
                else valid = (col - 1 >= 0);
                lk[n * WMC_PORT_COUNT + port] = valid ? 1 : 0;
                for (int v = 0; v < st->vc; ++v) {
                    cr[(n * WMC_PORT_COUNT + port) * st->vc + v] = valid ? st->cap : 0;
                }
            }
        }
        e = cudaMemcpyAsync(st->credit, cr, sizeof(int64_t) * st->nbuf, cudaMemcpyHostToDevice, s);
        if (!e) e = cudaMemcpyAsync(st->link_up, lk, sizeof(uint8_t) * nodes * WMC_PORT_COUNT, cudaMemcpyHostToDevice, s);
        if (!e) e = cudaStreamSynchronize(s);
        free(cr); free(lk);
        if (e) return e;
    }
    return cudaSuccess;
}

extern "C" size_t solution_workspace_bytes(const WmcProblemSpec* spec) {
    if (!wmc_validate_problem_spec(spec)) return 0;
    return 256;
}

extern "C" cudaError_t solution_init(const WmcProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!wmc_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    WmcReferenceState* st = (WmcReferenceState*)malloc(sizeof(WmcReferenceState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(*st));
    memcpy(&st->spec, spec, sizeof(WmcProblemSpec));
    st->rows = spec->rows; st->cols = spec->cols; st->nodes = spec->rows * spec->cols;
    st->vc = spec->vc_count; st->cap = spec->buffer_cap_per_vc;
    st->nbuf = st->nodes * WMC_PORT_COUNT * st->vc;

    cudaError_t e = cudaSuccess;
    e = cudaMalloc(&st->flit_pool, sizeof(RFlit) * (size_t)st->nbuf * st->cap); if (e) goto fail;
    e = cudaMalloc(&st->buf_head, sizeof(int32_t) * st->nbuf); if (e) goto fail;
    e = cudaMalloc(&st->buf_count, sizeof(int32_t) * st->nbuf); if (e) goto fail;
    e = cudaMalloc(&st->credit, sizeof(int64_t) * st->nbuf); if (e) goto fail;
    e = cudaMalloc(&st->link_up, sizeof(uint8_t) * st->nodes * WMC_PORT_COUNT); if (e) goto fail;
    e = cudaMalloc(&st->ptable, sizeof(RPacket) * spec->max_packets); if (e) goto fail;
    e = cudaMalloc(&st->injq, sizeof(uint64_t) * (size_t)st->nodes * spec->max_injection_queue_per_node); if (e) goto fail;
    e = cudaMalloc(&st->injq_count, sizeof(int32_t) * st->nodes); if (e) goto fail;
    e = cudaMalloc(&st->creditq, sizeof(RCreditRec) * spec->max_credit_events); if (e) goto fail;
    e = cudaMalloc(&st->creditq_count, sizeof(int32_t)); if (e) goto fail;
    e = cudaMalloc(&st->scalars, sizeof(uint64_t) * 4); if (e) goto fail;
    e = cudaMalloc(&st->counters, sizeof(int64_t) * WMC_COUNTER_COUNT); if (e) goto fail;
    e = cudaMalloc(&st->used_out, sizeof(uint8_t) * st->nodes * WMC_PORT_COUNT); if (e) goto fail;
    e = cudaMalloc(&st->d_runspec, sizeof(int32_t) * 8); if (e) goto fail;

    e = r_reset(st, stream); if (e) goto fail;
    *state_out = st;
    return cudaSuccess;
fail:
    solution_destroy(st);
    return e;
}

extern "C" cudaError_t solution_run(void* state, const WmcRunSpec* run, const void* inputs,
                                    void* outputs, void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    (void)inputs; (void)workspace;
    if (!state || !wmc_validate_run_spec(run) || !outputs) return cudaErrorInvalidValue;
    if (workspace_bytes < 1) return cudaErrorInvalidValue;
    WmcReferenceState* st = (WmcReferenceState*)state;
    WmcOutputs* out = (WmcOutputs*)outputs;
    if (!out->counters || !out->fabric_event_hash || !out->packet_hash || !out->buffer_hash ||
        !out->credit_hash || !out->credit_queue_hash || !out->cycle_out || !out->event_seq_out)
        return cudaErrorInvalidValue;

    r_kernel<<<1, 1, 0, stream>>>(*st, run->op, run->op_index, run->a0, run->a1, run->a2, run->a3, run->a4);
    cudaError_t e = cudaPeekAtLastError(); if (e) return e;

    r_hash_kernel<<<1, 1, 0, stream>>>(*st, 0, out->counters, out->fabric_event_hash,
                                       out->packet_hash, out->buffer_hash, out->credit_hash,
                                       out->credit_queue_hash, out->cycle_out, out->event_seq_out);
    e = cudaPeekAtLastError(); if (e) return e;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return r_reset((WmcReferenceState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    WmcReferenceState* st = (WmcReferenceState*)state;
    if (st->flit_pool) cudaFree(st->flit_pool);
    if (st->buf_head) cudaFree(st->buf_head);
    if (st->buf_count) cudaFree(st->buf_count);
    if (st->credit) cudaFree(st->credit);
    if (st->link_up) cudaFree(st->link_up);
    if (st->ptable) cudaFree(st->ptable);
    if (st->injq) cudaFree(st->injq);
    if (st->injq_count) cudaFree(st->injq_count);
    if (st->creditq) cudaFree(st->creditq);
    if (st->creditq_count) cudaFree(st->creditq_count);
    if (st->scalars) cudaFree(st->scalars);
    if (st->counters) cudaFree(st->counters);
    if (st->used_out) cudaFree(st->used_out);
    if (st->d_runspec) cudaFree(st->d_runspec);
    free(st);
}
