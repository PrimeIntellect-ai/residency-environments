// file: wormhole_mesh_credits_common.h

#ifndef WORMHOLE_MESH_CREDITS_COMMON_H_
#define WORMHOLE_MESH_CREDITS_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define WMC_ABI_VERSION 1

// Bounds for the persistent fabric. These bound the device allocations.
#define WMC_MIN_ROWS 1
#define WMC_MAX_ROWS 16
#define WMC_MIN_COLS 1
#define WMC_MAX_COLS 16
#define WMC_MIN_VC 2
#define WMC_MAX_VC 4
#define WMC_MIN_BUFCAP 1
#define WMC_MAX_BUFCAP 8
#define WMC_MAX_PACKETS 256
#define WMC_MAX_INJQ 16
#define WMC_MAX_CREDIT_EVENTS 4096
#define WMC_MAX_FLIT_COUNT 16
#define WMC_MAX_CREDIT_LATENCY 8

// Port ordinals.
#define WMC_PORT_LOCAL 0
#define WMC_PORT_N 1
#define WMC_PORT_E 2
#define WMC_PORT_S 3
#define WMC_PORT_W 4
#define WMC_PORT_COUNT 5

// Flit kinds.
#define WMC_KIND_HEAD 0
#define WMC_KIND_BODY 1
#define WMC_KIND_TAIL 2
#define WMC_KIND_SINGLE 3

// Packet states.
#define WMC_PS_QUEUED 0
#define WMC_PS_ACTIVE 1
#define WMC_PS_DONE 2
#define WMC_PS_DROPPED 3

// Operation codes carried in the run spec.
#define WMC_OP_INJECT 0
#define WMC_OP_SET_LINK 1
#define WMC_OP_DROP 2
#define WMC_OP_TICK 3

// Event kinds (emission order hashing). Ordinals are stable; the spec lists
// them in this order.
#define WMC_EV_PACKET_ACCEPT 0
#define WMC_EV_INJECT_REJECT 1
#define WMC_EV_LINK_SET 2
#define WMC_EV_PACKET_DROP 3
#define WMC_EV_CREDIT_RETURN 4
#define WMC_EV_CREDIT_CLAMP 5
#define WMC_EV_DROPPED_FLIT 6
#define WMC_EV_FLIT_EJECT 7
#define WMC_EV_PACKET_DONE 8
#define WMC_EV_INJECT_STALL 9
#define WMC_EV_FLIT_INJECT 10
#define WMC_EV_EJECT_WAIT 11
#define WMC_EV_ROUTE_STALL 12
#define WMC_EV_OUTPUT_STALL 13
#define WMC_EV_CREDIT_STALL 14
#define WMC_EV_FLIT_MOVE 15
#define WMC_EV_CREDIT_DESYNC 16
#define WMC_EV_INVALID 17

/*
CONTRACT: wormhole_mesh_credits  (T52)

Credit-Based Wormhole Mesh with Escape VC and Link Failures.

A persistent 2D mesh fabric. Each solution_run applies exactly ONE operation
(INJECT_PACKET, SET_LINK, DROP_PACKET, or TICK) and emits the updated counters
and the canonical hashes of the fabric state and the per-step event stream.

GEOMETRY
  node_id = row * cols + col.
  Ports: LOCAL=0, N=1, E=2, S=3, W=4. Opposite: N<->S, E<->W, LOCAL->LOCAL.
  A directed output (node, port) is "valid" iff it is not LOCAL and the
  neighbor in that direction is inside the mesh. N decreases row, S increases
  row, E increases col, W decreases col.
  escape_vc = 0; adaptive VCs are 1..vc_count-1.

PERSISTENT STATE
  cycle=0, event_seq=0, inject_seq_next=1.
  link_up[node][port] (u8) for valid directed outputs, init 1. Invalid outputs
    and LOCAL are always treated as down/absent.
  Input buffers: for every (node, input_port, vc) a FIFO of flits, capacity
    buffer_cap_per_vc.
  Flit fields: packet_id, flit_index, flit_count, src, dst, vc, kind,
    inject_seq, entered_cycle, credit_return_node (UINT32_MAX = none),
    credit_return_port (255 = none).
  credit[node][port][vc] init buffer_cap_per_vc, for every directed output.
  Credit-return queue: records (due_cycle, node, port, vc, amount,
    event_seq_created); canonical order = due_cycle, node, port, vc,
    event_seq_created.
  Packet table keyed by packet_id: src, dst, flit_count, prefer_adaptive,
    injected_flits, delivered_flits, inject_seq, state, chosen_vc.
  Injection queue per node: packet ids ordered by inject_seq (append order,
    since inject_seq strictly increases).

OPERATIONS  (run.op selects)
  INJECT_PACKET(packet_id, src, dst, flit_count, prefer_adaptive)
    Invalid if src/dst out of range, flit_count==0 or > WMC_MAX_FLIT_COUNT,
    or packet_id already exists in a nonterminal (QUEUED/ACTIVE) state.
    Else if packet table full OR source injection queue full -> INJECT_REJECT.
    Else create packet (inject_seq=inject_seq_next++, state=QUEUED,
    chosen_vc=UINT32_MAX), append to source injection queue, PACKET_ACCEPT.
    "Packet table full" means: a free slot is needed for a new packet id; a
    slot is free if it has never been used or holds a terminal (DONE/DROPPED)
    packet whose id differs. A re-used terminal id is replaced in place.
  SET_LINK(node, port, up)
    Invalid if node out of range, port==LOCAL, or the directed output leaves
    the mesh boundary. Else set link_up[node][port]=up, emit LINK_SET. The
    reverse direction is unaffected.
  DROP_PACKET(packet_id)
    Invalid if absent or already terminal (DONE/DROPPED). Else remove all
    not-yet-injected references of this id from every injection queue, mark
    packet DROPPED, emit PACKET_DROP. In-buffer flits drain/discard later.
  TICK(cycle_count)
    Invalid if cycle_count==0 or > a sane bound (WMC ticks one cycle at a
    time, repeated cycle_count times). Each cycle runs phases A,B,C,D then
    increments cycle. See below.

TICK CYCLE PHASES
  A apply due credits: for every credit-return record with due_cycle<=cycle,
    in canonical queue order, add amount to credit[node][port][vc]. If it would
    exceed buffer_cap_per_vc, clamp and emit CREDIT_CLAMP, else CREDIT_RETURN.
    Remove applied records.
  B eject / drop local-front: for node asc, input_port asc, vc asc, look at
    the single front flit:
      - if its packet is DROPPED: pop, schedule credit return, DROPPED_FLIT.
      - else if dst==node: pop, schedule credit return, ++delivered_flits,
        FLIT_EJECT; if delivered_flits==flit_count mark DONE and PACKET_DONE.
    At most one front flit per buffer in this phase.
  C inject: for node asc, head packet p of injection queue:
      - none -> continue.
      - DROPPED -> remove from queue, continue.
      - chosen_vc unset -> choose once: if prefer_adaptive!=0 pick the adaptive
        vc (1..vc-1) with the largest total usable minimal-route outgoing
        credit from src toward dst, tie lowest vc; if all adaptive candidates
        have zero usable credit pick escape_vc. If prefer_adaptive==0 pick
        escape_vc. A "usable adaptive credit" = credit on an up minimal output
        port toward dst for that vc, summed over those ports.
      - if (node,LOCAL,chosen_vc) buffer full -> INJECT_STALL, continue.
      - else create flit index injected_flits, kind by position, append to
        (node,LOCAL,chosen_vc), ++injected_flits, FLIT_INJECT; first flit sets
        state ACTIVE; if all injected remove id from injection queue.
        Injected LOCAL flits have credit_return_node=UINT32_MAX (no upstream).
  D switch traversal: for node asc, input_port asc, vc asc, front flit:
      - none -> continue.
      - DROPPED -> pop, schedule credit return, DROPPED_FLIT, continue.
      - dst==node -> EJECT_WAIT, continue.
      - compute output port:
          vc==escape_vc: XY. Horizontal first: E if dst col >, W if <; else
            vertical: S if dst row >, N if <. If required escape link down ->
            ROUTE_STALL.
          vc!=escape_vc: minimal ports reducing Manhattan distance whose link
            is up; pick largest credit[node][port][vc], tie lowest port. None
            -> ROUTE_STALL.
      - if that output was already used earlier this Phase D at this router ->
        OUTPUT_STALL.
      - if credit[node][output][vc]==0 -> CREDIT_STALL.
      - else if downstream (neighbor,opposite,vc) buffer is full despite
        positive credit -> CREDIT_DESYNC, no move.
      - else move: --credit[node][output][vc]; pop from input and schedule
        credit return for the popped buffer; append to downstream
        (neighbor,opposite,vc); entered_cycle=cycle; FLIT_MOVE. Escape vc move
        counts flits_moved_escape, else flits_moved_adaptive.
    After Phase D, ++cycle.

CREDIT SCHEDULING
  Whenever a flit leaves an input buffer (pop in B, C-drop, or D), if its
  credit_return_node != UINT32_MAX schedule one credit-return record:
    due_cycle = cycle + credit_latency, node = credit_return_node,
    port = credit_return_port, vc = flit.vc, amount = 1,
    event_seq_created = (current event_seq at insertion). Records are kept in
    canonical order. If the credit-return queue is full the record is dropped
    silently (deterministic; bounded by max_credit_events).
  A flit arriving from a link sets credit_return_node = upstream node and
  credit_return_port = upstream OUTPUT port used. LOCAL-injected flits carry
  UINT32_MAX/255 and schedule no upstream credit.

OUTPUTS (per run, after applying the op)
  Counters: packets_accepted, inject_rejected, links_set, packets_dropped,
  flits_injected, flits_moved_escape, flits_moved_adaptive, flits_ejected,
  packets_done, dropped_flits, credit_returns, credit_clamps, inject_stalls,
  route_stalls, output_stalls, credit_stalls, eject_waits, credit_desyncs,
  invalid_count.  These are CUMULATIVE persistent totals.
  fabric_event_hash: FNV-1a-64 over the events emitted THIS run, in order, each
    contributing event_kind:u8, event_seq:u64, cycle:u64, op_index:u32,
    node_or_UINT32_MAX:u32, port_or_255:u8, vc_or_255:u8,
    packet_or_UINT64_MAX:u64, flit_index_or_UINT64_MAX:u64, aux_u64.
    op_index = run.op_index (a per-run identifier supplied by the driver).
  packet_hash: nonterminal packets by packet_id asc.
  buffer_hash: buffers by node, input_port, vc; FIFO order within.
  credit_hash: directed credits by node, output_port, vc.
  credit_queue_hash: pending credit returns by canonical order.
  cycle_out, event_seq_out: persistent cycle and event_seq after the run.

DETERMINISM
  Phase order fixed: A,B,C,D, then cycle++. Ejection before traversal. Adaptive
  vc chosen once per packet at first injection. Escape vc always XY. A router
  output is used by at most one flit per cycle; earlier scan wins. Credits are
  authoritative; positive credit with full downstream buffer -> CREDIT_DESYNC.
  Directed link failures are one-way. cycle, inject_seq_next, event_seq wrap
  mod 2^64.
*/

// === DETERMINISM & EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section fully specifies, with NO deferral to any other file, every bit
// that enters every graded checksum. A solver that reads ONLY this header can
// reproduce all seven hashes and all 19 counters bit-for-bit. Everything below
// is the authoritative contract (it matches the graded reference exactly).
//
// ---------------------------------------------------------------------------
// 0. FNV-1a-64 PRIMITIVE
// ---------------------------------------------------------------------------
//   offset basis = 1469598103934665603  (0x14650FB0739D0383)
//   prime        = 1099511628211        (0x00000100000001B3)
//   Per byte b:   h ^= (uint64_t)b;  h *= prime;   (h is uint64_t, wraps 2^64)
//   To "fold" a field of W bytes, fold its W bytes in increasing memory-address
//   order. All multi-byte scalars below are folded in NATIVE LITTLE-ENDIAN byte
//   order (target is x86-64 / sm_120, little-endian): a u32 value V folds bytes
//   V&0xff, (V>>8)&0xff, (V>>16)&0xff, (V>>24)&0xff; a u64 folds its 8 bytes
//   least-significant first; a u8 folds its single byte. The fold is a raw
//   memcpy of the in-memory object representation; there are NO separators, NO
//   length prefixes, and NO terminators anywhere.
//
//   Field WIDTHS used below are EXACT and load-bearing:
//     u8  = 1 byte, u32 = 4 bytes, u64 = 8 bytes.
//   Signed counters that are folded (only credit values) are first cast to
//   uint64_t then folded as u64.
//
// ---------------------------------------------------------------------------
// 1. PERSISTENT vs RESEEDED HASH STATE
// ---------------------------------------------------------------------------
//   event_seq      : PERSISTENT across all runs and all cycles. Starts at 0.
//                    Incremented by exactly 1 AFTER each event is folded into
//                    the event hash (the value folded is the PRE-increment seq).
//   cycle          : PERSISTENT. Starts at 0. Incremented by 1 at the END of
//                    each TICK cycle (after phase D).
//   inject_seq_next: PERSISTENT. Starts at 1.
//   fabric_event_hash : RESEEDED to the offset basis at the START of EVERY run
//                    (before the op executes). It therefore covers ONLY the
//                    events emitted by THIS run, in emission order. event_seq
//                    and cycle folded into events are the persistent running
//                    values, NOT reseeded.
//   packet_hash, buffer_hash, credit_hash, credit_queue_hash : each RESEEDED to
//                    the offset basis at the start of ITS OWN computation, which
//                    runs AFTER the op has fully executed, over the resulting
//                    persistent state.
//   There is NO master combined "state_checksum": the seven outputs
//   (fabric_event_hash, packet_hash, buffer_hash, credit_hash,
//   credit_queue_hash, cycle_out, event_seq_out) plus the 19 counters are each
//   graded independently. cycle_out and event_seq_out are the raw persistent
//   u64 values after the run (NOT hashed). The 19 counters are the raw
//   cumulative int64 totals (NOT hashed).
//
// ---------------------------------------------------------------------------
// 2. EVENT HASH FOLD ORDER (fabric_event_hash)
// ---------------------------------------------------------------------------
//   Each emitted event folds EXACTLY these ten fields, in THIS order:
//     (1) event_kind                : u8   (WMC_EV_* ordinal)
//     (2) event_seq                 : u64  (persistent value BEFORE increment)
//     (3) cycle                     : u64  (persistent value at emission)
//     (4) op_index                  : u32  (= run.op_index, verbatim, as u32)
//     (5) node_or_UINT32_MAX        : u32
//     (6) port_or_255               : u8
//     (7) vc_or_255                 : u8
//     (8) packet_or_UINT64_MAX      : u64
//     (9) flit_index_or_UINT64_MAX  : u64
//    (10) aux                       : u64
//   Then event_seq += 1. Events are folded in the exact order they are emitted
//   (the phase/scan order defined in sections 4-5).
//
//   EXACT per-event argument values (kind, node_or, port_or, vc_or, packet_or,
//   flit_index_or, aux). "node n", "port p", "vc v", "pid", "fidx" are the
//   live values; UINT32_MAX, 255, UINT64_MAX are the "none" sentinels:
//
//   INJECT op:
//     INVALID (out-of-range src/dst, flit_count<=0 or >WMC_MAX_FLIT_COUNT, or
//        pid exists in QUEUED/ACTIVE state):
//        kind=WMC_EV_INVALID, node=UINT32_MAX, port=255, vc=255,
//        packet=pid, flit_index=UINT64_MAX, aux=0.
//     INJECT_REJECT (table full OR src injection queue full):
//        kind=WMC_EV_INJECT_REJECT, node=src(u32), port=255, vc=255,
//        packet=pid, flit_index=UINT64_MAX, aux=0.
//     PACKET_ACCEPT:
//        kind=WMC_EV_PACKET_ACCEPT, node=src(u32), port=255, vc=255,
//        packet=pid, flit_index=UINT64_MAX, aux=dst(u64).
//   SET_LINK op:
//     INVALID (node out of range, port==LOCAL or out of [0,WMC_PORT_COUNT), or
//        directed output leaves mesh):
//        kind=WMC_EV_INVALID, node=(node>=0 ? node : UINT32_MAX)(u32),
//        port=(u8)port, vc=255, packet=UINT64_MAX, flit_index=UINT64_MAX, aux=0.
//     LINK_SET:
//        kind=WMC_EV_LINK_SET, node=node(u32), port=(u8)port, vc=255,
//        packet=UINT64_MAX, flit_index=UINT64_MAX, aux=(up!=0 ? 1 : 0)(u64).
//   DROP op:
//     INVALID (absent OR already DONE/DROPPED):
//        kind=WMC_EV_INVALID, node=UINT32_MAX, port=255, vc=255,
//        packet=pid, flit_index=UINT64_MAX, aux=0.
//     PACKET_DROP:
//        kind=WMC_EV_PACKET_DROP, node=UINT32_MAX, port=255, vc=255,
//        packet=pid, flit_index=UINT64_MAX, aux=0.
//   TICK op:
//     INVALID (cycle_count<=0):
//        kind=WMC_EV_INVALID, node=UINT32_MAX, port=255, vc=255,
//        packet=UINT64_MAX, flit_index=UINT64_MAX, aux=0.
//     (otherwise emits the per-cycle phase events below.)
//   Unknown op (defensive; validators reject it earlier):
//        kind=WMC_EV_INVALID, node=UINT32_MAX, port=255, vc=255,
//        packet=UINT64_MAX, flit_index=UINT64_MAX, aux=0.
//
//   Phase A (credit application), per applied record in canonical queue order:
//     CREDIT_CLAMP (credit+amount would exceed cap; clamped to cap):
//        kind=WMC_EV_CREDIT_CLAMP, node=rec.node(u32), port=(u8)rec.port,
//        vc=(u8)rec.vc, packet=UINT64_MAX, flit_index=UINT64_MAX,
//        aux=rec.amount(u64, always 1).
//     CREDIT_RETURN (normal apply):
//        kind=WMC_EV_CREDIT_RETURN, node=rec.node(u32), port=(u8)rec.port,
//        vc=(u8)rec.vc, packet=UINT64_MAX, flit_index=UINT64_MAX,
//        aux=rec.amount(u64).
//   Phase B (eject/drop local-front), per affected front flit:
//     DROPPED_FLIT (front flit's packet is DROPPED):
//        kind=WMC_EV_DROPPED_FLIT, node=n(u32), port=(u8)port, vc=(u8)v,
//        packet=f.packet_id, flit_index=f.flit_index, aux=0.
//     FLIT_EJECT (dst==node):
//        kind=WMC_EV_FLIT_EJECT, node=n(u32), port=(u8)port, vc=(u8)v,
//        packet=f.packet_id, flit_index=f.flit_index, aux=0.
//     PACKET_DONE (emitted immediately AFTER the FLIT_EJECT that completes the
//        packet, i.e. delivered_flits==flit_count):
//        kind=WMC_EV_PACKET_DONE, node=n(u32), port=255, vc=255,
//        packet=f.packet_id, flit_index=UINT64_MAX, aux=0.
//   Phase C (inject), per node head packet:
//     INJECT_STALL ((node,LOCAL,chosen_vc) buffer full):
//        kind=WMC_EV_INJECT_STALL, node=n(u32), port=WMC_PORT_LOCAL(=0),
//        vc=(u8)chosen_vc, packet=pid, flit_index=UINT64_MAX, aux=0.
//     FLIT_INJECT (flit created and pushed to (node,LOCAL,chosen_vc)):
//        kind=WMC_EV_FLIT_INJECT, node=n(u32), port=WMC_PORT_LOCAL,
//        vc=(u8)chosen_vc, packet=pid, flit_index=fidx(= injected_flits BEFORE
//        increment), aux=0.
//   Phase D (switch traversal), per front flit, in scan order:
//     DROPPED_FLIT (front flit's packet DROPPED):
//        kind=WMC_EV_DROPPED_FLIT, node=n(u32), port=(u8)port(=INPUT port),
//        vc=(u8)v, packet=f.packet_id, flit_index=f.flit_index, aux=0.
//     EJECT_WAIT (dst==node; not yet ejected this cycle):
//        kind=WMC_EV_EJECT_WAIT, node=n(u32), port=(u8)port(=INPUT port),
//        vc=(u8)v, packet=f.packet_id, flit_index=f.flit_index, aux=0.
//     ROUTE_STALL (no usable output: escape XY required link down / no outp;
//        OR adaptive has no up minimal port):
//        kind=WMC_EV_ROUTE_STALL, node=n(u32), port=(u8)port(=INPUT port),
//        vc=(u8)v, packet=f.packet_id, flit_index=f.flit_index, aux=0.
//     OUTPUT_STALL (chosen OUTPUT port already used by an earlier flit this
//        cycle at this router):
//        kind=WMC_EV_OUTPUT_STALL, node=n(u32), port=(u8)outp(=OUTPUT port),
//        vc=(u8)v, packet=f.packet_id, flit_index=f.flit_index, aux=0.
//     CREDIT_STALL (credit[n,outp,v]==0):
//        kind=WMC_EV_CREDIT_STALL, node=n(u32), port=(u8)outp(=OUTPUT port),
//        vc=(u8)v, packet=f.packet_id, flit_index=f.flit_index, aux=0.
//     CREDIT_DESYNC (positive credit but downstream (neighbor,opposite,v)
//        buffer full):
//        kind=WMC_EV_CREDIT_DESYNC, node=n(u32), port=(u8)outp(=OUTPUT port),
//        vc=(u8)v, packet=f.packet_id, flit_index=f.flit_index, aux=0.
//     FLIT_MOVE (flit moved across the link):
//        kind=WMC_EV_FLIT_MOVE, node=n(u32), port=(u8)outp(=OUTPUT port),
//        vc=(u8)v, packet=f.packet_id, flit_index=f.flit_index,
//        aux=neighbor_node(u64).
//   NOTE on port semantics: stall/wait/drop events that occur BEFORE an output
//   port is chosen report the INPUT port; OUTPUT_STALL / CREDIT_STALL /
//   CREDIT_DESYNC / FLIT_MOVE report the chosen OUTPUT port. Phase B/C events
//   report the buffer's own port (B: the input port scanned; C: WMC_PORT_LOCAL).
//
// ---------------------------------------------------------------------------
// 3. INDEXING & ITERATION ORDERS
// ---------------------------------------------------------------------------
//   node_id  = row * cols + col.            row = node / cols, col = node % cols.
//   nodes    = rows * cols.
//   Buffer index bidx(node,port,vc) = (node*WMC_PORT_COUNT + port)*vc_count + vc.
//   Link  index lidx(node,port)     = node*WMC_PORT_COUNT + port.
//   Port ordinals: LOCAL=0,N=1,E=2,S=3,W=4 (WMC_PORT_COUNT=5).
//   Buffers exist for EVERY (node, port in [0,5), vc in [0,vc_count)) including
//   LOCAL and invalid directed outputs; only the credit/link semantics differ.
//   Within a buffer, flit order is FIFO (front = oldest). pos counts from 0 at
//   the front.
//   "node asc, input_port asc, vc asc" means the triple loop:
//     for n in [0,nodes): for port in [0,WMC_PORT_COUNT): for v in [0,vc_count).
//
// ---------------------------------------------------------------------------
// 4. ROUTING / ARBITRATION TIE-BREAKS (all derived from the reference)
// ---------------------------------------------------------------------------
//   minimal_ports(node,dst): the set of minimal (Manhattan-reducing) output
//     ports, BUILT AND ITERATED in this fixed port-ordinal order:
//       if dst_row < row : N ; if dst_col > col : E ;
//       if dst_row > row : S ; if dst_col < col : W .
//     (i.e. candidates are appended N,E,S,W in that sequence; link state is NOT
//     considered when building the set, only when consuming it.)
//   Escape VC (vc==0) routing = strict XY, horizontal-first:
//       if dst_col > col -> E ; else if dst_col < col -> W ;
//       else if dst_row > row -> S ; else if dst_row < row -> N .
//     If the resulting port has no valid direction (outp<0) or its link is
//     down -> ROUTE_STALL.
//   Adaptive VC (vc>=1) routing: among minimal_ports whose link is UP, pick the
//     port with the LARGEST credit[node,port,vc]; tie-break = FIRST in the
//     minimal_ports build order above (which, given that order, is the lowest
//     port ordinal among tied N/E/S/W). Comparison uses strict ">" so the first
//     candidate encountered wins on ties. If no up minimal port -> ROUTE_STALL.
//   Adaptive VC SELECTION (choose_vc, done ONCE at first injection when
//     chosen_vc==UINT32_MAX):
//       if prefer_adaptive==0 -> chosen_vc = 0 (escape).
//       else: for v in 1..vc_count-1, total_v = sum over minimal_ports(src,dst)
//         whose link is up of credit[src,port,v]; pick v with the LARGEST
//         total_v, tie-break = LOWEST v (strict ">" while scanning v ascending).
//         If best total <= 0 (or no adaptive vc) -> chosen_vc = 0 (escape).
//   Output-port arbitration within a router in Phase D: each OUTPUT port may be
//     used by at most ONE flit per cycle; the FIRST flit in (input_port asc,
//     vc asc) scan order to claim it wins; later claimants get OUTPUT_STALL.
//
// ---------------------------------------------------------------------------
// 5. CREDIT SCHEDULING (exact record contents & queue order)
// ---------------------------------------------------------------------------
//   Whenever a flit LEAVES an input buffer (popped in Phase B eject/drop, Phase
//   C never pops the input it pushes, Phase D drop or move) AND its
//   credit_return_node != UINT32_MAX, schedule ONE credit-return record:
//       due_cycle        = cycle + credit_latency  (cycle = current value)
//       node             = flit.credit_return_node
//       port             = flit.credit_return_port
//       vc               = flit.vc
//       amount           = 1
//       seq_created      = event_seq AT INSERTION (the current persistent
//                          event_seq value at the moment of scheduling)
//   LOCAL-injected flits carry credit_return_node=UINT32_MAX, port=255 and
//   schedule NO record. A flit that MOVES across a link sets the MOVED copy's
//   credit_return_node = the sending node n and credit_return_port = the OUTPUT
//   port used (so the downstream buffer can later credit it back). NOTE: in
//   Phase D the credit return for the popped (upstream) flit is scheduled from
//   the flit's PRE-move return fields, then the moved copy's return fields are
//   overwritten for its new hop.
//   If the credit-return queue already holds max_credit_events records, the new
//   record is dropped SILENTLY (no event, no counter).
//   Canonical credit-queue order (used in Phase A application AND in
//   credit_queue_hash) sorts records ascending by, in priority:
//       due_cycle, then node, then port, then vc, then seq_created.
//   Phase A applies ALL records with due_cycle <= current cycle, in this
//   canonical order, then removes them; records with due_cycle > cycle remain.
//
// ---------------------------------------------------------------------------
// 6. SNAPSHOT HASHES (computed after the op, over persistent state)
// ---------------------------------------------------------------------------
//   packet_hash : reseed to offset basis. Iterate nonterminal packets (state
//     QUEUED or ACTIVE; used==true) in ascending packet_id order. For each,
//     fold in THIS order:
//       packet_id        : u64
//       src              : u32
//       dst              : u32
//       flit_count       : u64
//       prefer_adaptive  : u8
//       injected_flits   : u64
//       delivered_flits  : u64
//       inject_seq       : u64
//       state            : u8
//       chosen_vc        : u32   (UINT32_MAX when unset)
//     Terminal (DONE/DROPPED) and unused slots are SKIPPED entirely.
//
//   buffer_hash : reseed to offset basis. Iterate (node asc, port asc[0..5),
//     vc asc) and, within each buffer, FIFO front-to-back with pos=0,1,2,...
//     For each flit fold in THIS order:
//       node             : u32   (the buffer's node)
//       port             : u8    (the buffer's port)
//       vc               : u8    (the buffer's vc)
//       pos              : u64    (FIFO position, 0 at front)
//       packet_id        : u64
//       flit_index       : u64
//       flit_count       : u64
//       src              : u32
//       dst              : u32
//       kind             : u8    (WMC_KIND_*)
//       inject_seq       : u64
//       entered_cycle    : u64
//     Empty buffers contribute nothing. (credit_return_node/port and pad are
//     NOT folded.)
//
//   credit_hash : reseed to offset basis. Iterate (node asc, port asc[0..5),
//     vc asc) over ALL directed slots (every node/port/vc, including LOCAL and
//     boundary-invalid ports). For each fold in THIS order:
//       node             : u32
//       port             : u8
//       vc               : u8
//       credit           : u64   (the int64 credit value cast to uint64)
//       link_up          : u8    (link_up[node,port]; for LOCAL and invalid
//                                  outputs this is 0)
//
//   credit_queue_hash : reseed to offset basis. Sort the pending credit-return
//     records into canonical order (due_cycle,node,port,vc,seq_created ascending
//     -- same comparator as section 5). For each record fold in THIS order:
//       due_cycle        : u64
//       node             : u32
//       port             : u8
//       vc               : u8
//       amount           : u64
//     (seq_created is the tie-break ONLY; it is NOT folded into the hash.)
//
// ---------------------------------------------------------------------------
// 7. COUNTERS (19, cumulative int64, index order -- folded NOWHERE, emitted raw)
// ---------------------------------------------------------------------------
//   [ 0] packets_accepted     [ 1] inject_rejected     [ 2] links_set
//   [ 3] packets_dropped      [ 4] flits_injected      [ 5] flits_moved_escape
//   [ 6] flits_moved_adaptive [ 7] flits_ejected       [ 8] packets_done
//   [ 9] dropped_flits        [10] credit_returns      [11] credit_clamps
//   [12] inject_stalls        [13] route_stalls        [14] output_stalls
//   [15] credit_stalls        [16] eject_waits         [17] credit_desyncs
//   [18] invalid_count
//   flits_moved_escape counts vc==0 moves; flits_moved_adaptive counts vc>=1
//   moves. Each counter increments exactly once at the corresponding event.
//
// === END DETERMINISM & EXACT OUTPUT SERIALIZATION (normative) ===

struct alignas(8) WmcProblemSpec {
    int32_t abi_version;
    int32_t rows;
    int32_t cols;
    int32_t vc_count;
    int32_t buffer_cap_per_vc;
    int32_t credit_latency;
    int32_t max_packets;
    int32_t max_injection_queue_per_node;
    int32_t max_credit_events;
    int32_t reserved[7];
};

struct alignas(8) WmcRunSpec {
    int32_t abi_version;
    int32_t op;              // WMC_OP_*
    int32_t op_index;        // per-run identifier folded into event hash
    // INJECT: a0=packet_id, a1=src, a2=dst, a3=flit_count, a4=prefer_adaptive
    // SET_LINK: a0=node, a1=port, a2=up
    // DROP: a0=packet_id
    // TICK: a0=cycle_count
    int32_t a0;
    int32_t a1;
    int32_t a2;
    int32_t a3;
    int32_t a4;
    int32_t reserved[8];
};

// No bulk input arrays are needed; all operands ride in the run spec. The
// inputs pointer is reserved (may be null) for ABI symmetry.
struct alignas(8) WmcInputs {
    const int32_t* reserved0;
    const int32_t* reserved1;
};

struct alignas(8) WmcOutputs {
    // 19 cumulative counters, contiguous.
    int64_t* counters;          // length 19
    uint64_t* fabric_event_hash;
    uint64_t* packet_hash;
    uint64_t* buffer_hash;
    uint64_t* credit_hash;
    uint64_t* credit_queue_hash;
    uint64_t* cycle_out;
    uint64_t* event_seq_out;
};

#define WMC_COUNTER_COUNT 19

static_assert(sizeof(WmcProblemSpec) == 64, "WmcProblemSpec layout drift");
static_assert(sizeof(WmcRunSpec) == 64, "WmcRunSpec layout drift");
static_assert(sizeof(WmcInputs) == 16, "WmcInputs layout drift");
static_assert(sizeof(WmcOutputs) == 64, "WmcOutputs layout drift");

static inline int wmc_validate_problem_spec(const WmcProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != WMC_ABI_VERSION) return 0;
    if (spec->rows < WMC_MIN_ROWS || spec->rows > WMC_MAX_ROWS) return 0;
    if (spec->cols < WMC_MIN_COLS || spec->cols > WMC_MAX_COLS) return 0;
    if (spec->vc_count < WMC_MIN_VC || spec->vc_count > WMC_MAX_VC) return 0;
    if (spec->buffer_cap_per_vc < WMC_MIN_BUFCAP ||
        spec->buffer_cap_per_vc > WMC_MAX_BUFCAP) return 0;
    if (spec->credit_latency < 0 || spec->credit_latency > WMC_MAX_CREDIT_LATENCY) return 0;
    if (spec->max_packets < 1 || spec->max_packets > WMC_MAX_PACKETS) return 0;
    if (spec->max_injection_queue_per_node < 1 ||
        spec->max_injection_queue_per_node > WMC_MAX_INJQ) return 0;
    if (spec->max_credit_events < 1 ||
        spec->max_credit_events > WMC_MAX_CREDIT_EVENTS) return 0;
    return 1;
}

static inline int wmc_validate_run_spec(const WmcRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != WMC_ABI_VERSION) return 0;
    if (run->op < WMC_OP_INJECT || run->op > WMC_OP_TICK) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const WmcProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const WmcProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const WmcRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // WORMHOLE_MESH_CREDITS_COMMON_H_
