# Lab worlds as PhySim evaluations: draft world contract

Status: draft, 28 Oct 2026. Evidence: the eval-readiness battery on three collective worlds
(phase_membrane, c53 fed vortices, c34 Marangoni rods) and a prototype adapter,
`physim/labworld.py`, on the local branch `feat/labkit-worlds` of residency-environments.

## What already works without changing the runner

PhySim's `OracleRunner` takes any stepper `stepper(sim, n, injections)` through its private
`_stepper` argument. The prototype wraps a lab world (a Python module with `init` and `step`
over a dict of square float arrays) as such a stepper and runs real experiment programs
through `sample_truth`, with PhySim's own devices, scheduling, member seeding and checks.

- **Public tick vs native time.** The runner's tick stays 0.02 public time units. The stepper
  takes a declared number of native steps per tick, so a world keeps its own dt.
- **Per-member state.** The runner copies only `F` and `rng` per member; the stepper keeps every
  other piece of module state in a per-member copy made on its first call.
- **Noise.** All draws come from the runner's per-member generator, a fixed number per step, so
  actions never shift a member's noise stream (paired controls stay paired).
- **Observe-only channels.** A derived field (velocity, eating rate) can ride in `F` as an extra
  field with no actuator; the runner samples it like any other port. `LabOracleRunner` (below)
  declares such channels and keeps them out of the inject roster.

## What a world declares

| Item | Why (evidence) |
|---|---|
| Fields in port order, grid, `dx` (world units per pixel) | Device geometry and source width are in world units. 1 px per world unit suited all three worlds. |
| Native dt and native steps per public tick | The runner checks only the public tick. phase_membrane: 1 step per tick (1 tu = 50 native t); vortices: 1 step per tick (1 tu = 1 native t). |
| Noise form and level per field | The form decides survival. Plain additive noise on fields the world clips at zero is a hidden source: it dissolved phase_membrane at sd 0.01 per sqrt(t), and filled the Marangoni bath with cargo. Proportional noise on concentrations, additive on order parameters and vorticity, conserved on conserved fields, kept each world intact up to about 0.01-0.03. |
| Ports: observed and driven separately; for each driven port an actuator kind, a gain and a clip | PhySim's maximum program (amp 3 for 50 tu) made the numerics fail at gain 1 through phase_membrane's substrate, catalyst and phase field, the vortices' vorticity, and the rods' surfactant and phase field. Gains of 0.01 (and 0.1 for fuel) kept every port finite and in range. Velocity is recomputed every step, so injecting into it does nothing: observe-only. |
| Signed actuators where the field is signed | Vorticity is signed; a non-negative pulse only adds one sign (it kills or flips the other). PhySim's grammar has no signed amplitude, so offer a +/- port pair (`Actuator(field, +1)`, `Actuator(field, -1)`) or one port of the needed sign. |
| Score groups and aggregate sensors | Chaotic worlds (vortices, rods) decorrelate within 50-200 t. Late in a window, whole-box statistics carry the mechanism (the rods' wall ablation separates only through cargo variance at 150 t). |
| Privileged ablations, each prepared on its own | All battery ablations were switched on from the native prepared state; a hidden ablated world starts from its own prepared state, which is the fair score-meaning check. |
| Prepared-state recipe (seed, time, overrides) | Identity must include parameter overrides, which several worlds read from the environment at import. |

## Acceptance checks before a world becomes an evaluation

1. **Parity.** At noise 0 and no actions, the runner's output equals the native module's run.
2. **Noise band.** Capability measures against noise level, for the declared form, on 2 or more seeds;
   the chosen level sits inside the band where the behaviour survives.
3. **Envelope.** Every driven port at PhySim's maximum program (amp 3, 50 tu, at the worst site)
   stays finite and inside the field's physical range.
4. **Spread.** Forecast members differ; native-vs-native scores set the null band.
5. **Detectability.** Each privileged ablation separates from the native world beyond the null band,
   in the declared score groups and windows.
6. **Horizon.** Measure the decorrelation time; choose windows and late-window queries (statistics)
   accordingly. Long windows help slow behaviour (divisions), but pointwise forecasts of chaotic
   worlds stop paying after 30-150 t.

## Runner and bundle changes still needed

- Make `_stepper` a declared, versioned interface; keep today's Blobkit path frozen as a legacy
  profile so existing bundles reproduce bit for bit.
- Bundle layer: `_validate_physics`, `make_oracle`, `NUMERICS` and the single 256^2 float32
  `fields` array assume a Blobkit genome. A lab bundle carries the world file, its overrides, the
  declaration above and a prepared state (named arrays, including module state), all hashed into
  its identity.
- Ports: observed and driven maps are now separate in the runner, with signed actuators
  (`LabOracleRunner`, below). Still open: actuator kinds other than a Gaussian source, and the
  scoring roster. `blobround6_eval.PublicRoster.n_ports` is both the inject-port bound
  (`validate_case` parses actions against it) and the observed-port count of every sample array,
  so `score_case` rejects valid requests when a world has more actuators than observed ports, and
  accepts ports the runner rejects when it has fewer. Separating the two needs a scoring change
  (that file's hash is in every suite's identity) or a lab-side roster.
- Ablations: hooks in the world declaration, replacing the template-array edits in
  `build_evaluation_bundle.py`.
- Agent spec: port counts and sensor descriptions come from the declaration.

## Prototype results through the runner (28 Oct)

`physim/labworld.py` runs phase_membrane and c53's fed vortices through `OracleRunner.sample_truth`
with PhySim's own devices (13-node square and 19-node hex arrays, the global sensor), scheduling and
per-member seeding. Scripts: `runner_demo.py`, `vortex_demo.py` (scratchpad `evalpilot/`).

- **Parity.** At zero noise with no actions the runner's output equals the native module exactly
  (phase_membrane, 100 steps: max difference 0.0; members identical).
- **Envelope** (amp 3 for 50 tu at device 0, the maximum PhySim program):
  - phase_membrane, 1 tick = 1 native step: gain 1 goes non-finite through S, K and phi; gain 0.1 is
    finite but leaves the physical range (S 2.1, K 1.19); gain 0.01 is finite and in range on all
    four ports.
  - fed vortices, 1 tick = 1 native step: vorticity goes non-finite at gain 1, reaches 1.5 (about 8x
    a body's peak) at 0.1 and a sane 0.33 at 0.01; fuel is safe at every gain up to 1.
- **Scores.** 8 forecast members vs 8 truth members per arm, readouts every 1 tu for 50 tu, energy
  score per sensor group with one pooled scale per field; ratio = score against the ablated world /
  score against an independent native truth. Every ablated arm starts from the native prepared state.

| World (noise) | Arm | device0 | device1 | global |
|---|---|---|---|---|
| phase_membrane (0.01, physical form) | link drift off (chi=0) | 7.8-17.6 | 5.7-18.6 | 13-177 |
| | uniform mobility (gate=0) | 6.4-26.9 | 5.8-15.9 | 32-246 |
| | tilt halved (a=2) | 7.7-27.3 | 5.8-17.3 | 10-96 |
| fed vortices (0.003, vorticity additive, fuel proportional) | flow off (adv=0) | 5.0-17.3 | 7.0-21.2 | 0.75-1.14 |
| | fuel regrowth off (kF=0) | 7.4-14.9 | 5.1-15.9 | 8-27 |
| | dt halved (numerical null) | 0.96-1.16 | 0.98-1.00 | 0.66-0.87 |

  Ranges span the 10, 25 and 50 tu windows and the sham and pulse programs. The global sensor
  cannot see the flow-off arm in the vortices (box means and variances are unchanged); the probes can.
  The vortex regrowth-off arm (a dynamics ablation of E) was missing from the battery and is detectable.
- **A bug the numerical null caught.** The first vortex run scored the dt-halved null at 5-12x: the
  world precomputes its diffusion operators from dt at init, and the adapter carried them over from
  the prepared state. The adapter now rebuilds non-field state with the arm's own `init` (keeping
  declared `carry` keys such as the clock). A numerical null arm belongs in every acceptance run.
- **Caveats.** One prepared state per world; ablations branched from the native state; the pooled
  per-field scales are a stand-in for PhySim's declared score groups; native-vs-native scores in the
  phase_membrane global group are small, which inflates its ratios.

## Ports interface: `LabOracleRunner` (28 Oct)

`physim/labworld.py` keeps the agent's inject roster apart from the observed ports. It does not
touch `blobround6.py`, `devices.py` or `bundles.py`, whose source hashes are part of PhySim's
implementation identity.

**Declaration** (`LabWorld`, new keys default to the old behaviour):

| Key | Meaning |
|---|---|
| `fields` | Dynamical fields, the first rows of `F`, as before. |
| `derived=(Derived(name, fn), ...)` | Observe-only channels: `fn(module, state)` on the module state at t=0 and after every runner advance (every tick at which anything is read). They ride in `F` after the fields, so the probes and the global sensor read them. They must not modify the state or draw random numbers. |
| `observed` (property) | `fields + derived`: what every sensor returns, in port order. |
| `actuators=(Actuator(field, sign), ...)` | The public inject roster: an inject on port `p` is a source of `sign * gain * amp` (the field's `Port` gain and clip) on `actuators[p].field`. Each actuator names a dynamical field with a declared `Port`; a derived channel, an unknown field, a field without a gain, a sign other than +1/-1 or a duplicate is a `ValueError` at declaration. |
| `actuators=None` | OracleRunner's roster: port `p` drives the field observed on port `p` with sign +1 (over the fields only, when derived channels exist). A field without a `Port` is still a silent no-op here, as before; an explicit roster refuses it. |
| `declaration()` | JSON-ready declaration; omits `actuators` and `derived` when undeclared, so suites frozen before them (labeval's) are unchanged. |

A signed field is offered as a +/- pair, or as a single port of either sign. One bipolar port is not
expressible within PhySim's grammar: `_parse` fixes the inject keys and requires `amp` in [0, 3].

**Runner.** `LabOracleRunner(OracleRunner)` takes the base constructor's arguments plus
`_actuators` (a sequence of `(F row, sign)`) and `_observe_only` (F rows that no actuator may name);
`LabOracleRunner.wrap(runner, actuators=None, observe_only=())` rebinds an existing OracleRunner
(for example a native Blobkit binding). `oracle(world, state, devices)` builds one from a
declaration. Requests are parsed with `inject.port` in [0, `n_actuators`); every sample array has
`n_observed` ports. A pulse reaches the stepper as `field=row, amp=sign * amp`, so `LabStepper`
and the native `step_chunk` (through `genome.poke`) both apply negative sources unchanged.

Reused from blobround6 unchanged: the constructor's checks, `_parse`, `_effective_actions`,
`_truth_member_seed`, `_clone_sim`, `_adjust_pose`, `_advance`, `_fill_queries` and `_sample`.
Copied: the body of `OracleRunner.sample_truth` (its scheduler, about 35 lines), with two changes:
the action parse uses the actuator count, and a pulse's field and amplitude come from its actuator.
With the default roster both are the base class's values, which the checks below confirm bit for bit.

**Checks** (`scripts/physim/labworlds/ports_check.py`; logs in scratchpad `evalpilot/ports/`):

| Check | Setup | Result |
|---|---|---|
| Identity, lab world | phase_membrane as in `runner_demo.py` (noise 0.01, gains 0.01): the pre-extension OracleRunner on LabStepper vs `oracle()` vs `LabOracleRunner.wrap`; the demo's pulseK program (2 members, 50 tu, every 1 tu) and a 20 tu program with pulses on all ports, moves, equal-time actions, a zero-amplitude and a one-tick pulse, ports permuted `[2, 0, 3, 1]` and a non-trivial adjustment map (3 members) | Bit-identical (byte comparison) in both; members differ; equal to the demo's saved forecast members; the same `ProtocolError` messages for seven invalid requests |
| Identity, native Blobkit | Packaged BF genome, `init_soup` (seed 7, noise 0.002), `devices.step_chunk`, BF's port permutation, the same 20 tu program (2 members) | Bit-identical (OracleRunner, `wrap`, direct construction). `blobround6._native_oracle` cannot run here (`physim.blobcore` is not in this package). The script's own native check is a separate `step_chunk` binding on a fresh `init_soup`. A reviewer also ran the offline gt_bf bundle (`make_oracle` on the prepared BF state, case c002, seed 71001, 2 members, 50 tu): `LabOracleRunner.wrap` is bit-identical to `OracleRunner` |
| Signed vorticity | Fed vortices as in `vortex_demo.py` (noise 0.003, om gain 0.01), roster (om,+), (om,-), (Y,+); amp 3 for 50 tu at each device centre, readouts every 1 tu | Finite throughout. First tick at the centre: +5.7e-4 and -5.7e-4 (ratio -1.000003; expected 6e-4). The '-' pulse turns the device0 body from +0.19 to -0.37 at 25 tu and -0.32 at 50 tu (+ pulse: 0.34, 0.30); probe range -0.37 to 0.20; global mean om -0.0017 (device0) and -0.0021 (device1) below sham at 50 tu. (om,+) equals the default roster's port 0 bit for bit and reproduces the earlier envelope run |
| Observe-only speed | `Derived("speed", adv * hypot(uy, ux))` from the module's own `_vel`, roster (om,+), (om,-), (Y,+) | Returned by both probes and the global sensor (3 observed ports); om and Y identical bit for bit to the world without it; equal to the speed recomputed from the state at t=0 and after 50 ticks. Rejected as a target: `Actuator("speed")` and runner row 2 raise `ValueError`; port 3 raises `ProtocolError` ("must be [0, 2]"); a world with observe-only rows must declare its actuator roster: the default roster is refused (`ValueError`), because inject port p would no longer name the field read on observed port p |

Native negative sources through `genome.poke` also work (BF activator 0, 2 tu at amp 3: global mean
+0.0040 for '+', -0.0056 for '-'); that run is a smoke check, not part of the identity proof.
Suites frozen by labeval before this change keep their declarations (checked for phase_membrane
and the fed vortices).

## Second round: suite scoring and two more worlds (28 Oct)

All four worlds (phase_membrane, c53 fed vortices, c34 Marangoni rods, c40 CIPS cells) run through
the runner with exact parity at zero noise. Controls are predictors from the same prepared state,
as in `build_evaluation_bundle.py`, scored against native truths; ratio = control / independent native.

- **Suite** (`scripts/physim/labworlds/labeval.py`, PhySim's `score_case`, frozen suite of 10-13
  programs, 15 query times to 50 tu, score groups per sensor and port with scales frozen from a
  calibration ensemble before any truth, 2 truths per case, 8-member controls). Aggregate ratios:
  - phase_membrane: ignore actions 2.3, wrong probe geometry 1.8, initial persistence 7.2,
    chi=0 7.6, gate=0 14.8, a=2 7.5, dt-halved null 0.87 (a second native ensemble scores 0.90
    against the same 8-member reference, so the reference is noisy; early in the window the null's
    global phi group is 1.5).
  - fed vortices: ignore actions 6.5, wrong probe geometry 9.9, initial persistence 15.9, adv=0
    10.8, kF=0 113, null 0.98.
  - Not separated: ignore actions on a 1-unit probe move (phase_membrane), wrong probe geometry on
    the global groups (by construction), adv=0 in the first 10 tu on the global groups.
- **Marangoni rods** (1 native step of dt 0.05 per tick; 19 to 23-26 bodies per 50 tu window):
  mar=0, passive, chis=0 and the naive controls score 2-50x. **The dt-halved null fails
  (1.06-1.66)**: the world's first-order time stepping at dt 0.05 is not converged relative to the
  noise (the deterministic dt error is 12-74% of the member spread at 1-5 tu). Any control ratio
  below about 1.7 there cannot be told from a change of numerics (kappa=0 late on the probes, the
  cargo pulse's ignore-actions control).
- **CIPS cells** (192^2, 1 native step of dt 0.1 per tick, prepared at t=3600 with two divisions in
  the window): kF=0 491-823x, turnover off (d=0) 39-74x, chi=0 224-404x, chi_s=0 48-103x, naive
  controls 5-28x; the null scores like the reference overall (0.96-1.08, inside the two-ensemble
  band 0.78-1.08) but fails at the pulse site (device0 1.18-1.41): the solver's handling of the
  injected source is not converged at dt 0.1.

**Decision for the contract.** A world is either (a) a discrete map whose time step is part of its
definition, or (b) a continuum model whose numerics must pass the dt-halved null. Under (a) an
agent that integrates the right equations more accurately is penalized up to 1.3-1.7x (rods); under
(b) the rods need a smaller dt (about 2x cost) and the null re-tested.

## World definition (decided 29 Oct)

A world is a continuum model; a discretization is an approximation of it. A resolution (dt, grid)
is admissible for an evaluation when refining it (dt halved; grid refined at fixed physical size,
with devices and sources in world units) changes the evaluation's scores by no more than the
spread between two independent native ensembles. Coarser resolutions show the integrator, not the
world. Consequences: dynamics noise is defined in the continuum (a fixed correlation length in
world units, not white per pixel), and a world's claimed behaviour is re-checked at an admissible
resolution before it becomes an evaluation.

## Convergence study (29 Oct): Marangoni rods and CIPS cells

Correlated dynamics noise is now available (`LabWorld.noise_corr = {field: ell}`, a periodic
Gaussian of standard deviation `ell` world units, rescaled to the white form's per-point variance;
conserved noise filters the divergence; default bit-identical; commit b23a593). Both studies used
ell = 2 world units and sd 0.01.

| | Marangoni rods | CIPS cells |
|---|---|---|
| Time (dt-halved null, 18 cells, native band from pooled independent natives) | native dt 0.05 passes | native dt 0.1 passes (thin margin at the pulse site) |
| Space (grid refined at fixed box) | 128^2 fails; 192^2 converged for every field except cargo, which fails at every window and is not converged at 256^2 | fails: 288^2 and 384^2 not converged; the amphiphile layer is about a pixel thick and keeps sharpening |
| Claimed behaviour at finer grids (2 fresh seeds, no noise) | rods, rings and swim-then-divide survive; persistent splits 21/22 -> 16/18; cargo containment 11.9/11.0 -> 17.5/18.2 and still rising | splits, layers on every body and death under a local fuel cut survive; first division 320-940 t earlier; the diagonal division preference weakens (likely a grid artefact) |
| Cost per member per 50 tu | 192^2: about 113 s (2.3x native) | 288^2: 54 s; 384^2: about 93 s; converged: unknown |

Lessons:
- **The noise must be specified before the resolution.** At equal per-point variance, ell = 2 noise
  is several times stronger than white noise (member spread 3-12x early in the window), which is
  why the native dt now passes, and it costs sensitivity (the rods' ignore-actions control fell from
  2.0-2.4x to 1.2-1.4x). Specify noise by its intensity in world units (sd scaled with 1/ell in 2D)
  so that adding a correlation length does not also strengthen it, then find the admissible dt.
- **Qualitative claims survive refinement; some numbers do not.** Division, rings, swim-then-divide,
  layered cells and starvation death persist. Cargo containment ratios, division timing and division
  axes are discretization numbers at the native grids: the worlds were tuned at the edge of resolution.
- **Resolvable worlds make better evaluations.** Interfaces about a pixel thick cannot be converged
  affordably. Interface width (gradient-energy and diffusion coefficients) is a parameter of the
  continuum world; widening it until the behaviour converges at an affordable grid is legitimate,
  and a world whose behaviour needs unresolvable interfaces is a poor evaluation candidate.

## Acceptance versus curation (decided 29 Oct)

An evaluation is defined at one declared resolution and time step; its truths are that simulation.
No continuum limit has to be shown for scoring to be valid.

**Acceptance (required for a valid evaluation):** parity of the runner with the native module;
an amplitude envelope that keeps every driven port finite and in range; nonzero member spread;
every mechanism and naive control separating from an independent native ensemble beyond the
native-vs-native band; noise declared per field.

**Curation (which worlds we publish; not part of the evaluation):**
- the claimed behaviours persist, and their measures trend smoothly, under 1.5-2x refinement in
  space and time;
- no grid-direction effects (a 45-degree rotation of the initial state gives the same statistics);
- noise defined in world units (correlation length and intensity);
- the numerics penalty (the dt-halved and refined-grid predictors' score ratio) is small next to
  the smallest control separation the evaluation relies on.

A world whose claimed behaviour is a grid artefact rewards modelling the lattice instead of the
mechanism, so it fails curation even if it passes acceptance.

## Curation results (29 Oct)

Noise by intensity (`LabWorld.noise_intensity`, commit b4bdebc): sd is read as an intensity in world
units (white draws scaled by 1/dx, 0.5/dx^2 for conserved; a correlation length smooths without
changing the k=0 forcing), so changing ell or the grid no longer changes the large-scale noise.
Low-k spectra agree to 0.98-1.01 across ell and grids; existing declarations are bit-identical.
Control separation falls roughly as 1/intensity; the numerics penalty rises as noise falls.

| World | Refinement (a) | Isotropy (b) | Numerics penalty (d) | Noise candidate (ell = 2) | Verdict |
|---|---|---|---|---|---|
| fed vortices | splits, generations, same-sign daughters, speed consistent at 128^2/192^2/256^2 and dt/2 within chaotic spread | isotropic: division and motion axes within +-0.1 of no preference, rotated and refined alike | 1.02-1.07 against a smallest separation of 1.41 | intensity 0.003: behaviour holds, dt null 1.02, controls 11-18x; continued splitting under noise not yet tested long (splits after t=300 fall as noise rises) | passes; first evaluation candidate |
| phase_membrane | splits and generations show no trend; the link ring's rim/core contrast persists (+5% then level), but rim/outside contrast is about 40% low at 128^2 (9-11 against 13-19) | division and motion isotropic; elongated cells lean weakly toward the grid axes (10 of 10 runs positive at 128^2), weakening with refinement | 1.05 at 10 tu (global phi 1.37) | 0.03 is marginal: a second dt/2 null draw gives 1.04-1.08; 0.01 needs dt halved and a dt/4 null; nucleations appear at 0.03 | passes with flags |
| CIPS cells | (from the convergence study) division survives; layer peak and timing not converged | **diagonal division is a grid artefact**: the axis-only drift fluxes turn bodies toward the diagonals (planted ellipses rotate 15-25 degrees; rotated starts still split on diagonals); an isotropic drift stencil (2/3 axis + 1/3 diagonal fluxes) gives chance-level axes while division, a second round and the layer survive, later than with the standard stencil | not measured | not measured | fails (b) as written; passes with the isotropic stencil |

Also: device1's ignore-actions group never separates in either suite (a device0 pulse does not reach
device1 within 50 tu), so an evaluation must not rely on that group; and the suite's own numerical
null shared member seeds with independent_native, which overstated its precision (an independent
dt-halved ensemble confirms the result).

## Open questions

- Whether to keep raw Python steppers or widen Blobkit's genome into a term list for field worlds
  (both run through the same seam).
- Trusted-code policy for world files in the grader.
- Cost: the prototype runs about 3.4 ms per native step at 128^2; a 50 tu window at one step per tick
  is about 8.5 s per member.
