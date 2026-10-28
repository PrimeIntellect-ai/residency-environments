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
  field with no actuator; the runner samples it like any other port.

## What a world declares

| Item | Why (evidence) |
|---|---|
| Fields in port order, grid, `dx` (world units per pixel) | Device geometry and source width are in world units. 1 px per world unit suited all three worlds. |
| Native dt and native steps per public tick | The runner checks only the public tick. phase_membrane: 1 step per tick (1 tu = 50 native t); vortices: 1 step per tick (1 tu = 1 native t). |
| Noise form and level per field | The form decides survival. Plain additive noise on fields the world clips at zero is a hidden source: it dissolved phase_membrane at sd 0.01 per sqrt(t), and filled the Marangoni bath with cargo. Proportional noise on concentrations, additive on order parameters and vorticity, conserved on conserved fields, kept each world intact up to about 0.01-0.03. |
| Ports: observed and driven separately; for each driven port an actuator kind, a gain and a clip | PhySim's maximum program (amp 3 for 50 tu) made the numerics fail at gain 1 through phase_membrane's substrate, catalyst and phase field, the vortices' vorticity, and the rods' surfactant and phase field. Gains of 0.01 (and 0.1 for fuel) kept every port finite and in range. Velocity is recomputed every step, so injecting into it does nothing: observe-only. |
| Signed actuators where the field is signed | Vorticity is signed; a non-negative pulse only adds one sign (it kills or flips the other). Offer a signed amplitude or a +/- port pair. |
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
- Ports: separate observed and driven maps (today an undriven port is a silent no-op that the agent
  still sees); signed amplitudes; per-port actuator kinds.
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

## Open questions

- Whether to keep raw Python steppers or widen Blobkit's genome into a term list for field worlds
  (both run through the same seam).
- Trusted-code policy for world files in the grader.
- Cost: the prototype runs about 3.4 ms per native step at 128^2; a 50 tu window at one step per tick
  is about 8.5 s per member.
