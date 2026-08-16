# 032: ik teacher continuity

## question

is the deterministic-ik target used in experiments 027--031 locally sensitive
to axis-aligned translational perturbations near held-out teacher commands?

## setup

- reuse all 230 first-step samples from held-out cache episodes 50--59.
- restore each cached native state in the pinned simulator and verify the
  reconstructed deterministic-ik target against the cached label.
- perturb canonical translation independently by plus and minus 1 mm along x, y,
  and z, recomputing ik from the same state.
- report maximum body-joint target change per perturbation and the fraction of
  samples where any 1 mm perturbation changes a body target by more than 5
  degrees.

classify translational sensitivity as material only if more than 5% of held-out
samples cross the 5-degree threshold. otherwise conclude only that these local
translation probes do not explain the broad decoder deficit; rotation,
state-restoration, and decoder-formulation hypotheses remain unresolved. perform
no training or closed-loop evaluation in this experiment.

## result

only 4/230 held-out samples (1.74%) exceeded a 5-degree target change under any
plus/minus 1 mm translation perturbation, below the pre-registered 5% threshold.
the median maximum body-joint change was 0.56 degrees, p95 was 2.10 degrees, and
the maximum was 7.40 degrees. reconstructed cached labels differed by more than
0.1 native unit for 65/230 samples, more than 0.5 for 3/230, and at most 1.03.
per-sample results and cache provenance are in
`reports/selected-core-exp32-ik-continuity.json`.

## finding

axis-aligned translation probes are below the pre-registered discontinuity
threshold for 98.3% of restored held-out states. this does not establish
rotation-channel, coupled-perturbation, or exact cached-label continuity, and
state restoration error limits the claim to the currently reconstructed solver.

## decision

- reject widespread translational discontinuity as the sole explanation.
- leave rotation sensitivity and the 1.03-unit reconstruction discrepancy open.
- treat decoder formulation as a candidate obstacle, not an established cause.
- stop decoder capacity, update-budget, and scalar loss-weight sweeps until a new
  formulation is specified from control geometry.
