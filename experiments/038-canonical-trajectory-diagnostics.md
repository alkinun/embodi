# 038: canonical trajectory diagnostics

## question

does execution horizon, policy-action geometry reconstruction, chunk inconsistency,
or policy-visited state shift explain the Experiment 037 closed-loop failure?

## setup

- freeze benchmark definition SHA-256
  `9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f`,
  development manifest SHA-256
  `4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576`,
  Experiment 036 summary SHA-256
  `76d7a3b0dc8954fc1fdbf8def99c840881ee3d9b523bf9f3386be5b82be0308f`,
  and Experiment 037 summary SHA-256
  `7efde348c16863399167794636d9941360a144b565919bd176f181cca1e15aa4`.
  perform no training, admission, checkpoint selection, or final-split evaluation.
- use exact development scenario indices 0000 through 0006 in every morphology
  by pretraining-task cell: SO-101 and Panda crossed with push, lift, and
  pick/place. these seven scenarios activate the seven factor axes once per
  cell. exclude stack as untrained and non-diagnostic. this gives 42 scenarios,
  21 for each execution morphology.
- compare only the joint core and matched specialists at seeds 36001--36003.
  matched means the SO specialist on SO execution and Panda specialist on Panda
  execution. do not run opposite specialists. evaluate horizons 1, 4, and 16.
  one condition by seed by morphology shard loads one checkpoint and runs all
  21 scenarios at all horizons: 12 shards, 63 episodes each, 756 total. registered
  reports are `benchmark-exp38-{condition}-seed{seed}-{morphology}-diagnostic.json`.
- retain the Experiment 037 top 640x480, supplied FK canonical state,
  `state=None`, regression, 32-command, 30 Hz, and maximum-500-step path. decode
  all 32 commands against the unchanged observation anchor before stepping.
  compare each H16 result with the same scenario and checkpoint in
  `benchmark-exp37-{source_condition}-seed{seed}-{morphology}-geometry.json`,
  where the source is `joint` or the execution morphology for matched. validate
  that expected and observed comparisons are truthfully encoded, then report
  exact outcome match rate, source and replay success rates, and the full steps,
  chunks, stage, failure, command, and clipping signature descriptively. this is
  a frozen-reference comparison, not an integrity or advancement gate.
- this descriptive comparison is a pre-completion protocol correction. no
  registered shard completed under either superseded exact-replay gate. probes
  first reached the same successful outcome with 481 steps/31 chunks and 467
  steps/30 chunks, while an isolated retry exactly matched 481/31. the immutable
  Experiment 037 loop then reran its frozen failed `so101/push_to_zone/0002`
  outcome as success at step 500. neither full trajectory nor boundary outcome
  is therefore stable enough to serve as a fail-closed replay invariant. run
  registered shards sequentially to reduce concurrent-CUDA variation, retain
  all comparison rates, and make no mechanism claim from replay mismatch alone.
- for every policy-generated command, decode at the unchanged anchor and
  re-encode with `canonical_arm_action`. accumulate translation norm, rotation
  geodesic, gripper error, and native clipping by lead. compare p95 and maxima
  descriptively with the frozen 1 mm, 1 degree, and 1-native references. the
  privileged expert admission does not cover policy-generated actions; geometry
  is only mechanism evidence if joint policy reconstruction p95 exceeds a
  reference. report joint and matched reconstruction separately so comparative
  attribution is retained, and aggregate native clipping by lead and condition.
- compose each chunk into absolute base-frame targets using anchor translation
  plus relative translation, anchor rotation times relative rotation, and the
  absolute gripper target. measure adjacent within-chunk jumps and cross-replan
  overlap `old[H+k]` versus `new[k]` in physical units. retain lead-specific
  count, sum, squared sum, and maximum records, not raw chunks. aggregate these
  records by condition and horizon while preserving lead. H32 is not run.
- instantiate one `PrivilegedBenchmarkExpert` at episode start. call its
  stateful `action` exactly once before every executed live policy action and
  never apply that action. on the first action of each replan, while the expert
  was nonterminal before its call, compare policy canonical action zero with
  `canonical_arm_action(expert_native)`. separate step-zero from policy-visited
  anchors and record phase counts and expert terminal step. this is a
  phase-stateful reference on the live trajectory, not a unique oracle action.
  do not clone or roll out a counterfactual environment and do not intervene
  expert-first.
- after each `step_control_period`, use its returned status and existing
  diagnostics to record first hitting steps and attainment for task-relevant
  milestones: EE-object distance below 0.06 m, object inside target tolerance,
  speed below 0.05, morphology-threshold gripper close/release, lift at 0.07 m,
  retained lift predicate at 0.08 m, and success. never call `task_status`
  separately.
- pair horizon outcomes by scenario, seed, morphology, and condition and use
  equal-weight seed means. call the horizon mechanism strong only when the
  H1-to-H16 three-seed mean paired success gain is at least 5 percentage points,
  its direction is nonnegative in at least two seeds, and the three-seed mean
  task-equal milestone progression improves. within each outcome, progression
  is the fraction of task-relevant milestones attained; average outcomes within
  task, then average the three tasks equally. otherwise classify the mechanism
  partial or unsupported. 5 points is the existing benchmark materiality
  margin, not a model-admission gate.
- distribution-shift evidence is descriptive. label a strong association only
  if the late-to-initial normalized physical first-action error ratio is at least
  two in at least two seeds. compute this on H16 only: within each condition and
  seed, pool initial replans and policy-visited replans separately across tasks
  and morphologies, calculate the sample-weighted mean of each physical error,
  normalize the three component means by their frozen references, then average
  the components and divide policy-visited by initial. make no causal claim.
  perform no online IK intervention.

## result

all 12 registered shards completed sequentially from evaluator commit `45fcab4`
without infrastructure errors: 756 development episodes across joint and matched
conditions, three seeds, two morphologies, three tasks, and horizons 1, 4, and
16. the aggregate report is `reports/benchmark-exp38-summary.json` at SHA-256
`e0e7eaf7ec25600b611cd66780a6dc3d0ca8c374a4f871b15a801716703307bb`.

the horizon mechanism was partial, not strong. joint H16-minus-H1 success gained
3.17 percentage points in the three-seed mean, with per-seed changes of +14.29,
-4.76, and 0.00 points. matched gained 1.59 points, with +2.38, +7.14, and
-4.76 points. milestone progression improved for both, but neither condition
reached the registered 5-point materiality threshold. H4 had the highest pooled
success: joint was 6/126 at H1, 13/126 at H4, and 10/126 at H16; matched was
18/126, 23/126, and 20/126.

policy-visited state shift had no strong association. late-to-initial normalized
first-action error ratios were 0.89, 0.73, and 0.82 for joint and 0.92, 0.89, and
0.87 for matched; no seed reached the preregistered twofold criterion.

policy-generated geometry reconstruction exceeded the frozen references in both
conditions. translation p95 upper bounds were 0.05 m for joint and 0.10 m for
matched against 0.001 m; rotation was 10 degrees for both against 1 degree.
gripper p95 stayed at or below 0.0001 native units, and decoded-command clipping
was zero. adjacent within-chunk translation changes averaged 0.040--0.053 mm,
while cross-replan overlap error increased from about 8 mm at H1 to 12.6--13.8
mm at H4 and 20.0--21.6 mm at H16 in both conditions.

at H16, joint versus matched replay success was 19.05% versus 38.10% for push,
4.76% versus 7.14% for lift, and 0% versus 2.38% for pick/place. both usually
reached the object, but lift attainment remained 4.76% for joint and 9.52% for
matched; pick/place lift attainment was 0% and 2.38%. the frozen-reference H16
outcome match rate was 99.21% for both conditions, while exact trajectory match
was 97.62% for joint and 93.65% for matched. the mismatches changed joint push
from 21.43% source to 19.05% replay and matched lift from 4.76% source to 7.14%
replay. aggregate replay success differed from the frozen source by -0.79 and
+0.79 points respectively, consistent with the documented boundary-outcome
instability.

## finding

execution horizon does not receive strong or material support in this
seven-scenario diagnostic: H16 misses the materiality criterion, H4 performs
best descriptively, and effects reverse across seeds. this does not exclude a
horizon contribution in the broader Experiment 037 population. policy-visited
first-action error does not amplify, so this assay does not support the
registered distribution-shift association.

the strongest registered evidence is a shared policy-action reconstruction
problem: both conditions exceed the physical references by large margins. both
also show centimeter-scale descriptive cross-replan target differences despite
smooth within-chunk commands, but no expert baseline, success association, or
defect threshold was registered for overlap. these observations plausibly help
explain why both conditions remain weak, but they do not explain the
joint-specific deficit: matched reconstruction is at least as poor by p95 and
its overlap differences are similar or larger. the remaining joint gap appears
in task achievement, especially push and lift progression, rather than in a
diagnosed joint-only geometry, horizon, or late-state mechanism. these are
associations, not causal isolation.

## decision

- do not advance or reject a model from this diagnostic.
- keep the final split untouched.
- treat policy-action FK/IK reconstruction as a shared prerequisite before
  another closed-loop comparison; retain cross-replan target differences as a
  candidate issue requiring a baseline or intervention.
- do not attribute the joint-versus-matched gap to execution horizon or measured
  policy-visited first-action shift; a future causal intervention must isolate
  task-conditioned control quality after the shared geometry path is improved.
- do not add training, nearest-neighbor references, counterfactual cloning,
  opposite specialists, stack, H8, H32, expert-first control, or online IK.
