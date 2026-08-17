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

pending registered diagnostic execution.

## finding

pending. this experiment distinguishes diagnostic associations; it cannot by
itself establish a causal mechanism.

## decision

- do not advance or reject a model from this diagnostic.
- keep the final split untouched.
- do not add training, nearest-neighbor references, counterfactual cloning,
  opposite specialists, stack, H8, H32, expert-first control, or online IK.
