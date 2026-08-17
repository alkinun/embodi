# 039: canonical feasibility projection

## question

does enforcing physically reconstructable policy commands causally improve H4
closed-loop control for the joint canonical core or matched specialists?

## setup

- freeze benchmark definition SHA-256
  `9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f`,
  development manifest SHA-256
  `4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576`,
  Experiment 036 summary SHA-256
  `76d7a3b0dc8954fc1fdbf8def99c840881ee3d9b523bf9f3386be5b82be0308f`,
  and Experiment 038 summary SHA-256
  `e0e7eaf7ec25600b611cd66780a6dc3d0ca8c374a4f871b15a801716703307bb`.
  perform no training, admission, checkpoint selection, or final-split
  evaluation. H4 is fixed from Experiment 038's descriptive horizon screen;
  this is intervention follow-up, not independent horizon selection.
- use exact development scenario indices 0007 through 0013 in every
  morphology by pretraining-task cell: SO-101 and Panda crossed with push,
  lift, and pick/place. the factor schedule repeats every seven indices, so
  this fresh slice activates each of the seven factor axes once per cell while
  avoiding the 0000--0006 sample that generated the geometry hypothesis and
  selected H4. exclude stack as untrained and non-diagnostic. this gives 42
  scenarios, 21 for each execution morphology.
- compare the joint core and matched specialists at seeds 36001--36003.
  matched means the SO specialist on SO execution and Panda specialist on
  Panda execution. do not run opposite specialists. one condition by seed by
  morphology shard loads one fixed Experiment 036 step-1200 checkpoint and
  runs paired baseline and projection episodes for all 21 scenarios: 12
  shards, 252 pairs, and 504 fresh episodes. registered reports are
  `benchmark-exp39-{condition}-seed{seed}-{morphology}-projection.json`.
  stage resumable registered reports under the ignored
  `reports/exp39-runtime/` directory so atomic progress writes do not dirty the
  evaluator worktree. after completion and aggregate validation, promote the
  immutable reports to tracked report storage without changing their registered
  basenames or contents.
- retain the top 640x480 camera, supplied FK canonical state, `state=None`,
  regression, 32-command, 30 Hz, maximum-500-step path. execute H4: predict a
  fresh chunk and decode only the next four commands against the unchanged
  observation anchor before stepping. instantiate a fresh environment and
  reset policy state for each arm. use no privileged expert.
- run each baseline/projection pair adjacently. counterbalance arm order by the
  parity of scenario ordinal plus condition ordinal plus seed ordinal plus
  morphology ordinal. run registered shards sequentially because exact replay
  and boundary outcomes were unstable in Experiments 037--038. enforce this
  operationally with one nonblocking Linux advisory lock shared by individual
  registered shard commands and the frozen-order 12-shard orchestrator; record
  the schedule, lock identity, runtime signature, and nonoverlapping UTC run
  intervals in each report.
- before either arm acts, hash the exact initial top image, canonical state,
  and instruction. require exact initial-input hash agreement within each pair.
  compare the first unprojected policy chunks and require maximum absolute
  difference at most `1e-5`; report their exact hashes and difference. any
  mismatch invalidates treatment delivery and is an infrastructure error, not
  a task failure. do not require later trajectories or outcomes to reproduce.
- baseline executes the policy command after the existing deterministic
  20-iteration IK decode. projection preserves normalized gripper intent and
  tests translation/rotation scales on the fixed descending grid
  `1, 15/16, ..., 1/16, 0`. translation is multiplied by the scale. rotation
  follows the deterministic shortest path from identity to the policy relative
  rotation on SO(3), then is encoded back to canonical 6D. reject nonfinite,
  non-orthonormal, or non-positive-determinant rotations.
- every scale trial starts from the same unchanged live configuration because
  the decoder mutates only scratch FK state. for each tested scale, decode once,
  re-encode with `canonical_arm_action`, and measure translation, rotation,
  and native gripper round-trip error. accept the first grid value with errors
  at most 0.001 m, 1 degree, and 1 native gripper unit. execute the exact native
  action returned by that accepted trial; never decode it again. if scale zero
  is not feasible, execute nothing, record a treatment-delivery infrastructure
  error, and make causal support for that condition false. preserve telemetry
  from commands executed before that failure, still run the paired fresh arm,
  and exclude the pair from success and progression estimates rather than
  counting it as a task failure. continue the shard and report eligible-pair
  counts and nullable estimates when no eligible pair remains. do not claim the
  selected grid point is a globally maximal feasible scale.
- record reconstruction and native clipping for every executed command in both
  arms. for projection, report intervention count (`alpha < 1`), alpha-zero and
  alpha-one counts, fixed-grid alpha histograms, and accepted error maxima by
  condition, seed, morphology, task, and lead. the geometry delivery gate
  requires no treatment failures, every projected executed command within all
  three thresholds, and at most 1% native clipping separately in every arm of
  every shard. clipping is a regression guard, not mechanism evidence.
- after each `step_control_period`, use its returned status and existing
  diagnostics. track ordered task progression without success: push is
  EE-object proximity, target tolerance, then settling; lift is proximity,
  gripper close, 0.07 m lift, then the retained 0.08 m lift predicate; pick/place
  adds target tolerance, settling, and gripper release after the lift predicate.
  advance through currently true consecutive predicates and record first steps.
  episode progression is the attained prefix fraction; average episodes within
  task, then the three tasks equally. success remains the primary endpoint and
  is not included in progression.
- pair outcomes by scenario within each condition, seed, and morphology. report
  paired discordant counts, per-task and per-morphology gains, and equal-weight
  seed means. classify each condition separately as receiving development
  diagnostic causal support only if its three-seed mean paired success gain is
  at least 5 percentage points, at least two seed gains are nonnegative, its
  three-seed mean task-equal ordered progression gain is positive, and the
  geometry delivery gate passes. report both condition hypotheses without a
  multiplicity-adjusted inferential claim. five points is the existing benchmark
  materiality margin, not a model-admission or advancement gate.

## result

pending.

## finding

pending.

## decision

- do not advance or reject a model from this development-only causal diagnostic.
- keep the final split untouched.
- if projection receives support, repair or replace the canonical-to-native
  action path before a newly preregistered closed-loop comparison.
- if projection does not receive support despite successful treatment delivery,
  do not attribute weak control to round-trip geometry alone; next isolate
  task-conditioned policy quality or cross-replan consistency.
