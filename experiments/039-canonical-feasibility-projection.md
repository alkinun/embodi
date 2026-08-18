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
  and instruction. require exact canonical-state and instruction hash agreement.
  allow renderer variation only when the uint8 image maximum absolute difference
  is at most 1 and the fraction of differing image channel values is at most
  `1e-4`; retain exact component hashes and report exact image-hash agreement
  descriptively. report image maximum difference, differing count and fraction,
  exact state and instruction checks, and overall initial-input equivalence.
  use the actual snapshot from the first arm in the registered counterbalanced
  arm order as one shared pair anchor for both initial policy inferences. reset
  policy state and predict separately for each arm from that identical snapshot,
  in arm order. report the source arm, exact anchor hashes, and require those
  hashes to equal the source arm's actual snapshot. compare the two first
  unprojected policy chunks and require maximum absolute difference at most
  `1e-5`; report their exact hashes and difference. this direct same-input GPU
  repeatability guard remains an infrastructure invariant and is not loosened.
  shared first inference isolates the intervention by giving both arms identical
  initial policy intent while counterbalancing the renderer-source arm. later
  replans use each live arm observation after trajectories diverge. a state,
  instruction, image-tolerance, or chunk mismatch invalidates treatment delivery
  and is an infrastructure error, not a task failure. do not require later
  trajectories or outcomes to reproduce.
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

## pre-completion protocol corrections

- merged evaluator commit `d54589e` at evaluator SHA-256
  `908bee414c8e4a70906dcff9f676e83a62a4643d2cbee0eecbb56c46b9c37757`
  completed the first `joint/seed36001/so101` shard, then failed closed in the
  `joint/seed36001/panda` shard at
  `development-panda-push_to_zone-0010` because the exact initial-input hashes
  differed. the preserved completed SO-101 report SHA-256 is
  `9a3c50ee7cd624b1a4fd2c8b662c8c108a79bd5c884d4827c697d0b4f1010ff8`;
  the preserved partial Panda report SHA-256 is
  `128f04699e1aad2169a1383f44f64a97807f0ac44c6d4baaa0d67553d2b19850`.
  no full aggregate existed and no condition hypothesis was evaluated.
- a direct five-pair reproduction produced one sporadic mismatch. canonical
  state and instruction hashes matched exactly, while 3 of 921,600 uint8 image
  channel values differed by exactly 1. exact renderer-byte equality is therefore
  not a valid infrastructure invariant. the bounded image criterion above is a
  pre-completion operational correction; the first unprojected chunk tolerance
  remains unchanged as the direct policy-output invariant.
- the attempt reports from evaluator `d54589e` are superseded and must not be
  resumed or aggregated. discard any staged reports under the old protocol
  contract and rerun all 12 registered shards from the beginning. no compatibility
  path or report migration is permitted.
- the all-shard restart under merged evaluator commit `0e76ce6`, evaluator
  SHA-256 `da4ad4f525eca3a3188e126f85f76e5121e9894833c1369defb7d7263991160a`,
  completed `joint/seed36001/so101` and then failed closed in
  `joint/seed36001/panda` at `development-panda-push_to_zone-0008` on the
  unchanged first-chunk `1e-5` gate. the completed SO-101 report SHA-256 is
  `dbed3df5e6ef1b803ec83656af5c07f46f85505695e283470e6b051199343543`;
  the partial Panda report SHA-256 is
  `1598f5ecd83e706a230cec4359b9653dbfba32ed229c125b74649284cfa762e8`.
  no aggregate existed.
- characterization on that exact Panda scenario and model found bitwise-equal
  first chunks whenever the fresh images matched bytewise. when 3 white pixel
  channels differed as 254 versus 255, all 320 chunk outputs differed, with
  maximum absolute difference `0.00101360865` and mean absolute difference
  `0.0005492731`. twenty repeated GPU predictions from one identical frozen
  snapshot were bitwise equal. this identifies renderer input variation, not GPU
  repeatability, as the source of the failed action comparison; the first-chunk
  tolerance therefore remains unchanged.
- the `0e76ce6` reports are also superseded and must not be resumed or aggregated.
  the shared counterbalanced pair-anchor rule above changes the frozen report
  contract, so discard all staged reports and restart all 12 registered shards
  from the beginning without migration or compatibility handling.

## result

all 12 registered shards completed sequentially from evaluator commit `2b0b2b8`
under one runtime signature: 252 development pairs and 504 fresh episodes. the
aggregate report is `reports/benchmark-exp39-summary.json` at SHA-256
`87bd3d30f350e3cda16a8ac91dfeeec5249b46ace97e42de860362d7bef5a8a8`.
all 12 source report hashes, checkpoint artifacts, pair input invariants, outcome
chains, telemetry, schedules, and aggregates validated.

neither condition received registered causal support. the joint core improved
from 11/125 to 13/125 eligible-pair successes, an equal-weight three-seed gain of
1.59 percentage points. seed gains were +2.38, +2.38, and 0.00 points; ordered
progression improved by 0.0146, and discordant outcomes favored projection 6 to
4. this missed the 5-point materiality margin. in addition, one
`joint/seed36003/so101` lift pair exhausted all 17 grid points at chunk 30,
including alpha zero. its 120 prior projected commands were retained, its task
outcome was excluded, and the joint delivery gate correctly failed. the other
executed projected commands passed all physical thresholds.

matched specialists remained 24/126 successes in both arms. seed changes were
-4.76, 0.00, and +4.76 points, discordant outcomes tied 9 to 9, and ordered
progression decreased by 0.0196. matched delivery passed, but success and
progression criteria did not.

projection changed 24,827/57,889 joint commands (42.89%) and 25,951/53,682
matched commands (48.34%); accepted alpha zero occurred in 10.80% and 10.70%.
exposure was strongly morphology-dependent: 72.33% of SO-101 commands versus
20.91% of Panda commands were scaled, with alpha zero at 21.77% versus 0.65%.
native clipping was zero in both arms of every shard. joint descriptive gains
were concentrated in push (+7.14 points) and SO-101 (+4.76), while Panda was
-1.59; matched SO-101 and Panda effects canceled at -6.35 and +6.35 points.

## finding

this deterministic feasibility projection does not receive development
diagnostic causal support for either model condition. the small joint gain is
directionally consistent and accompanies better ordered progression, but it is
not material, is concentrated in SO-101 push behavior, and cannot clear the
registered delivery gate. matched specialists show no mean success gain,
negative progression, and opposing morphology effects despite successful
delivery.

the result does not support round-trip geometry as a sufficient explanation for
weak closed-loop control, nor does it establish that geometry is irrelevant.
the intervention frequently suppresses arm motion, especially on SO-101, and
one alpha-zero failure shows that the registered treatment was not universally
deliverable. subgroup effects are exploratory because treatment intensity is
trajectory- and morphology-dependent. no final-split, model-admission,
advancement, or uniform-effect claim follows.

## decision

- do not advance or reject a model from this development-only causal diagnostic.
- keep the final split untouched.
- do not deploy or advance this projection and do not open the final split.
- for matched specialists, successful delivery with zero mean gain does not
  support attributing weak control to round-trip geometry alone.
- for the joint core, retain geometry as unresolved rather than excluded because
  delivery failed once, but do not treat the sub-material gain as causal support.
- next isolate task-conditioned policy quality or cross-replan consistency. any
  future geometry intervention must avoid alpha-zero intent suppression and pass
  an offline universal-delivery check before another closed-loop diagnostic.
