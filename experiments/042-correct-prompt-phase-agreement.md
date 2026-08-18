# 042: correct-prompt phase agreement

## question

on a hypothesis-fresh development slice, do joint-core lead-zero actions under
the correct task prompt have materially and broadly worse agreement with the
deterministic expert than matched specialists across tasks, morphologies, and
phase endpoints?

## setup

- freeze benchmark definition SHA-256
  `9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f`,
  development manifest SHA-256
  `4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576`,
  Experiment 036 summary SHA-256
  `76d7a3b0dc8954fc1fdbf8def99c840881ee3d9b523bf9f3386be5b82be0308f`,
  Experiment 041 summary SHA-256
  `d8195516eb6c84d90da31e74459b33779aab3382000650df8e9b941f35cd217f`,
  imported Experiment 041 evaluator SHA-256
  `2ca9ae5b48782b4b378882accf886dd9b65bd1c7ed3905bb20dfc3c859f9d62c`,
  and expert implementation SHA-256
  `7fb869a10756864530889f809a8305ee1758bdc51eaeeeda0ff2f451f3b7f8e9`.
  use only fixed Experiment 036 step-1,200 cores at seeds 36001--36003.
  matched means the SO-101 specialist on SO-101 and Panda specialist on Panda.
  perform no training, checkpoint selection, policy-controlled rollout,
  opposite-specialist comparison, stack evaluation, or final-split access.
- use exact development indices 0028--0034 in SO-101 and Panda crossed with push,
  lift, and pick/place. this is hypothesis-fresh to Experiment 041 and slice-fresh
  to Experiments 038--041, not globally unseen because Experiment 037 evaluated
  all development cases. require every registered factor axis once per cell.
- use two non-resumable morphology shards, ordered `so101,panda`, named
  `benchmark-exp42-{morphology}-phase-action-quality.json` and staged under
  ignored `reports/exp42-runtime/`. use lock path
  `reports/exp42-runtime/.registered.lock` and lock identity
  `exp42-so101-panda-phase-endpoints-v1`. write only completed reports atomically;
  one orchestrator holds the lock across the complete `so101,panda` schedule. any
  partial or existing shard, including a completed first shard followed by second-
  shard failure, forbids aggregation and requires deleting both staged shards
  before a full restart. after both validate, aggregate to
  `reports/benchmark-exp42-summary.json` and promote both immutable source shards
  to tracked `reports/` without changing their registered bytes.
- each morphology shard runs 21 unchanged native `PrivilegedBenchmarkExpert`
  episodes for at most 500 steps at 30 Hz. require ordered scored phases
  `precontact,push` for push; `precontact,contact,close,lift` for lift; and
  `precontact,contact,close,lift,transit,release,retreat` for pick/place, between
  initial `open` and successful terminal `done`.
- capture exactly two endpoint records per scored phase. `first` is the loop-entry image
  and state when that phase first appears at `phase_tick == 1`. `last` is the
  pre-call image and state for the final expert call whose loop-entry phase is
  that scored phase, detected after the single `action()` call changes phase.
  label using the pre-call phase, convert the returned native action to canonical
  form against the unchanged captured state, then execute that exact native
  action. when a phase has one action, materialize both endpoint labels with the
  same tick, arrays, and hashes; their within-phase mean remains one observation.
  terminal transition no-ops returned for final push, lift, and retreat calls are
  intentional `last` targets, not prior motion commands. exclude `open`, `done`,
  and `failed`. require the post-`action()` canonical-state hash to equal the
  pre-call captured-state hash before converting the target.
- retain all 182 anchors for one morphology in memory while evaluating all three
  paired model seeds, so all six seed-condition arms receive byte-identical top
  640x480 uint8 images, FK canonical states, expert targets, and exact correct task
  prompts. persist canonical tensor hashes, expert arrays, and actions, not raw
  images. there are 42 expert episodes, 182 scenario-phase units, 364 endpoint
  records, potentially fewer unique expert calls represented when one-action
  phases duplicate endpoints, 2,184 scored chunks, 2,184 repeat chunks, 4,368
  policy inferences, and 1,092 paired joint/matched comparisons.
- reset policy state before every prediction. query the exact correct prompt only,
  using the frozen Experiment 041 strings and hashes. immediately repeat after
  another reset; score the first output and require full-chunk maximum difference
  at most `1e-5`. require finite `[1,32,1,10]` output and score lead zero only.
  later leads have no contemporaneous expert target. do not decode or execute
  policy output. counterbalance condition order at every zero-based anchor and
  seed as `(anchor_ordinal + seed_ordinal + morphology_ordinal) % 2`, where zero
  means joint first. enumerate anchors by task `push,lift,pick/place`, ascending
  scenario index, registered phase order, then endpoint `first,last`; reset the
  ordinal per morphology. seed ordinals map 36001, 36002, 36003 to 0, 1, 2 and
  morphology ordinals map SO-101 and Panda to 0 and 1. this gives each condition
  91 first queries per seed and morphology.
- for prediction `a`, expert target `e`, and frozen scales
  `(0.25,0.25,0.25,1,1,1,1,1,1,0.5)`, primary loss is
  `mean(((a-e)/scale)^2)`. report physical translation norm error in metres,
  rotation geodesic error in degrees, and effective-gripper error after clipping
  both channel 9 values to `[0,1]`. finite degenerate predicted 6D rotation remains
  in normalized loss, is counted, and receives 180 degrees physically; a
  degenerate expert target is invalid delivery. also report all ten normalized
  channel losses without defining a new composite. agreement with this stateful
  deterministic expert is not unique optimal-action identification.
- aggregate every metric by equal-weight first and last within scenario-phase,
  phases within scenario, seven scenarios within morphology/task, three tasks
  within morphology, two morphologies within model seed, then three paired seeds.
  compute task, morphology, endpoint, and task-phase subgroups by restricting
  this hierarchy, never sample-weighted pooling. apply the same hierarchy
  separately to every channel loss. at each level report
  joint, matched, paired difference `joint-matched`, and ratio `joint/matched`.
  define the overall ratio exactly as the mean of the three seed-level joint
  hierarchical losses divided by the mean of the three matched losses; seed
  checks use each seed's ratio. task, morphology, and endpoint differences below
  refer to primary normalized loss. zero or undefined matched denominator makes
  the dependent criterion false.
- delivery requires all scenarios, ordered phases, endpoint anchors, and successful
  expert episodes; finite valid expert actions and targets; per-shard expert
  round-trip decoded-action clipping at most 1% under the existing `2e-5` rule,
  translation p95 at most 1 mm using `np.percentile(values,95)`, rotation p95 at
  most 1 degree, and maximum gripper error at most one native unit; exact image,
  state, target, instruction, tokenizer, checkpoint, output, evaluator, runtime,
  schedule, and outcome-chain hashes; identical anchor hashes across all six
  model arms; deterministic repeats; common clean commit/runtime; nonoverlapping
  shard intervals; immutable source-report hashes; a contract field
  `final_split_loaded: false`; and no final-split artifact loaded or accessed.
  compute delivery clipping and physical percentiles over all 182 endpoint
  records per morphology, including duplicated labels for one-action phases.
  persist requested native action, canonical target, decoded and realizable native
  action, realized canonical target, tokenizer arrays, and both full scored and
  repeat `[32,1,10]` chunks with dtype, shape, and contiguous-byte hashes so every
  delivery and metric claim independently recomputes.
- confirm a `broad_correct_prompt_expert_action_agreement_deficit` only if delivery passes,
  the equal-weight three-seed normalized joint/matched ratio is at least 1.10, at
  least two seed ratios are at least 1.10, both independently aggregated `first`
  and `last` endpoint ratios are at least 1.10, every task paired difference is
  nonnegative, both morphology paired differences are nonnegative, and gripper
  directionally corroborates with an overall joint/matched error ratio at least
  1.10, an absolute overall error difference at least 0.01 effective opening
  fraction, and at least two seed ratios at least 1.10. the absolute floor is one
  percentage point of the effective opening range. this reuses the preregistered 10%
  materiality scale and predeclares gripper from Experiment 041's exploratory
  decomposition without selecting a phase. otherwise classify the broad deficit
  as not confirmed on the fresh slice; localized differences remain descriptive
  and no broad physical-component claim follows.
  make no inferential, population, closed-loop, admission, or advancement claim.

## result

both registered morphology shards completed sequentially from evaluator commit
`7fcebe1` under one orchestrator and runtime signature: 42 successful expert
episodes, 182 scenario-phase units, 364 endpoint records, 4,368 policy
inferences, and 1,092 paired joint/matched comparisons. the aggregate report is
`reports/benchmark-exp42-summary.json` at SHA-256
`104c16ca2552eb0d2ba7efc65c9228eeecd488d67d765f49d70931aa0d15abf8`.
both promoted source reports are byte-identical to their staged files; all frozen
inputs, 57 checkpoint artifacts, expert and policy arrays, tokenizer and output
hashes, endpoint transitions, outcome chains, schedules, delivery gates, and
hierarchical aggregates validated independently.

the registered broad correct-prompt expert-action agreement deficit was
confirmed. joint normalized loss was 0.030812 versus 0.027209 for matched, a
paired difference of 0.003602 and ratio of 1.1324. seed ratios were 1.0407,
1.1654, and 1.1923, satisfying two of three. every task difference was positive:
push, lift, and pick/place ratios were 1.1393, 1.1306, and 1.1334. SO-101 and
Panda differences were both positive, with ratios 1.0940 and 1.1381. first and
last endpoint ratios were 1.1574 and 1.1252.

gripper physically corroborated the deficit. joint effective-gripper error was
0.13439 versus 0.11687, a 0.01752 absolute difference and 1.1499 ratio. seed
ratios were 1.1184, 1.2007, and 1.1306. by contrast, joint translation error was
slightly lower at 0.02737 versus 0.02828 m, and rotation error was 3.2347 versus
3.2439 degrees. normalized channel 9 carried the dominant absolute excess.

delivery passed both shards with zero clipping and exact deterministic repeats.
SO-101 expert round-trip translation and rotation p95 were 0.0064 mm and 0.0048
degrees; Panda values were 0.0051 mm and 0.0114 degrees. no expert or predicted
rotation was degenerate. all nine registered confirmation gates passed.

## finding

the fresh development slice confirms that the joint core has broadly worse
correct-prompt agreement with the deterministic expert than matched specialists
across tasks, morphologies, and first/last phase endpoints. the effect is not a
uniform physical action degradation: translation and rotation slightly favor the
joint core, while gripper target error is materially and seed-consistently worse.

this narrows the next mechanism to gripper target quality. it does not determine
whether the source is half per-cell exposure in joint training, shared-capacity
interference, or another gripper-specific representation defect. agreement with
one phase-stateful expert is not unique optimality, and endpoint states are
expert-visited rather than policy-visited. no inferential, population,
closed-loop, admission, advancement, broad-physical-deficit, or final-split claim
follows.

## decision

- do not advance or reject a model from this development-only diagnostic.
- keep the final split untouched.
- isolate gripper target quality next, separating the joint model's lower per-cell
  exposure from shared-capacity interference before any retraining decision.
- do not modify translation, rotation, geometry projection, or replan alignment
  from this result.
