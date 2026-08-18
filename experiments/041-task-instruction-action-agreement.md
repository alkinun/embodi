# 041: task-instruction action agreement

## question

does the correct task instruction on average make the frozen policies' lead-zero
canonical actions closer to the deterministic task expert than counterfactual
pretraining-task instructions, and is this prompt effect weaker in the joint core
than in matched specialists?

## setup

- freeze benchmark definition SHA-256
  `9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f`,
  development manifest SHA-256
  `4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576`,
  Experiment 036 summary SHA-256
  `76d7a3b0dc8954fc1fdbf8def99c840881ee3d9b523bf9f3386be5b82be0308f`,
  Experiment 038 summary SHA-256
  `e0e7eaf7ec25600b611cd66780a6dc3d0ca8c374a4f871b15a801716703307bb`,
  Experiment 039 summary SHA-256
  `87bd3d30f350e3cda16a8ac91dfeeec5249b46ace97e42de860362d7bef5a8a8`,
  Experiment 040 summary SHA-256
  `05018ce882e01a5e30d00672ce84db007bdf60992c629ee86485ee7dab55efea`,
  and imported Experiment 040 evaluator SHA-256
  `d9c73baff704b5c0e56fd46127f3be07488a1401fa7add285a096478d893f3dd`.
  use only fixed Experiment 036 step-1,200 cores. perform no training,
  checkpoint selection, policy-controlled rollout, or final-split evaluation.
- use exact development indices 0021--0027 in SO-101 and Panda crossed with
  push, lift, and pick/place. this is intervention-fresh to Experiments 038--040,
  not globally unseen because Experiment 037 evaluated all development cases.
  the seven indices cover every registered factor axis once per cell.
- retain model seeds 36001--36003 and compare joint with the matched SO-101 or
  Panda specialist. use six seed-by-morphology shards. each shard contains both
  model conditions, 21 deterministic-expert episodes, and 91 shared anchors:
  two phases for push, four for lift, and seven for pick/place per scenario.
  there are 126 expert episodes, 546 anchors, and 4,368 policy inferences. reports
  are `benchmark-exp41-seed{seed}-{morphology}-task-instruction.json`, staged in
  ignored `reports/exp41-runtime/` under one seed-major, morphology-major schedule
  ordered seeds `36001,36002,36003` then morphologies `so101,panda`. use lock path
  `reports/exp41-runtime/.registered.lock` and lock identity
  `exp41-seed-major-morphology-major-v1`. registered shards are non-resumable:
  write only a completed report atomically; any partial file aborts validation and
  must be discarded before restarting that entire shard.
- run the native `PrivilegedBenchmarkExpert` for 500 steps at 30 Hz. require the
  exact ordered unique scored phases between initial `open` and terminal `done`:
  `precontact,push` for push;
  `precontact,contact,close,lift` for lift; and
  `precontact,contact,close,lift,transit,release,retreat` for pick/place. capture
  one anchor at loop entry when the current scored phase has `phase_tick == 1`,
  after any prior transition has completed. then call `action()` exactly once,
  convert that returned native action to canonical form against the captured
  state, and execute the unchanged native action. retain the exact top 640x480
  uint8 image and FK canonical state in memory; persist their canonical tensor
  hashes, scenario identity, phase, and expert actions, not raw images. stop only
  when the expert is terminal. every episode must finish successfully in `done`.
- query each frozen policy on the identical image and state under all three exact
  pretraining instructions. their SHA-256 values are push
  `ef3724e245f1e3424e91fd211030b7593747d03e0be0c191cddd686533a50a6b`,
  lift
  `c89dbd7f6818e34f3d37c485393d1a67464be96efca3e54d06daac4331828e9a`,
  and pick/place
  `f345e35bfcc1697c3beb961531cbf9b7dcf0c14d3b96cf84c0b64aa7b55d5f0c`.
  correct instruction is treatment; the mean of both wrong instructions is the
  paired control. stack is excluded because it was not a pretraining task.
- reset policy state before every prediction. counterbalance the three instruction
  orders over the zero-based anchor ordinal with permutations `(push,lift,pick)`,
  `(push,pick,lift)`, `(lift,push,pick)`, `(lift,pick,push)`, `(pick,push,lift)`,
  `(pick,lift,push)`, selected by `ordinal % 6`, and repeat the correct-instruction
  query after each block as a determinism guard. score the first correct query,
  never an average. require shape `[1,32,1,10]`, finite output, and repeated-correct
  chunk maximum difference at most `1e-5`. retain exact image, state, instruction,
  tokenizer, output, checkpoint, evaluator, runtime, schedule, and outcome-chain
  hashes. use dtype plus shape plus contiguous bytes hashing as in Experiment 040;
  require distinct `input_ids` hashes for the three instructions and separately
  record `attention_mask`, source uint8 image, float32 canonical state, and float32
  output hashes. define anchor ordinal by registered scenario order followed by
  required phase order. even zero-based shards process joint first and odd shards
  matched first while using the same in-memory anchors. the aggregate records
  immutable source-report SHA-256 values; a report does not hash itself.
- score lead zero only. for predicted canonical action `a`, expert target `e`,
  and frozen scales `(0.25,0.25,0.25,1,1,1,1,1,1,0.5)`, primary loss is
  `mean(((a-e)/scale)^2)`. aggregate correct loss and the mean of the two wrong
  losses separately: equal-weight phases within scenario, scenarios within task
  and morphology, tasks, then morphologies. at each seed define margin as
  `wrong-correct` and relative benefit as `margin/wrong`; zero wrong loss makes
  benefit `null` and support false. condition-level losses and benefit use the
  equal-weight mean of the three seed-level values. task margins use the same
  hierarchy restricted to that task and then equal-weight seeds. anchors and
  phases are not independent replicates.
- report correct and wrong loss, margin, relative benefit, correct-instruction
  top-one rate with ties, translation norm error, rotation geodesic error,
  effective-gripper error, and output separation by condition, seed, morphology,
  task, and phase. top-one gives the correct instruction `1/k` credit when `k`
  instruction losses tie for the minimum within absolute tolerance `1e-12`.
  output separation is the mean normalized lead-zero MSE between correct output
  and each wrong output. effective-gripper error is the absolute difference after
  clipping prediction and expert channel 9 to `[0,1]`. a 6D rotation is degenerate
  when its first axis norm or orthogonalized second axis norm is at most `1e-12`;
  it remains in primary loss, is counted, and receives exactly 180 degrees in the
  physical secondary metric. report paired joint/matched correct-loss ratio and
  matched-minus-joint relative-benefit interaction. the condition mean loss ratio
  is the ratio of three-seed mean losses; two-seed checks use per-seed ratios.
  report physical secondary errors separately for correct and mean-wrong arms.
  any zero denominator or undefined required seed value produces `null` for the
  dependent ratio or interaction and makes the corresponding support or joint-
  deficit classification false.
- delivery requires all expected scenarios, phase anchors, and successful expert
  episodes and finite dimensionally valid native and canonical expert actions.
  for every anchor in every shard, decode its canonical expert target against the
  unchanged captured environment state, re-encode without executing the decoded
  command, and use the existing `2e-5` native clipping rule. each shard separately
  requires at most 1% decoded-action clipping, translation p95 at most 1 mm,
  rotation p95 at most 1 degree, and maximum gripper error at most 1 native unit.
  compute p95 exactly as `np.percentile(values, 95)` with NumPy's default linear
  method.
  delivery also requires all exact instruction and distinct
  tokenizer-input hashes; identical image and state hashes across instruction
  and model arms; valid deterministic policy outputs; complete checkpoint and
  source hashes; one locked sequential schedule with nonoverlapping shard times;
  and no final-split artifact loaded. instruction-invariant output is an outcome,
  not a delivery failure.
- classify counterfactual-instruction expert-agreement support separately. support
  requires passing delivery, equal-weight three-seed mean relative instruction
  benefit at least 10%, nonnegative seed-level margin in at least two seeds, and
  nonnegative three-seed mean margin for every task. classify a joint-specific
  prompt-effect deficit only if matched is supported, the ratio of joint to
  matched three-seed mean correct loss is at least 1.10 and at least two per-seed
  ratios are at least 1.10, and matched relative benefit exceeds joint by at least
  10 percentage points in the condition mean and at least two seeds. make no
  inferential, closed-loop-success, admission, or advancement claim.

## result

pending.

## finding

pending.

## decision

- do not advance or reject a model from this development-only diagnostic.
- keep the final split untouched.
- if both conditions use the correct instruction but remain inaccurate, target
  broader phase-conditioned action quality rather than language conditioning.
- instruction invariance is ambiguous because vision may identify the task; do
  not prescribe a conditioning repair without a visually task-ambiguous assay.
- if a joint-specific prompt-effect deficit is supported, isolate shared-core
  task interference.
