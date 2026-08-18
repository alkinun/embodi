# 043: gripper exposure and shared-training interference

## question

is the joint core's gripper-target agreement deficit caused by its lower
per-cell training exposure, by adding the other morphology's gradients to one
shared core, or materially by both?

## setup

- freeze benchmark definition SHA-256
  `9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f`,
  train generation-manifest SHA-256
  `9d858858c217c2b1200507a503148c99562a06f0beecea054a2b9cc184b840af`,
  validation generation-manifest SHA-256
  `2735a044d11db30172ae6039f09dfe06adc49462f3b47d5e7b99cb312c354f45`,
  development manifest SHA-256
  `4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576`,
  validation-frame manifest SHA-256
  `32681217e8d6f4142ce754b18b2a3a69527edea9abce3506fb55e7d25c28cc1a`,
  Experiment 036 summary SHA-256
  `76d7a3b0dc8954fc1fdbf8def99c840881ee3d9b523bf9f3386be5b82be0308f`,
  Experiment 042 summary SHA-256
  `104c16ca2552eb0d2ba7efc65c9228eeecd488d67d765f49d70931aa0d15abf8`,
  registered trainer SHA-256
  `e6dbf985ea4b79c7cdf5d2bc0bf6c5fa17186f82f8e8297d5454fbb501cba9a3`,
  imported Experiment 042 evaluator SHA-256
  `cc4fb718a3f6341dbaaea0026aca193fd86d5b84412ea7061204ed8f4ce24b7d`,
  and expert implementation SHA-256
  `7fb869a10756864530889f809a8305ee1758bdc51eaeeeda0ff2f451f3b7f8e9`.
  do not access the final split.
- compare three paired conditions at model seeds 36001--36003 and loader seeds
  36101--36103: existing Experiment 036 full-exposure matched specialists `F`,
  new half-exposure matched specialists `H`, and existing Experiment 036 joint
  cores `J`. train only the six missing `H` runs, one SO-101 and one Panda
  specialist per seed. use outputs
  `outputs/benchmark-exp43-half-{morphology}-seed{seed}` and fixed update 1,200;
  perform no checkpoint selection, early stopping, joint retraining, loss-weight
  change, or architecture change.
- invoke `embodi-train-joint --protocol exp43-half-specialist` with the original
  Experiment 036 datasets, configs, deterministic mode, workers zero, learning
  rate `1e-4`, 120-update warmup, cosine decay, AdamW betas `(0.9,0.95)`, weight
  decay `1e-10`, and gradient clipping at 10. retain core-only canonical-state
  training, the frozen vision encoder, trainable backbone LoRA, 53.12M expert,
  top camera, regression objective, 32-step chunks, first-step weight 10, and
  ten unit canonical channel weights.
- each `H` update consumes five samples from each of its three task cells: 15
  actual samples per update, 18,000 total, and 6,000 per cell. divide every
  sample loss by the fixed Experiment 036 denominator 30 rather than by 15.
  therefore the target-morphology samples in paired `H` and `J` runs have the
  same loader stream, count, update position, and per-sample gradient
  coefficient; `J` adds the other morphology's 15 gradients. log the ordinary
  15-sample mean separately from optimization loss and record gradient clipping.
  this is a shared-training intervention, including gradient interaction, Adam
  moments, clipping, and parameter sharing; it does not identify static
  representational capacity alone.
- record each run's per-cell dataset-index sequence SHA-256. independently
  reconstruct the first 6,000 indices for each cell from the frozen dataset,
  cell-specific loader seed, loader implementation, and runtime. require every
  `H` digest to match the corresponding target-cell stream reconstructed under
  the Experiment 036 joint schedule. require exactly 6,000 samples per cell,
  fixed denominator 30, finite metrics and weights, paired seeds, clean runtime,
  fixed update 1,200, byte-identical hard-linked wrapper cores, and complete
  checkpoint/config/trainer/data-manifest hashes. any mismatch is delivery
  failure, not a model result.
- summarize the six training runs without choosing among them. on the frozen
  256-frame validation cells, report existing aggregate regression loss plus
  lead-zero channel-9 normalized loss, raw gripper absolute error and bias,
  effective `[0,1]`-clipped gripper error, and saturation rate by task,
  morphology, and seed. training and validation outcomes must not alter the
  development assay or its gates below.
- after the six fixed checkpoints and their immutable training summary exist,
  freeze a new evaluator and its source-summary hash before running any phase
  assay. the evaluator may import Experiment 042's anchor capture, physical
  scoring, hierarchy, and integrity helpers, but must have an Experiment 043
  report contract and resolve all `F`, `H`, and `J` checkpoints by registered
  hashes. implementation-only corrections before any phase shard begins must be
  recorded; a contract-changing correction requires a new preregistration.
- use exact development indices 0035--0041 in SO-101 and Panda crossed with push,
  lift, and pick/place. this is hypothesis-fresh to Experiment 042 and
  slice-fresh to Experiments 038--042, not globally unseen because Experiment
  037 evaluated all development cases. require every frozen factor axis once per
  cell. run no stack, policy-controlled rollout, opposite-specialist comparison,
  or final-split evaluation.
- reuse Experiment 042's unchanged native `PrivilegedBenchmarkExpert`, exact
  correct prompts, 500-step limit, 30 Hz control, ordered phases, first/last
  phase endpoints, intentional terminal transition no-op targets, unchanged
  native action execution, canonical round-trip evidence, immediate deterministic
  repeat, and lead-zero scoring. retain all anchors for one morphology in memory
  while evaluating all nine seed-condition arms on byte-identical images, states,
  targets, and prompts. use the six condition permutations
  `F,H,J`; `F,J,H`; `H,F,J`; `H,J,F`; `J,F,H`; `J,H,F`, selected by
  `(anchor_ordinal + seed_ordinal + morphology_ordinal) % 6`.
- use two non-resumable sequential morphology shards under one registered lock.
  expect 42 successful expert episodes, 182 scenario-phase units, 364 endpoint
  records, 3,276 scored chunks, 3,276 repeat chunks, 6,552 policy inferences,
  and 1,092 paired observations for each of `J-F`, `H-F`, and `J-H`. write only
  complete reports atomically; any partial schedule requires discarding all
  staged shards and restarting. promote immutable source shards only after both
  independently validate.
- primary endpoint is lead-zero effective-gripper absolute error after clipping
  prediction and expert target to `[0,1]`. use the exact Experiment 042 hierarchy:
  equal-weight first/last within scenario-phase, phases within scenario, seven
  scenarios within morphology/task, three tasks within morphology, two
  morphologies within seed, then three paired seeds. apply the hierarchy
  independently to `F`, `H`, and `J`, then compute
  `D_total=J-F`, `D_exposure=H-F`, and `D_shared=J-H`; these differences sum
  exactly. ratios are descriptive and no clipped percent-explained metric is
  defined.
- secondary fixed metrics are raw gripper absolute error and bias, normalized
  channel-9 squared loss, saturation rate, all ten normalized channel losses,
  overall normalized loss, translation norm, rotation geodesic error, and task,
  morphology, endpoint, task-phase, and seed subgroups. report no selected phase
  or new composite. define signed bias as `prediction-target`; define saturation
  as a prediction strictly outside `[0,1]`, with exact boundary values not
  saturated. agreement with one stateful expert on expert-visited states is not
  unique action optimality or closed-loop evidence.
- delivery reuses Experiment 042's exact scenario, phase, endpoint, success,
  finite-array, `[1,32,1,10]` shape, repeat `1e-5`, common-input hash, prompt,
  tokenizer, round-trip, clipping, translation, rotation, gripper, checkpoint,
  source, evaluator, runtime, schedule, outcome-chain, atomicity, shard-order,
  common-commit, and `final_split_loaded: false` gates, extended from six to nine
  model arms and to the Experiment 043 training artifacts.
- first require the fresh-slice premise: `J/F >= 1.10`, `J-F >= 0.01` effective
  opening fraction, at least two seed ratios at least 1.10, and valid delivery.
  if it fails, report the decomposition but classify the mechanism assay as
  premise-not-confirmed.
- call lower per-cell exposure materially supported only if `H/F >= 1.10`,
  `H-F >= 0.01`, at least two seed ratios are at least 1.10, and delivery passes.
  call shared-training interference materially supported only if `J/H >= 1.10`,
  `J-H >= 0.01`, at least two seed ratios are at least 1.10, and delivery passes.
  classify exposure-dominant, shared-training-dominant, both-supported, or
  neither-materially-supported from these two gates. append `broad` only when the
  supported contrast is nonnegative at both endpoints, all three tasks, and both
  morphologies. negative `J-H` is descriptive positive cross-morphology transfer.
  a failed materiality gate is not equivalence or evidence of no effect. make no
  inferential, population, closed-loop, admission, advancement, or final-split
  claim.

## result

pending registered execution.

## finding

pending registered execution.

## decision

- merge this preregistration and trainer support before launching any new run.
- keep the final split untouched and do not retrain or reweight the joint core
  until the registered mechanism result is complete.
