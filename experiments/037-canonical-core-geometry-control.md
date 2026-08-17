# 037: canonical core geometry control

## question

can the shared-capacity canonical core admitted by experiment 036 retain
closed-loop development-split task success relative to matched specialists when
both use the same deterministic geometry state and action path?

## setup

- freeze benchmark definition SHA-256
  `9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f`
  and development manifest SHA-256
  `4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576`.
  use all 400 development scenarios, exactly 50 in each morphology/task cell.
  do not load or evaluate the final split.
- freeze experiment 036 summary SHA-256
  `76d7a3b0dc8954fc1fdbf8def99c840881ee3d9b523bf9f3386be5b82be0308f`
  and its fixed step-1,200 `core` checkpoints for SO-101-only, Panda-only, and
  joint conditions at model seeds 36001, 36002, and 36003. perform no model or
  checkpoint selection.
- before policy scoring, run the privileged expert actions through
  `canonical_arm_action` and deterministic `native_action_from_canonical`, then
  execute the decoded actions over all development scenarios. require command
  clipping at most 1%, FK/IK translation p95 at most 0.001 m, rotation p95 at
  most 1 degree, gripper round-trip error at most 1 native unit, and no cell's
  success rate to fall by more than 5 percentage points from the frozen
  development-admission report.
- evaluate the top 640x480 camera at 30 Hz with FK canonical state, supplied
  canonical part mask, and native `state=None`. call
  `predict_canonical_action_chunk`, decode with the benchmark environment's
  deterministic geometry path, and execute the first 16 actions of each
  32-step chunk for at most 500 steps. use the regression output only.
- run condition x model seed x execution morphology as 18 resumable shards,
  each containing the 200 scenarios for one execution morphology. specialists
  evaluated on the opposite morphology are descriptive controls. registered
  shard names are
  `benchmark-exp37-{condition}-seed{seed}-{morphology}-geometry.json`.
- count a command as clipped when native-to-simulator-to-native realization
  changes any component by more than `2e-5` native units. this excludes the
  observed float32 degree/radian round-trip floor while retaining a physical
  command-clipping check.
- the matched specialist for each seed is the SO-101 specialist on SO-101
  scenarios plus the Panda specialist on Panda scenarios. the primary gate is a
  5-percentage-point noninferiority margin for joint success versus this matched
  baseline on the overall eight-cell macro, held-out stack macro, and each
  morphology's four-task macro, all in three-seed means. additionally, the
  overall per-seed margin must hold in at least two of three seeds. opposite
  specialists do not enter any gate. require policy-command clipping at most 1%
  in every shard.
- require the matched-specialist overall mean to reach 10% success so the assay
  is sensitive. also require the joint core to reach 10% overall and
  pretraining-task macro success, 5% held-out stack success, and 5% four-task
  macro success on each morphology. these absolute floors prevent equal
  all-failure conditions from passing noninferiority.

this tests shared-capacity closed-loop geometry control. it does not test a
learned native wrapper or decoder, and it is not evidence of representation
invariance.

## result

pending registered execution.

## finding

pending registered execution.

## decision

- run geometry admission before policy shards.
- keep the final split untouched.
- registered admission and shard reports may be written to an external reports
  directory with their exact registered filenames so each run can execute from
  the same clean detached worktree; Experiment 036 artifacts are resolved from
  an explicit read-only artifact root.
