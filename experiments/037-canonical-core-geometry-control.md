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

the registered geometry admission passed all 400 development scenarios. it
retained 100% privileged success in every cell with zero command clipping,
translation p95 `7.83e-6 m`, rotation p95 `0.01095` degrees, and maximum gripper
round-trip error `3.73e-9` native units. the deterministic geometry path was
therefore admitted on privileged expert trajectories, but this does not measure
FK/IK reconstruction error for off-distribution policy-generated commands.

all 18 policy shards completed without infrastructure errors. the primary
three-seed results were:

| endpoint | matched specialists | joint core | joint difference |
| --- | ---: | ---: | ---: |
| overall eight-cell macro | 13.08% | 6.00% | -7.08 pp |
| pretraining-task macro | 17.33% | 8.00% | -9.33 pp |
| held-out stack macro | 0.33% | 0.00% | -0.33 pp |
| SO-101 four-task macro | 14.83% | 9.17% | -5.67 pp |
| Panda four-task macro | 11.33% | 2.83% | -8.50 pp |

the joint-minus-matched overall differences were -4.75, -7.25, and -9.25
percentage points for seeds 36001, 36002, and 36003. only seed 36001 met the
registered 5-point margin, below the required two of three. the joint core also
missed its 10% overall and pretraining-task floors, its 5% Panda floor, and its
5% held-out-stack floor. all policy shards retained zero command clipping.

success was concentrated in push: the joint core averaged 21.67% push success
across morphology/seed cells, 2.33% lift success, and zero pick/place or stack
success. raw admission, shard, and aggregate reports are
`reports/benchmark-exp37-*.json`; the aggregate report is
`reports/benchmark-exp37-summary.json` at SHA-256
`7efde348c16863399167794636d9941360a144b565919bd176f181cca1e15aa4`.

## finding

the offline canonical-regression result from experiment 036 does not translate
to matched-specialist closed-loop geometry control. joint training remains much
better than opposite-specialist transfer, but loses too much success relative to
the in-domain specialists, especially on Panda and across later seeds. neither
joint nor specialist cores demonstrate meaningful zero-shot stack behavior, and
all conditions remain weak on lift and pick/place.

the failure is not explained by the registered expert-trajectory geometry check
or native command clipping. policy-action reconstruction outside the privileged
trajectory distribution remains unmeasured. temporally coherent,
task-conditioned canonical prediction under closed-loop replanning is the
priority hypothesis to diagnose, not an established sole cause. low fixed-frame
regression loss is not an adequate admission criterion for control.

## decision

- reject the Experiment 036 joint core as a closed-loop geometry-control recipe.
- do not advance these checkpoints to learned-state or full-learned control;
  adding adapters or native decoders would confound an already failing core.
- diagnose anchor-relative chunk consistency, task-stage progression, and
  closed-loop trajectory error before another training experiment.
- keep the final split untouched.
