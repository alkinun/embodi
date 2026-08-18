# experiments

records are numbered by execution order. each record uses the same sections:
question, setup, result, finding, and decision.

raw metrics are in `reports/`. checkpoints are local under `outputs/`.

| id | question | decision |
| --- | --- | --- |
| [001](001-xperience-pipeline.md) | can xperience labels train the model? | use the pipeline. |
| [002](002-full-core-transfer.md) | does full-core transfer help? | do not transfer the full core. |
| [003](003-control-decoder.md) | why did low loss fail in control? | fix limits and train on policy states. |
| [004](004-deterministic-regression.md) | does regression improve control? | use regression for this benchmark. |
| [005](005-expert-only-transfer.md) | which component transfers? | transfer only the action expert. |
| [006](006-action-expert-scaling.md) | which expert size works best? | keep the 53M expert. |
| [007](007-decoder-robustness.md) | why was decoder training unstable? | reuse one broad teacher cache. |
| [008](008-xperience-data-scaling.md) | does more human data help transfer? | 1,000 clips win at the fixed budget. |
| [009](009-nested-human-data-scaling.md) | does matched-exposure scaling help? | more unique data helps. |
| [010](010-position-shift-robustness.md) | does success survive x shifts? | expand robot and decoder coverage. |
| [011](011-cube-position-coverage.md) | do wider cube positions fix x shifts? | no. coverage alone causes forgetting. |
| [012](012-checkpoint-selection.md) | did an earlier checkpoint work better? | no. no robust control peak appeared. |
| [013](013-execution-horizon-screen.md) | can execution horizon recover near shifts? | horizon 16 wins the small screen. |
| [014](014-execution-horizon-confirmation.md) | does horizon 16 beat horizon 8 on new scenes? | use horizon 16, but improve the policy. |
| [015](015-center-weighted-cube-coverage.md) | can center weighting add coverage without forgetting? | inconclusive. the effect reverses across core seeds. |
| [016](016-training-seed-factorization.md) | does initialization or data order drive seed instability? | initialization dominates, with a large loader interaction. |
| [017](017-same-seed-reproduction.md) | is the exceptional center cell reproducible with the same seeds? | behavior broadly replicates, but training is not bitwise reproducible. |
| [018](018-full-deterministic-reproduction.md) | does strict determinism reproduce a full training run? | yes, exactly, but the deterministic policy is weak. |
| [019](019-deterministic-model-seed-screen.md) | can deterministic initialization recover strong control? | seed 3993 peaks at step 1,000, then control collapses. |
| [020](020-deterministic-peak-localization.md) | can a nearby checkpoint improve the deterministic peak? | step 1,100 reaches 66/150 but the peak is narrow. |
| [021](021-control-peak-interpolation.md) | can checkpoint interpolation smooth the narrow control peak? | the basin is connected, but interpolation does not improve it. |
| [022](022-near-failure-telemetry.md) | where does the selected policy fail on near positions? | near failures are approach/alignment failures before lift. |
| [023](023-deterministic-expert-transfer.md) | does expert transfer survive deterministic replication? | yes; 63/150 versus 26/150 with matched selection. |
| [024](024-near-weighted-cube-coverage.md) | can near-weighted data fix approach failures without forgetting? | near improves by 6/50, but retention falls to 70.7%. |
| [025](025-additive-near-coverage.md) | can additive near data improve control without forgetting? | near gain misses target; unexpected far gain needs confirmation. |
| [026](026-additive-policy-confirmation.md) | does the additive policy's broad advantage replicate? | no overall; its far advantage independently replicates. |
| [027](027-selected-core-native-decoder.md) | can the selected general core support a robust learned native decoder? | no; optimization is stable but success falls from 66/100 to about 20/100. |
| [028](028-native-decoder-error-decomposition.md) | where does the learned native decoder lose deterministic control? | local first-action approximation error, not horizon growth or seed variance. |
| [029](029-decoder-optimization-budget.md) | can a longer fixed decoder optimization run clear the offline fidelity gate? | no; longer training helps, but normalized and elbow-error gates still fail. |
| [030](030-decoder-capacity.md) | can a higher-capacity decoder clear the fixed offline fidelity gate? | no; capacity helps, but exposes a loss-weighting mismatch. |
| [031](031-control-tolerance-decoder-loss.md) | can a control-tolerance loss meet the held-out native fidelity gate? | no; elbow and direction gates narrowly fail, so closed loop remains blocked. |
| [032](032-ik-teacher-continuity.md) | is the ik teacher translation-sensitive near held-out commands? | rarely under axis-aligned probes; rotation and reconstruction remain open. |
| [033](033-calibration-error-robustness.md) | what joint-offset accuracy does geometry-first control require? | sampled fixed offsets retain aggregate success through 5 degrees; physical gates remain required. |
| [034](034-session-task-diversity.md) | does session/task breadth help at fixed human pretraining volume? | yes; five sessions reduce three-seed held-out loss by 23.0%. |
| [035](035-diversity-transfer.md) | does broader human pretraining improve expert-only SO-101 transfer? | fixed-step single-pair screen favors five sessions 7/30 to 3/30, but is not significant. |
| [036](036-joint-canonical-core.md) | can one canonical core learn SO-101 and Panda jointly at matched exposure? | yes; joint loss is within 4.64% of specialists and 85.64% below the zero-shot specialist-transfer baseline. |
| [037](037-canonical-core-geometry-control.md) | can the shared canonical core retain matched-specialist closed-loop geometry-control success? | no; joint success is 6.00% versus 13.08% for specialists and misses the 5-point margin. |
| [038](038-canonical-trajectory-diagnostics.md) | which trajectory mechanism explains the canonical core's closed-loop failure? | no measured joint-specific mechanism among the assayed candidates; both conditions show large reconstruction error. |
| [039](039-canonical-feasibility-projection.md) | does feasible canonical action projection causally improve H4 control? | no; joint gains 1.59 points with one delivery failure, while matched gains zero. |
| [040](040-lead-zero-replan-continuity.md) | does lead-zero replan alignment causally improve H4 control? | no; continuity improves, but joint loses 0.79 points and matched gains only 2.38 points inconsistently. |

## current decision

```text
frozen vlm during human pretraining
                 +
pretrained action expert
                 +
fresh robot adapters and state path
                 +
one decoder for each robot
```

for robot coverage, retain deterministic center-weighted seed 3993 at step 1,100
as the general policy and additive-near step 1,000 as a far-range specialist.
for human pretraining, add more sessions and tasks rather than more windows from
the same episodes. experiment 034 confirms this direction across three paired
seeds, while leaving semantic diversity and independent temporal coverage
confounded.

the selected general core is not approved for learned native-action execution:
experiments 027--031 show stable optimization but severe closed-loop loss with
local decoder approximation error; experiment 032 finds low discontinuity
frequency under reconstructed axis-aligned translation probes but does not test
rotation sensitivity or exact cached-state reconstruction. keep
deterministic ik as the simulation reference and require a newly specified decoder to pass offline
physical-unit gates before any further closed-loop or physical soak evaluation.
the geometry-first path retains aggregate simulation success under the sampled
fixed joint-zero offsets through 5 degrees, but experiment 033 is not a
worst-case or physical safety guarantee. require measured calibration and the
existing hardware acceptance gates before autonomous motion.
