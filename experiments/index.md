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

for robot coverage, recover control under deterministic training before
changing the data distribution again. for human pretraining, add more sessions
and tasks rather than more windows from the same five episodes.
