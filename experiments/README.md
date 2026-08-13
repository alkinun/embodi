# Experiments

Experiments are numbered in execution order. Each file records the question,
controlled comparison, result, and decision. Raw metrics remain under
`reports/`; checkpoints remain under `outputs/`.

| Experiment | Question | Decision |
| --- | --- | --- |
| [001](001-xperience-pipeline.md) | Are Xperience labels and training usable? | Pipeline validated. |
| [002](002-full-core-transfer.md) | Does full-core pretraining help SO-101? | No; full transfer is harmful. |
| [003](003-control-decoder.md) | Why does low canonical loss fail in control? | Fix action limits and train decoders on policy predictions. |
| [004](004-deterministic-regression.md) | Does removing flow sampling improve control? | Yes; use regression for this benchmark. |
| [005](005-expert-only-transfer.md) | Which pretrained component transfers? | Transfer the action expert only. |
| [006](006-action-expert-scaling.md) | How large should the action expert be? | Keep the 53M expert. |
| [007](007-decoder-robustness.md) | Why was decoder training unstable? | Cache one broad teacher dataset and reuse it. |
| [008](008-xperience-data-scaling.md) | Does diverse human data improve transfer? | Yes to 1,000 clips; 10,000 clips needs more optimization. |
| [009](009-nested-human-data-scaling.md) | Does human validation improve with nested data at matched exposure? | Yes, monotonically with diminishing returns. |
| [010](010-position-shift-robustness.md) | Does nominal decoder robustness survive a one-axis position shift? | No; expand robot training and decoder-rollout coverage. |

## Locked Architecture

```text
frozen foundational VLM during egocentric pretraining
                         +
pretrained embodiment-agnostic action expert
                         +
fresh robot-domain VLM adapters/state path
                         +
small decoder trained per robot
```

The next exploration should add more diverse sessions beyond the current five
training episodes, rather than repeat these clips for additional epochs.
