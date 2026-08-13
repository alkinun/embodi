# 011: robot position coverage

## question

do wider robot demonstrations fix the x-shift failure from experiment 010?

## setup

- 105 successful demonstrations per condition.
- nominal x range: `[0.28, 0.32]` m.
- wide x range: `[0.24, 0.36]` m.
- fixed y range: `[-0.025, 0.025]` m.
- generator seed 11000 and top camera only.
- 24,405 nominal frames and 24,172 wide frames.
- the same 1,000-clip human expert checkpoint.
- core seed 3991, 2,500 updates, and effective batch size 32.
- fixed config, split rule, schedule, and deterministic-ik evaluation.
- 50 paired scenes per near, nominal, and far range. evaluation seed 11500.

the oracle accepted 105 nominal episodes in 108 attempts. it accepted 105 wide
episodes in 113 attempts. oracle filtering changed some accepted scenes. the
datasets therefore share a sampling process, but not exact episode pairs.

## result

| evaluation range | nominal training | wide training |
| --- | ---: | ---: |
| near `[0.24, 0.28]` | 2/50 | 6/50 |
| nominal `[0.28, 0.32]` | 10/50 | 1/50 |
| far `[0.32, 0.36]` | 3/50 | 1/50 |

final validation regression loss was `0.000616` for nominal training. it was
`0.000946` for wide training.

## finding

wider demonstrations alone do not fix position robustness. they add a small
near-side gain. they also cause severe nominal forgetting at this update budget.

this is a one-seed result. the new cores also underperform the older experiment
010 core. training protocol changes across experiments include a new dataset
sample and direct batch size 32. compare conditions within this experiment only.

## decision

- do not replace nominal data with uniform wide data.
- preserve nominal density when adding shifted scenes.
- test a nominal-plus-edge mixture next.
- keep deterministic ik as the first policy control.

raw results:

- `reports/coverage-exp11-near-ik-h8-2model-50.json`
- `reports/coverage-exp11-nominal-ik-h8-2model-50.json`
- `reports/coverage-exp11-far-ik-h8-2model-50.json`
- `reports/coverage-exp11-summary.json`
