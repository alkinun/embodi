# 010: position-shift robustness

## question

does nominal success survive a shift in the cube x position?

## setup

- fixed 53M expert-only so-101 core from training seed 3991.
- fixed decoders from seeds 1, 2, and 3.
- 50 paired scenes per range, seed 65000.
- nominal x: `[0.28, 0.32]` m.
- near x: `[0.24, 0.28]` m.
- far x: `[0.32, 0.36]` m.
- fixed y: `[-0.025, 0.025]` m.
- fixed target, camera, task, physics, 500-step budget, and horizon 8.
- deterministic ik as a decoder control.

the evaluator records each range and initial cube position.

## result

learned-decoder success:

| cube range | decoder 1 | decoder 2 | decoder 3 | mean | population sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| nominal | 45/50 | 44/50 | 43/50 | 88% | 1.6 pp |
| near | 20/50 | 13/50 | 6/50 | 26% | 11.4 pp |
| far | 6/50 | 5/50 | 13/50 | 16% | 7.1 pp |

decoder 1 had 27 nominal-only and 2 near-only wins (`p=1.62e-6`). it had 39
nominal-only and 0 far-only wins (`p=3.64e-12`). tests used exact mcnemar.

deterministic ik control:

| cube range | success | lifts |
| --- | ---: | ---: |
| nominal | 41/50 | 41/50 |
| near | 12/50 | 12/50 |
| far | 19/50 | 28/50 |

near failure remains with deterministic ik. far ik improves decoder 1 from 12%
to 38%.
the canonical policy still stays below its 82% nominal result. most failures
happen before lift.

## finding

decoder stability is local to the nominal range. the canonical policy is the
main limit. decoder coverage adds far-side error.

this study uses one core seed and one scene axis. decoder seeds are not core
replications. this result does not map the full workspace.

## decision

- do not call nominal success workspace robustness.
- expand demonstrations across x and y.
- collect decoder rollouts from the same wider range.
- repeat with independent core seeds and a held-out grid.
- keep deterministic ik as a control.

raw results:

- `reports/so101-decoder-robustness-reproduction-h8-100.json`
- `reports/so101-position-shift-near-h8-3seed-50.json`
- `reports/so101-position-shift-far-h8-3seed-50.json`
- `reports/so101-position-shift-nominal-ik-h8-50.json`
- `reports/so101-position-shift-near-ik-h8-50.json`
- `reports/so101-position-shift-far-ik-h8-50.json`
- `reports/so101-position-shift-summary.json`
