# 010: Position-Shift Robustness

## Question

Does the 89% nominal success of the fixed expert-only core and its three robust
decoders survive a disjoint shift in the cube's initial x position?

## Setup

- Fixed 53M expert-only SO-101 core from training seed 3991.
- Fixed decoders from optimizer seeds 1, 2, and 3; no retraining.
- Same 50 paired random quantiles, seed 65000, for each position range.
- Nominal x range: `[0.28, 0.32]` m.
- Near x range: `[0.24, 0.28]` m.
- Far x range: `[0.32, 0.36]` m.
- The y range remained `[-0.025, 0.025]` m.
- Target, camera, task, physics, 500-step budget, and horizon 8 remained fixed.
- Deterministic IK evaluated the same core on the same scenes as a decoder
  control.

The evaluator now records the configured ranges and each episode's initial cube
position. Its default ranges are unchanged.

## Results

Learned-decoder success:

| Cube range | Decoder 1 | Decoder 2 | Decoder 3 | Mean | Population SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | 45/50 | 44/50 | 43/50 | 88% | 1.6 pp |
| Near | 20/50 | 13/50 | 6/50 | 26% | 11.4 pp |
| Far | 6/50 | 5/50 | 13/50 | 16% | 7.1 pp |

For decoder 1, exact paired tests against nominal found 27 nominal-only versus
2 near-only successes (`p=1.62e-6`), and 39 nominal-only versus 0 far-only
successes (`p=3.64e-12`). These are two-sided exact McNemar tests.

Deterministic-IK control on the same fixed core:

| Cube range | Success | Lift |
| --- | ---: | ---: |
| Nominal | 41/50 | 41/50 |
| Near | 12/50 | 12/50 |
| Far | 19/50 | 28/50 |

The near collapse persists with exact IK, so it is not explained by the learned
robot decoder. Far-shift IK improves over decoder 1 from 12% to 38%, indicating
that decoder extrapolation adds error there, but the canonical policy still
remains far below its 82% nominal IK result. Most learned-controller failures
occur before a successful lift.

## Finding

The previous decoder-optimizer robustness result is local to the narrow nominal
scene distribution. Stable decoder training does not imply position robustness.
The principal limitation is canonical-policy generalization, with an additional
learned-decoder coverage problem on the far side.

This experiment varies only one scene axis and uses one robot-core training
seed. Decoder seeds are not independent core-training replications, and the
shifted ranges touch the nominal boundary. The result therefore establishes a
clear local distribution-shift failure, not a complete workspace boundary.

## Decision

- Do not report nominal simulator success as workspace robustness.
- Expand robot demonstrations across x and y before changing architecture.
- Collect decoder distillation rollouts from the expanded scene distribution.
- Repeat with independent core-training seeds and a held-out two-dimensional
  position grid.
- Keep deterministic IK as a control to localize policy versus decoder errors.

Raw results are in:

- `reports/so101-decoder-robustness-reproduction-h8-100.json`
- `reports/so101-position-shift-near-h8-3seed-50.json`
- `reports/so101-position-shift-far-h8-3seed-50.json`
- `reports/so101-position-shift-nominal-ik-h8-50.json`
- `reports/so101-position-shift-near-ik-h8-50.json`
- `reports/so101-position-shift-far-ik-h8-50.json`
- `reports/so101-position-shift-summary.json`
