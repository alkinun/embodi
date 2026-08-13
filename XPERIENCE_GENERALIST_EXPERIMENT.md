# Xperience Action-Expert Pretraining

## Locked Hypothesis

Egocentric hand-motion data should pretrain the embodiment-agnostic action
expert. The foundational VLM remains frozen during human pretraining, and each
robot receives fresh perception adapters and a small robot-specific decoder.

```text
egocentric RGB + instruction + canonical hand state
                         |
                    frozen VLM
                         |
              trainable action expert
                         |
              canonical motion trajectory
```

Robot adaptation uses:

```text
foundational VLM + fresh robot-domain adapters
                         +
       Xperience-pretrained action expert
                         +
            decoder trained per robot
```

Do not transfer human-trained VLM adapters into robot training.

## Canonical Contract

- Part: `primary_effector / pose_scalar`.
- Channels: translation `0:3`, rotation6d `3:9`, opening `9`.
- Coordinates: initial root frame.
- Translation: future position relative to the initial hand position.
- Rotation: future orientation relative to the initial hand orientation.
- Opening: absolute normalized aperture in `[0, 1]`.
- Horizon: 32 steps at 30 Hz.

SO-101 uses the same contract with the robot base as the root frame.

## Human Pretraining

- Input: stereo-left RGB, current caption, current canonical hand state.
- Target: future right-hand canonical trajectory.
- Train the action expert only.
- Freeze the complete VLM/backbone path.
- Do not create a human action decoder.
- Balance meaningful motion, small adjustment, and stationary clips.
- Keep train and validation caption segments disjoint.

The reference config is `configs/xperience-exp0.json` with
`freeze_backbone=true`.

## Robot Transfer

Load only the pretrained expert:

```bash
uv run embodi-train \
  --dataset embodi/sim-so101-pickplace \
  --dataset-root datasets/xperience-exp0-so101-v3-105 \
  --config configs/so101-exp0-top.json \
  --correct-image-rescale \
  --stage core \
  --init-core outputs/xperience-pretraining/final \
  --init-core-component expert \
  --init-embodiment outputs/so101-decoder/final \
  --output-dir outputs/so101-expert-transfer
```

The baseline uses the same foundational VLM, fresh robot adapters, fresh action
expert, decoder initialization, data, schedule, and seeds.

## Evaluation

- Use deterministic regression for the current SO-101 benchmark.
- Evaluate canonical loss and simulator success, but select conclusions by
  closed-loop success.
- Use paired simulator scenes and at least three training seeds.
- Train robot decoders on policy-predicted canonical trajectories with
  deterministic IK targets.
- Cache at least 40 teacher episodes and reuse the exact tensors across decoder
  seeds and model comparisons.
- Keep stale per-chunk native action limits disabled.

## Current Evidence

The 100-clip pilot found:

| Initialization | Deterministic-IK success, two runs |
| --- | ---: |
| Fresh action expert | 49% |
| Full human-adapted core | 25% |
| Human-pretrained action expert only | 74% |

With a fixed 40-episode decoder cache, three learned expert-only controllers
achieved 89%, 89%, and 90% success. The earlier decoder instability came from
rollout-data variation rather than optimizer randomness.

The parameter-count screen selected the existing 53M expert over 9M and 109M
alternatives. Concise records are in `experiments/`.
