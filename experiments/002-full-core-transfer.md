# 002: Full-Core Transfer

## Question

Does transferring both Xperience-trained VLM adapters and the action expert
improve SO-101 learning?

## Comparison

- Baseline: foundational VLM with fresh adapters and fresh action expert.
- Full generalist: Xperience-trained VLM adapters and action expert.
- Same robot data, decoder initialization, schedules, and seeds.

## Result

Full transfer reduced offline canonical loss but did not robustly improve task
success. Across three 2,500-step deterministic-IK runs:

| Model | Success |
| --- | ---: |
| Baseline | 57/150 (38.0%) |
| Full generalist | 50/150 (33.3%) |

Later matched component tests were worse: full generalist reached 22% and 28%
where the corresponding baselines reached 56% and 42%.

## Decision

Do not transfer the complete human-adapted core into the robot domain. Offline
canonical loss is not a sufficient model-selection metric.
