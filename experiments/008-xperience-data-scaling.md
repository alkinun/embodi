# 008: Xperience Data Scaling

## Question

Does scaling frozen-VLM action-expert pretraining across diverse Xperience
episodes improve held-out human loss and SO-101 transfer?

## Setup

- Eight cached episodes from eight sessions at revision
  `ce943cf271a758b60240084892d05cf6dc12dd90`.
- 18,574 valid anchors; one episode had no valid anchors because its synchronized
  labels were non-finite.
- Whole-episode split with fixed held-out episodes for every budget.
- Globally motion-balanced training samples at 100, 1,000, and 10,000 clips.
- Frozen foundational VLM and fixed 53.12M action expert.
- 1,000 human-pretraining updates for every data budget.
- Expert-only transfer followed by 1,500 SO-101 regression updates on 100
  demonstrations.
- 30 paired simulator scenes, seed 63000, deterministic IK, execution horizon 8.

## Results

| Human clips | Human validation loss | Robot validation loss | Lift | Success |
| ---: | ---: | ---: | ---: | ---: |
| 100 | `0.4380` | `0.000502` | 14/30 (47%) | 11/30 (37%) |
| 1,000 | `0.2217` | `0.000375` | 18/30 (60%) | 15/30 (50%) |
| 10,000 | `0.1950` | `0.000545` | 14/30 (47%) | 13/30 (43%) |

Human held-out loss improved monotonically, with a large gain from 100 to 1,000
clips and a smaller gain from 1,000 to 10,000. Closed-loop transfer peaked at
1,000 clips under the fixed update budget. The 10,000-clip model saw only 3.2
effective passes over its sampled data, compared with 32 and 320 passes for the
1,000- and 100-clip models, so this is an optimization-limited screen rather
than evidence that additional data is harmful.

An initial robot-transfer run accidentally omitted the required corrected image
preprocessing and stored `image_do_rescale=true`; all three policies scored
0/30. Those runs are invalid. `configs/so101-exp0-top.json` now locks
`image_do_rescale=false` so the benchmark contract no longer depends on a CLI
flag.

## Decision

Use the 1,000-clip expert as the best checkpoint under the fixed 1,000-update
budget. Keep the 10,000-clip expert for a future matched-epoch or longer-update
study before making a final data-scaling conclusion.

Canonical control results are in
`reports/so101-xperience-multi8-data-scaling-rescale-h8-30.json`.
