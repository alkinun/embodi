# 007: decoder robustness

## question

did decoder variance come from optimization or rollout collection?

## setup

- fixed 53m expert-only so-101 core.
- one cached deterministic-ik teacher dataset.
- 40 rollout episodes, 35 train and 5 held out.
- 1,294 policy-conditioned canonical/native chunks.
- teacher success: 35/40. all held-out episodes succeeded.
- fixed initialization, tensors, split, architecture, and 2,500 updates.
- training seeds 1, 2, and 3.
- 100 paired unseen simulator scenes.

## result

| decoder seed | validation loss | success | lifts |
| ---: | ---: | ---: | ---: |
| 1 | `0.03571` | 89/100 | 92/100 |
| 2 | `0.03455` | 89/100 | 90/100 |
| 3 | `0.03532` | 90/100 | 91/100 |

mean success was 89.3%. population standard deviation was 0.47 points. all
decoders solved 72 common scenes.

## finding

optimization variance is small with a fixed dataset. rollout coverage caused
the earlier instability.

## decision

- cache tensors with source hashes.
- use at least 40 teacher episodes.
- split validation by episode.
- reuse the exact cache across comparisons.
- keep closed-loop success as the final metric.
