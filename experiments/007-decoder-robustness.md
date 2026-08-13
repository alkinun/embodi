# 007: Decoder Robustness

## Question

Did learned-decoder variance come from optimizer randomness or from recollecting
small policy-rollout datasets?

## Setup

- Fixed 53M expert-only SO-101 core.
- One cached deterministic-IK teacher dataset.
- 40 rollout episodes, 35 train and 5 held out.
- 1,294 policy-conditioned canonical/native chunks.
- Teacher success: 35/40; all held-out episodes succeeded.
- Same decoder initialization, cached tensors, split, architecture, and 2,500
  updates.
- Training seeds 1, 2, and 3.
- 100 paired unseen simulator scenes.

## Result

| Decoder seed | Validation loss | Success | Lifts |
| ---: | ---: | ---: | ---: |
| 1 | `0.03571` | 89/100 | 92/100 |
| 2 | `0.03455` | 89/100 | 90/100 |
| 3 | `0.03532` | 90/100 | 91/100 |

Mean success was 89.3%, with 0.47 percentage-point population standard
deviation and a one-point range. All three decoders succeeded on 72 common
scenes; every scene was solved by at least one decoder.

## Finding

Optimizer/minibatch randomness is small when the decoder dataset is fixed.
Earlier instability came primarily from recollecting small rollout datasets
whose teacher success and state coverage varied substantially.

## Decision

- Cache decoder-distillation tensors with source-core and embodiment hashes.
- Use at least 40 teacher episodes with an episode-disjoint validation split.
- Reuse the exact cache across decoder seeds and model comparisons.
- Keep closed-loop success as the final metric.
