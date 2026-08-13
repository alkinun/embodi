# 009: nested human-data scaling

## question

does more unique human data help when exposure stays fixed?

## setup

- human-only experiment.
- frozen vlm and fixed 53.12m expert.
- nested budgets: `100 subset 1,000 subset 10,000`.
- five training episodes and two validation episodes.
- one fixed 1,000-clip validation set.
- motion-balanced ordering without replacement.
- batch size 1 and gradient accumulation 32.
- 32 presentations per unique clip.
- learning rate `1e-4`, 10% warmup, cosine decay, seed 80000.

## result

| unique clips | updates | presentations | passes | train loss | validation loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 100 | 3,200 | 32 | `0.7217` | `0.8715` |
| 1,000 | 1,000 | 32,000 | 32 | `0.0983` | `0.2821` |
| 10,000 | 10,000 | 320,000 | 32 | `0.0436` | `0.2637` |

the 10,000-clip run reached `0.2611` at step 7,500. it ended at `0.2637`.

validation improved 67.6% from 100 to 1,000 clips. it improved 6.5% from 1,000
to 10,000 clips.

## finding

more unique data helps with diminishing returns. this is one seed. it does not
measure robot transfer.

## decision

add sessions and tasks. do not add more overlapping windows from these five
episodes.

results are in `reports/xperience-nested-data-scaling.json`.
