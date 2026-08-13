# 009: Nested Human-Data Scaling

## Question

Does the fixed 53M action expert improve as human pretraining data increases
when dataset composition and effective exposure are controlled?

## Setup

- Human-only experiment; no robot adaptation or control evaluation.
- Frozen foundational VLM and fixed 53.12M action expert.
- Deterministic nested datasets: `100 subset 1,000 subset 10,000` clips.
- Five fixed training episodes and two fixed held-out validation episodes.
- The same 1,000 validation clips for every condition.
- Motion-balanced ordering with no replacement within each dataset budget.
- Batch size 1, gradient accumulation 32.
- Updates equal unique clip count: 100, 1,000, and 10,000.
- Each condition therefore receives 32 clip presentations per unique clip.
- Learning rate `1e-4`, 10% warmup, cosine decay, seed 80000.
- Validation loss averages 100 fixed batches.

## Results

| Unique clips | Updates | Clip presentations | Effective passes | Final train loss | Final validation loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 100 | 3,200 | 32 | `0.7217` | `0.8715` |
| 1,000 | 1,000 | 32,000 | 32 | `0.0983` | `0.2821` |
| 10,000 | 10,000 | 320,000 | 32 | `0.0436` | `0.2637` |

The 10,000-clip run reached its best measured validation loss, `0.2611`, at
step 7,500 and ended at `0.2637`. Both are better than the 1,000-clip result,
but the gain from 1,000 to 10,000 clips is small compared with the gain from 100
to 1,000 clips.

Relative final validation improvements were 67.6% from 100 to 1,000 clips and
6.5% from 1,000 to 10,000 clips. The curve is monotonic under matched exposure,
with strongly diminishing returns on the current five-episode training pool.

## Decision

The dataset pipeline now supports a valid nested scaling comparison. More unique
human data improves held-out prediction, but future scaling should add sessions
and tasks rather than draw more overlapping windows from these same episodes.
This is a one-seed result and does not measure robot transfer.

Machine-readable results are in `reports/xperience-nested-data-scaling.json`.
