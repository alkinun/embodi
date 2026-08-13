# 006: Action-Expert Scaling

## Question

What action-expert size gives the best egocentric-transfer result under a fixed,
short training budget?

## Setup

- One screening seed.
- Same frozen VLM and fixed 100 Xperience clips.
- 1,000 human-pretraining updates.
- 1,500 SO-101 updates on 100 demonstrations.
- Matched scratch and expert-transfer runs.
- 30 paired simulator scenes with deterministic IK.

| Size | Width | Layers | Heads | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Small | 256 | 8 | 8 | 9.24M |
| Medium | 512 | 12 | 8 | 53.12M |
| Large | 640 | 16 | 10 | 109.05M |

## Results

| Size | Human validation loss | Robot scratch loss | Robot transfer loss | Scratch success | Transfer success |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9.24M | `0.3347` | `0.001021` | `0.000661` | 4/30 (13%) | 11/30 (37%) |
| 53.12M | `0.2127` | `0.002908` | `0.000486` | 6/30 (20%) | 19/30 (63%) |
| 109.05M | `0.2081` | `0.001358` | `0.000477` | 3/30 (10%) | 7/30 (23%) |

Egocentric expert pretraining improved success at every size. Human validation
loss saturated near 53M parameters: doubling to 109M improved it by only 2.2%.
The 109M expert was also under-optimized and performed poorly in closed loop
under the fixed update budget.

## Decision

Keep the 53M expert (`width=512`, `layers=12`, `heads=8`) as the default. It is
the best transfer model in this screen and the best compute/performance tradeoff.

This was a bounded one-seed screen, not a scaling-law result. Do not spend more
compute on the 109M model until human data and training budgets are substantially
larger.
