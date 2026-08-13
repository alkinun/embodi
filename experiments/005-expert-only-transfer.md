# 005: expert-only transfer

## question

is useful xperience transfer in the vlm adapters or the action expert?

## setup

- baseline: fresh robot adapters and expert.
- full transfer: xperience adapters and expert.
- backbone only: xperience adapters and a fresh expert.
- expert only: fresh robot adapters and the xperience expert.

## result

seed 3991 used 50 paired scenes.

| initialization | success |
| --- | ---: |
| baseline | 56% |
| full transfer | 22% |
| backbone only | 40% |
| expert only | 88% |

seed 3992 reached 60% expert-only, 42% baseline, and 28% full transfer.
aggregate expert-only deterministic-ik success was 74%.

the best learned expert-only controller reached 83/100. its baseline reached
56/100. expert transfer won 33 scenes uniquely. baseline won 6.

## finding

the action expert stores the reusable knowledge. human vlm adapters hurt robot
vision transfer.

## decision

- freeze the vlm during human pretraining.
- transfer only the action expert.
- start robot adapters and state paths fresh.
- train one small decoder for each robot.
