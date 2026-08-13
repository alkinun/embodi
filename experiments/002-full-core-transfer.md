# 002: full-core transfer

## question

does transferring xperience vlm adapters and the action expert help so-101?

## setup

- baseline: fresh adapters and a fresh action expert.
- full transfer: xperience adapters and action expert.
- robot data, decoder initialization, schedules, and seeds stayed fixed.

## result

full transfer reduced offline loss. it did not improve task success across three
2,500-step deterministic ik runs.

| model | success |
| --- | ---: |
| baseline | 57/150 (38.0%) |
| full transfer | 50/150 (33.3%) |

later tests found 22% and 28% for full transfer. baselines reached 56% and 42%.

## finding

human vlm adapters hurt robot transfer. offline loss does not predict control.

## decision

do not transfer the full human core. use closed-loop success for selection.
