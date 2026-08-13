# 012: checkpoint selection

## question

did experiment 011 pass through a better control checkpoint before step 2,500?

## setup

- reuse the nominal and wide runs from experiment 011.
- evaluate steps 500, 1,000, 1,500, 2,000, and 2,500.
- use deterministic ik and execution horizon 8.
- use 10 paired scenes from each near, nominal, and far range.
- use evaluation seed 12000.
- hold the model, data, and optimizer path fixed within each run.

this is a 30-scene checkpoint screen. it is not a final success estimate.

## result

successes across all 30 scenes:

| step | nominal training | wide training |
| ---: | ---: | ---: |
| 500 | 0 | 0 |
| 1,000 | 0 | 0 |
| 1,500 | 0 | 0 |
| 2,000 | 2 | 0 |
| 2,500 | 1 | 1 |

the nominal step-2,000 checkpoint solved one near and one nominal scene. the
wide step-2,500 checkpoint solved one near scene. no checkpoint solved a far
scene.

## finding

experiment 011 did not hide a useful early control peak. offline validation
loss improved while closed-loop success stayed near zero.

the screen is small. it can reject a large hidden peak, but it cannot rank
checkpoints that differ by one or two successes.

## decision

- do not recover experiment 011 with checkpoint selection.
- keep closed-loop screening during training.
- stop runs early only after a larger paired control screen confirms a peak.
- test inference horizon next because it needs no retraining.

raw results:

- `reports/checkpoint-exp12-near-screen-ik-h8-10model-10.json`
- `reports/checkpoint-exp12-nominal-screen-ik-h8-10model-10.json`
- `reports/checkpoint-exp12-far-screen-ik-h8-10model-10.json`
- `reports/checkpoint-exp12-summary.json`
