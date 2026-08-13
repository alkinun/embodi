# 013: execution horizon screen

## question

can a different execution horizon recover near-shift control?

## setup

- reuse the two final cores from experiment 011.
- test execution horizons 1, 4, 8, 16, and 32.
- use deterministic ik.
- use 10 paired near-shift scenes per core and horizon.
- use x range `[0.24, 0.28]` m.
- use y range `[-0.025, 0.025]` m.
- use evaluation seed 13000 and a 500-step limit.

this is a screen. horizon 16 needs confirmation on new scenes.

## result

| horizon | nominal training | wide training |
| ---: | ---: | ---: |
| 1 | 0/10 | 0/10 |
| 4 | 0/10 | 0/10 |
| 8 | 2/10 | 1/10 |
| 16 | 4/10 | 2/10 |
| 32 | 2/10 | 0/10 |

horizon 16 had the best success for both cores. horizon 1 used 500 model calls
per failed episode. horizon 32 used 16 calls.

## finding

the horizon effect is non-monotonic. fast replanning does not fix a weak policy.
horizon 16 gives the best balance in this small near-shift screen.

the screen has only 10 scenes. differences of one or two successes are not a
final estimate.

## decision

- keep horizon 16 as a candidate, not a new default.
- confirm horizons 8 and 16 on 50 held-out scenes.
- keep the same cores and deterministic ik control.
- reject horizon 1 for this benchmark because it is slow and ineffective.

raw results:

- `reports/horizon-exp13-near-ik-h1-2model-10.json`
- `reports/horizon-exp13-near-ik-h4-2model-10.json`
- `reports/horizon-exp13-near-ik-h8-2model-10.json`
- `reports/horizon-exp13-near-ik-h16-2model-10.json`
- `reports/horizon-exp13-near-ik-h32-2model-10.json`
- `reports/horizon-exp13-summary.json`
