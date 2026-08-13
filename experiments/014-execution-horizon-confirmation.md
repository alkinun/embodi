# 014: execution horizon confirmation

## question

does horizon 16 outperform horizon 8 on more held-out near-shift scenes?

## setup

- reuse the two final cores from experiment 011.
- compare execution horizons 8 and 16.
- use deterministic ik.
- use 50 paired near-shift scenes per core and horizon.
- use x range `[0.24, 0.28]` m.
- use y range `[-0.025, 0.025]` m.
- use evaluation seed 14000 and a 500-step limit.

## result

| training | horizon 8 | horizon 16 | horizon 8 only | horizon 16 only |
| --- | ---: | ---: | ---: | ---: |
| nominal | 5/50 | 10/50 | 2 | 7 |
| wide | 5/50 | 9/50 | 2 | 6 |

lifts matched successes. horizon 16 reduced elapsed evaluation time from 191 to
92 seconds for the nominal core and from 205 to 99 seconds for the wide core.

## finding

horizon 16 preserves the screen ordering on new scenes. it approximately
doubles near-shift success and halves evaluation time relative to horizon 8.
the paired differences remain small, and success stays at or below 20%.

## decision

- use execution horizon 16 for this benchmark.
- do not treat horizon tuning as a fix for position robustness.
- improve policy data and training before screening more horizons.

raw results:

- `reports/horizon-exp14-near-confirm-ik-h8-2model-50.json`
- `reports/horizon-exp14-near-confirm-ik-h16-2model-50.json`
- `reports/horizon-exp14-summary.json`
