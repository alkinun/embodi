# 019: deterministic model-seed screen

## question

can another deterministic robot-side initialization recover strong
center-weighted control, and do useful policies appear before the final
checkpoint?

## setup

- keep the center-weighted dataset and loader seed 3992 fixed.
- use strict deterministic training for every new run.
- reuse experiment 018 model seed 3992 at step 2,500.
- train model seeds 3991 and 3993 for 2,500 updates with checkpoints every 500
  updates.
- keep the expert source, config, optimizer, batch size 32, accumulation 1,
  workers 0, 250-step warmup, and validation split fixed.
- screen steps 500, 1,000, 1,500, 2,000, and 2,500 for seeds 3991 and 3993,
  plus the seed-3992 final.
- evaluate ten paired scenes in each near, nominal, and far range with seed
  19000, deterministic ik, horizon 16, and a 500-step limit.
- rank checkpoints by total success over the 30 scenes; validation loss is not
  a selection metric.

a candidate advances only if it reaches at least 10/30 total successes and at
least one success in every range. ties prefer the earlier checkpoint. an
advancing candidate is confirmed on 50 scenes per range with new evaluation
seed 19500 and the otherwise identical protocol. if no candidate qualifies,
the experiment ends after the screen.

this is a deterministic initialization screen, not an estimate of expected
performance over random seeds.

## result

screen successes over ten scenes per range were:

| model seed | step | near | nominal | far | total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3991 | 500 | 0 | 0 | 0 | 0/30 |
| 3991 | 1,000 | 1 | 3 | 2 | 6/30 |
| 3991 | 1,500 | 0 | 0 | 0 | 0/30 |
| 3991 | 2,000 | 0 | 2 | 2 | 4/30 |
| 3991 | 2,500 | 0 | 4 | 3 | 7/30 |
| 3992 | 2,500 | 2 | 1 | 0 | 3/30 |
| 3993 | 500 | 1 | 1 | 1 | 3/30 |
| 3993 | 1,000 | 2 | 4 | 4 | 10/30 |
| 3993 | 1,500 | 0 | 0 | 0 | 0/30 |
| 3993 | 2,000 | 2 | 0 | 0 | 2/30 |
| 3993 | 2,500 | 2 | 2 | 1 | 5/30 |

only model seed 3993 at step 1,000 met the preregistered promotion rule.
confirmation on new scenes, with the fixed deterministic seed-3992 endpoint as
a paired baseline, produced:

| checkpoint | near | nominal | far | total |
| --- | ---: | ---: | ---: | ---: |
| seed 3993, step 1,000 | 8 | 18 | 13 | 39/150 |
| seed 3992, step 2,500 | 2 | 9 | 12 | 23/150 |

paired candidate-only versus baseline-only successes were 7 vs 1 near, 11 vs
2 nominal, and 9 vs 8 far. across all ranges they were 27 vs 11 (exact mcnemar
p=0.0139). the range-level p values were 0.0703, 0.0225, and 1.0 respectively.

control changed non-monotonically while validation improved. for model seed
3993, validation regression loss fell from 0.002188 at step 1,000 to 0.001682 at
step 1,500 while screen success fell from 10/30 to 0/30.

raw screen and confirmation rollouts are in
`reports/det-seed-exp19-*-{screen,confirm,confirm-baseline}-*.json`; the compact
analysis is in `reports/det-seed-exp19-summary.json`.

## finding

deterministic initialization seed 3993 recovers a reproducible, broader policy
at step 1,000, improving paired near and nominal control over the deterministic
seed-3992 endpoint. it does not recover the 74-91/150 behavior of the earlier
non-deterministic policies.

more training can abruptly destroy control even as held-out imitation loss
improves. initialization was therefore only part of the instability: checkpoint
time is another causal factor, and a final-only training protocol systematically
misses useful deterministic policies.

## decision

- use model seed 3993 at step 1,000 as the current deterministic center-weighted
  core.
- keep closed-loop checkpoint selection separate from the validation set.
- narrow the next experiment around the step-1,000 control peak and reduce
  intermediate-checkpoint storage before increasing checkpoint frequency.
- do not infer a general model-seed ranking from this three-seed screen.
