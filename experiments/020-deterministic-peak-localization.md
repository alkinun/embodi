# 020: deterministic peak localization

## question

is the deterministic seed-3993 control peak centered at step 1,000, or does a
nearby checkpoint improve broad control?

## setup

- repeat the exact deterministic model-seed-3993, loader-seed-3992 training
  protocol from experiment 019 for 2,500 updates.
- save compact inference checkpoints only at steps 700, 800, 900, 1,000, 1,100,
  1,200, and 1,300; retain a full final checkpoint.
- verify that the repeated step-1,000 and final model states exactly match
  experiment 019 before evaluating.
- screen all seven checkpoints on ten paired scenes per near, nominal, and far
  range with seed 20000, deterministic ik, horizon 16, and a 500-step limit.
- rank by total success over 30 scenes; ties prefer the earlier checkpoint.

a checkpoint other than step 1,000 replaces the current selection only if it
beats step 1,000 by at least 3/30 successes and succeeds in every range. such a
candidate is confirmed against step 1,000 on 50 paired scenes per range with
new seed 20500. otherwise retain step 1,000 without confirmation.

## result

the repeated step-1,000 and final core, state-adapter, and decoder hashes exactly
matched experiment 019. changing checkpoint cadence did not change training.

screen successes over ten scenes per range were:

| step | near | nominal | far | total |
| ---: | ---: | ---: | ---: | ---: |
| 700 | 1 | 3 | 1 | 5/30 |
| 800 | 5 | 3 | 1 | 9/30 |
| 900 | 0 | 0 | 0 | 0/30 |
| 1,000 | 1 | 4 | 2 | 7/30 |
| 1,100 | 2 | 6 | 6 | 14/30 |
| 1,200 | 1 | 1 | 0 | 2/30 |
| 1,300 | 1 | 0 | 0 | 1/30 |

step 1,100 met the replacement criterion. confirmation on new scenes produced:

| checkpoint | near | nominal | far | total |
| --- | ---: | ---: | ---: | ---: |
| step 1,000 | 7 | 15 | 17 | 39/150 |
| step 1,100 | 8 | 27 | 31 | 66/150 |

paired step-1,100-only versus step-1,000-only successes were 5 vs 4 near, 20
vs 8 nominal, and 22 vs 8 far. across all ranges they were 47 vs 20 (exact
mcnemar p=0.00131). range-level p values were 1.0, 0.0357, and 0.0161.

raw rollouts are in
`reports/det-peak-exp20-{near,nominal,far}-{screen,confirm}-*.json`; the compact
analysis is in `reports/det-peak-exp20-summary.json`.

## finding

the deterministic control peak is at step 1,100 on the tested 100-step grid.
it materially improves nominal and far control over step 1,000 and is now the
strongest reproducible deterministic center-weighted policy.

the peak is narrow and non-monotonic: step 900 has no screen successes, step
1,100 has 14/30, and step 1,200 falls to 2/30. useful control can appear and
disappear within 100 optimizer updates even though training remains numerically
stable. near control remains the primary weakness at 8/50.

## decision

- select deterministic model seed 3993 at step 1,100 as the current core.
- do not use validation loss to rank local checkpoints.
- test weight interpolation around steps 1,000, 1,100, and 1,200 before spending
  more training compute; interpolation can reveal whether the peak occupies a
  connected weight-space region and may smooth the sharp temporal instability.
- keep near-range success as the primary robustness target.
