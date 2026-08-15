# 026: additive policy confirmation

## question

does the post-hoc broad-control advantage of additive-near step 1,000 replicate
strongly enough to replace the selected center-weighted policy?

## setup

- fix the baseline at center-weighted model seed 3993, step 1,100.
- fix the candidate at additive-near model seed 3993, step 1,000.
- perform no further checkpoint selection or training.
- evaluate 100 paired scenes per near, nominal, and far range with seed 26000,
  deterministic ik, horizon 16, and a 500-step limit.

the primary endpoint is paired success over all 300 scenes. replace the baseline
only if the candidate has more total successes with exact mcnemar `p < 0.05`
and is no worse by more than 5/100 successes in any individual range. range-
level success, lift, and paired tests are secondary.

## result

pending.

## finding

pending.

## decision

- replace the selected policy only if both aggregate superiority and range
  retention criteria pass.
