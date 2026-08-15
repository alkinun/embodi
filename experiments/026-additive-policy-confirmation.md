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

successes over 100 paired scenes per range were:

| policy | near | nominal | far | total |
| --- | ---: | ---: | ---: | ---: |
| center-weighted step 1,100 | 25 | 56 | 59 | 140/300 |
| additive-near step 1,000 | 28 | 52 | 74 | 154/300 |

paired baseline-only versus candidate-only successes were 15 vs 18 near
(`p=0.728`), 27 vs 23 nominal (`p=0.672`), and 15 vs 30 far (`p=0.0357`).
across all ranges they were 57 vs 71 (`p=0.250`). the candidate's largest
range regression was 4/100 nominal, inside the 5/100 retention margin, but the
aggregate exact test did not meet `p < 0.05`.

raw rollouts are in
`reports/additive-confirm-exp26-{near,nominal,far}-ik-h16-2model-100.json`; the
compact analysis is in `reports/additive-confirm-exp26-summary.json`.

## finding

the additive policy's far-range advantage independently replicates, reaching a
15/100 gain with paired significance. its 14/300 aggregate advantage is not
statistically confirmed, so the evidence supports a far specialist rather than
a general replacement. additive near data did not reliably solve near control.

## decision

- retain center-weighted step 1,100 as the selected general policy.
- keep additive-near step 1,000 as a confirmed far-range specialist.
- do not claim aggregate superiority or a near-range fix.
