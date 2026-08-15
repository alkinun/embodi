# 023: deterministic expert transfer

## question

does xperience action-expert transfer still improve robot control under strict
deterministic training and matched checkpoint selection?

## setup

- compare expert-only transfer against a fully fresh robot core.
- use the center-weighted dataset, model seed 3993, loader seed 3992, strict
  determinism, and the experiment 020 training protocol.
- reuse transferred checkpoints at steps 700 through 1,300.
- train one scratch run and save the same 100-step checkpoint grid.
- screen both conditions on ten paired scenes per range with seed 23000.
- select the best checkpoint within each condition by total success over 30
  scenes; ties prefer the earlier checkpoint.
- confirm both selected checkpoints on 50 paired scenes per range with seed
  23500, deterministic ik, horizon 16, and a 500-step limit.

the primary endpoint is paired total success. range-level success and lift are
secondary. both conditions receive identical checkpoint-selection freedom.

## result

screen successes over ten scenes per range were:

| initialization | step | near | nominal | far | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| expert transfer | 700 | 1 | 4 | 2 | 7/30 |
| expert transfer | 800 | 2 | 5 | 5 | 12/30 |
| expert transfer | 900 | 0 | 0 | 0 | 0/30 |
| expert transfer | 1,000 | 2 | 5 | 3 | 10/30 |
| expert transfer | 1,100 | 2 | 8 | 4 | 14/30 |
| expert transfer | 1,200 | 2 | 3 | 1 | 6/30 |
| expert transfer | 1,300 | 2 | 1 | 0 | 3/30 |
| scratch | 700 | 0 | 0 | 0 | 0/30 |
| scratch | 800 | 3 | 0 | 1 | 4/30 |
| scratch | 900 | 1 | 0 | 1 | 2/30 |
| scratch | 1,000 | 1 | 0 | 0 | 1/30 |
| scratch | 1,100 | 1 | 0 | 1 | 2/30 |
| scratch | 1,200 | 1 | 2 | 1 | 4/30 |
| scratch | 1,300 | 0 | 0 | 1 | 1/30 |

the pre-registered ranking selected transferred step 1,100 and scratch step 800;
the scratch tie at step 1,200 resolved to the earlier checkpoint. confirmation
on new scenes produced:

| initialization | near | nominal | far | total |
| --- | ---: | ---: | ---: | ---: |
| expert transfer | 7 | 26 | 30 | 63/150 |
| scratch | 9 | 14 | 3 | 26/150 |

paired transfer-only versus scratch-only successes were 6 vs 8 near, 17 vs 5
nominal, and 29 vs 2 far. across all ranges they were 52 vs 15 (exact mcnemar
`p=6.46e-6`). range-level p values were 0.791, 0.0169, and `4.63e-7`.

raw rollouts are in
`reports/det-transfer-exp23-{near,nominal,far}-{screen,confirm}-*.json`; the
compact analysis is in `reports/det-transfer-exp23-summary.json`.

## finding

action-expert transfer survives strict deterministic replication and matched
checkpoint selection. it provides a 37-success aggregate advantage and is
especially important for far-range generalization. scratch is not uniformly
inferior: it slightly improves near success and lifts, reinforcing that the
remaining near weakness is distribution-specific rather than a universal
benefit of human pretraining.

## decision

- retain action-expert transfer as the default initialization.
- do not interpret expert transfer as a substitute for near-targeted robot data.
- proceed with a near-position data intervention and measure nominal/far
  retention.
