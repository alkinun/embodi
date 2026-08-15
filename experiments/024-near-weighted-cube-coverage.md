# 024: near-weighted cube coverage

## question

can reallocating robot demonstrations toward near cube positions correct the
selected policy's pre-lift approach failures without destroying nominal and far
control?

## setup

- generate 105 successful top-camera demonstrations with seed 11000.
- allocate 53 episodes to near positions `[0.24, 0.28]`, 26 to nominal
  `[0.28, 0.32]`, and 26 to far `[0.32, 0.36]`.
- stratify the final five validation episodes as 3 near, 1 nominal, and 1 far,
  leaving a 50/25/25 training split.
- compare against the center-weighted dataset's 25/50/25 near/nominal/far
  training split at the same total data budget.
- train with expert-only initialization, model seed 3993, loader seed 3992,
  strict determinism, and the experiment 020 optimizer protocol.
- save checkpoints at steps 700 through 1,300 in 100-step increments.
- screen the seven checkpoints on ten paired scenes per range with seed 24000,
  deterministic ik, horizon 16, and a 500-step limit.
- select by near success; ties prefer higher total success, then the earlier
  checkpoint.
- confirm the selected near-weighted checkpoint against the fixed center-weighted
  step-1,100 baseline on 50 paired scenes per range with seed 24500.

the primary endpoint is paired near success. nominal plus far success is the
retention endpoint. adopt the intervention only if it improves near by at least
5/50 successes and retains at least 85% of the baseline's combined nominal and
far successes.

## result

screen successes over ten scenes per range were:

| step | near | nominal | far | total |
| ---: | ---: | ---: | ---: | ---: |
| 700 | 3 | 2 | 1 | 6/30 |
| 800 | 0 | 0 | 0 | 0/30 |
| 900 | 3 | 4 | 6 | 13/30 |
| 1,000 | 0 | 0 | 0 | 0/30 |
| 1,100 | 0 | 0 | 0 | 0/30 |
| 1,200 | 2 | 2 | 1 | 5/30 |
| 1,300 | 1 | 0 | 1 | 2/30 |

steps 700 and 900 tied on near success. the pre-registered total-success
tiebreak selected step 900. confirmation on new scenes produced:

| dataset | near | nominal | far | nominal + far | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| center-weighted | 9 | 33 | 25 | 58/100 | 67/150 |
| near-weighted | 15 | 25 | 16 | 41/100 | 56/150 |

near weighting gained 6/50 near successes, meeting the 5/50 gain threshold,
but retained only 70.7% of baseline nominal-plus-far success, below the 85%
requirement. paired baseline-only versus near-weighted-only successes were 7 vs
13 near (`p=0.263`), 17 vs 9 nominal (`p=0.169`), and 19 vs 10 far (`p=0.136`).
combined nominal and far discordances were 36 vs 19 (`p=0.0300`).

raw rollouts are in
`reports/near-weight-exp24-{near,nominal,far}-{screen,confirm}-*.json`; the
compact analysis is in `reports/near-weight-exp24-summary.json`.

## finding

near-weighted data moves capability in the intended direction, showing that the
diagnosed near approach weakness is data-sensitive. under a fixed 105-demo
budget, however, doubling near coverage by removing nominal demonstrations
causes unacceptable nominal and far forgetting. the result rejects simple
reallocation, not near-targeted data itself.

## decision

- reject the 50/25/25 near/nominal/far training allocation as the default.
- retain the center-weighted step-1,100 policy.
- test additive near data or balanced replay rather than taking demonstrations
  away from nominal coverage.
