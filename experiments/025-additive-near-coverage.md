# 025: additive near coverage

## question

can adding near demonstrations improve near approach control while preserving
the nominal and far demonstration counts that experiment 024 removed?

## setup

- generate 130 successful top-camera demonstrations with seed 11000.
- allocate 52 episodes to near positions `[0.24, 0.28]`, 52 to nominal
  `[0.28, 0.32]`, and 26 to far `[0.32, 0.36]`.
- stratify the final five validation episodes as 2 near, 2 nominal, and 1 far,
  leaving a 50/50/25 near/nominal/far training split.
- compare against the center-weighted dataset's 25/50/25 training split. this
  preserves nominal and far counts while adding 25 near demonstrations.
- train with expert-only initialization, model seed 3993, loader seed 3992,
  strict determinism, and the experiment 020 optimizer protocol.
- save checkpoints at steps 700 through 1,300 in 100-step increments.
- screen the seven checkpoints on ten paired scenes per range with seed 25000,
  deterministic ik, horizon 16, and a 500-step limit.
- select by near success; ties prefer higher total success, then the earlier
  checkpoint.
- confirm the selected additive checkpoint against the fixed center-weighted
  step-1,100 baseline on 50 paired scenes per range with seed 25500.

the primary endpoint is paired near success. nominal plus far success is the
retention endpoint. adopt the intervention only if it improves near by at least
5/50 successes and retains at least 85% of the baseline's combined nominal and
far successes.

## result

screen successes over ten scenes per range were:

| step | near | nominal | far | total |
| ---: | ---: | ---: | ---: | ---: |
| 700 | 0 | 0 | 0 | 0/30 |
| 800 | 0 | 0 | 0 | 0/30 |
| 900 | 0 | 0 | 0 | 0/30 |
| 1,000 | 5 | 2 | 6 | 13/30 |
| 1,100 | 0 | 0 | 1 | 1/30 |
| 1,200 | 2 | 5 | 4 | 11/30 |
| 1,300 | 1 | 0 | 2 | 3/30 |

the pre-registered ranking selected step 1,000. confirmation on new scenes
produced:

| dataset | near | nominal | far | nominal + far | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| center-weighted | 11 | 26 | 25 | 51/100 | 62/150 |
| additive near | 13 | 26 | 37 | 63/100 | 76/150 |

additive data gained only 2/50 near successes, below the required 5/50, while
retaining 123.5% of baseline nominal-plus-far success. paired baseline-only
versus additive-only successes were 9 vs 11 near (`p=0.824`), 9 vs 9 nominal
(`p=1.0`), and 8 vs 20 far (`p=0.0357`). the aggregate was 26 vs 40
(`p=0.109`).

raw rollouts are in
`reports/additive-near-exp25-{near,nominal,far}-{screen,confirm}-*.json`; the
compact analysis is in `reports/additive-near-exp25-summary.json`.

## finding

adding near demonstrations avoids the nominal/far forgetting caused by fixed-
budget reallocation, but it does not meet the intended near-improvement target.
the significant far gain and 14-success aggregate improvement are unexpected,
post-hoc evidence that the additive dataset may support a stronger broad policy.
they require independent confirmation before replacing the selected controller.

## decision

- reject additive coverage as a confirmed near-failure fix.
- retain center-weighted step 1,100 under the pre-registered decision rule.
- independently confirm additive step 1,000 as a broad-policy candidate on new
  paired scenes before considering replacement.
