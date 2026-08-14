# 016: training-seed factorization

## question

does model initialization or shuffled training order explain the extreme
center-weighted seed divergence in experiment 015?

## setup

- keep the experiment 015 center-weighted dataset, expert source, model config,
  optimizer, schedule, and 2,500-update budget fixed.
- split the former training seed into a model-initialization seed and a training
  dataloader seed.
- reuse the experiment 015 diagonal cells `(model, loader) = (3991, 3991)` and
  `(3992, 3992)`.
- train only the missing off-diagonal cells `(3991, 3992)` and `(3992, 3991)`.
- use batch size 32, accumulation 1, workers 0, 250 warmup steps, and the same
  five held-out episodes and ten validation batches.
- do not enable new determinism controls or change incomplete-batch handling;
  either change would make the off-diagonal cells incompatible with the reused
  experiment 015 cells.
- evaluate the off-diagonal cells on the exact experiment 015 protocol: 50
  paired scenes in each near, nominal, and far range, evaluation seed 15500,
  deterministic ik, execution horizon 16, and a 500-step limit.

initialization counts as the dominant observed factor only if model seed 3992
beats model seed 3991 under both loader seeds. loader order counts as dominant
only if loader seed 3992 beats loader seed 3991 under both model seeds. mixed
directions indicate an interaction or unresolved training instability. these
two seed levels diagnose the existing divergence; they do not estimate a
population-level seed effect.

## result

successes and lifts across the three 50-scene ranges were:

| model seed | loader seed | near | nominal | far | total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3991 | 3991 | 0/0 | 1/1 | 4/4 | 5/5 |
| 3991 | 3992 | 0/0 | 0/0 | 4/4 | 4/4 |
| 3992 | 3991 | 7/7 | 4/5 | 3/8 | 14/20 |
| 3992 | 3992 | 28/28 | 26/35 | 37/39 | 91/102 |

each cell is `successes/lifts`; totals are out of 150.

model seed 3992 beat model seed 3991 under both loader seeds in aggregate:

- loader 3991: 14 versus 5 successes.
- loader 3992: 91 versus 4 successes.

loader seed 3992 did not win under both model seeds:

- model 3991: 4 versus 5 successes.
- model 3992: 91 versus 14 successes.

the aggregate success contrasts were +32.0 percentage points for model seed,
+25.3 points for loader seed, and +52.0 points for their interaction. the
loader main effect is not independently interpretable because its direction
changes by model seed.

offline metrics did not rank control:

| model seed | loader seed | validation regression | mean translation error |
| ---: | ---: | ---: | ---: |
| 3991 | 3991 | 0.000911 | 0.553 cm |
| 3991 | 3992 | 0.000788 | 0.485 cm |
| 3992 | 3991 | 0.000991 | 0.542 cm |
| 3992 | 3992 | 0.000897 | 0.572 cm |

the lowest-loss, lowest-open-loop-error cell nearly failed, while the strongest
controller had the worst mean open-loop translation error.

raw rollouts are in `reports/seed-factor-exp16-{near,nominal,far}-ik-h16-2model-50.json`.
offline diagnostics and the compact analysis are in
`reports/seed-factor-exp16-offline-diagnostics-h16-4model-250.json` and
`reports/seed-factor-exp16-summary.json`.

## finding

at these two seed levels, robot-side model initialization is the dominant
observed factor under the preregistered aggregate criterion. loader order does
not rescue model seed 3991, but it changes model seed 3992 from 14 to 91
successes. the exceptional controller therefore requires a large
initialization-by-curriculum interaction rather than a generally good loader
order or a purely additive initialization effect.

the factorization does not estimate a population seed effect, and each cell was
initially represented by one run. experiment 017 directly tests same-cell
reproducibility.

## decision

- retain separate model and loader seed controls.
- compare closed-loop control rather than selecting by validation loss or
  open-loop trajectory error.
- do not change the center-weighted data recipe from this diagnostic alone.
- isolate initialization components only under deterministic training; the
  current model seed jointly changes lora and state/part conditioning.
