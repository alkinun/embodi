# 015: center-weighted cube coverage

## question

can a center-weighted cube distribution improve edge success without repeating
the nominal forgetting from uniform-wide training?

## setup

- compare nominal-only, uniform-wide, and center-weighted datasets.
- use 105 successful demonstrations per dataset and the same top camera.
- reuse the experiment 011 nominal-only and uniform-wide datasets.
- allocate center-weighted episodes as 53 nominal, 26 near, and 26 far.
- stratify the center-weighted split as 50/25/25 training episodes and 3/1/1
  validation episodes.
- use generator seed 11000 for all datasets.
- use the same 1,000-clip human expert checkpoint.
- train cores with seeds 3991 and 3992 for 2,500 updates at batch size 32.
- hold initialization, config, split size, optimizer, and schedule fixed.
- use deterministic ik with execution horizon 16.
- evaluate 50 paired scenes in each range with seed 15500 and a 500-step limit.

the primary comparisons are center-weighted versus nominal-only on combined
near and far success, and center-weighted versus uniform-wide on nominal
success. an effect counts as replicated only if its direction agrees across
both training seeds. per-seed paired outcomes will be reported without pooling
cores as independent scenes.

## result

successes out of 50 paired scenes per range were:

| train seed | dataset | near | nominal | far | near + far |
| --- | --- | ---: | ---: | ---: | ---: |
| 3991 | nominal-only | 11 | 2 | 0 | 11/100 |
| 3991 | uniform-wide | 6 | 8 | 4 | 10/100 |
| 3991 | center-weighted | 0 | 1 | 4 | 4/100 |
| 3992 | nominal-only | 3 | 9 | 0 | 3/100 |
| 3992 | uniform-wide | 18 | 6 | 7 | 25/100 |
| 3992 | center-weighted | 28 | 26 | 37 | 65/100 |

the primary paired comparisons reversed across core seeds:

| comparison | train seed | successes | discordant pairs | exact mcnemar p | direction |
| --- | ---: | --- | --- | ---: | --- |
| center vs nominal, combined edges | 3991 | 4 vs 11 | 4 vs 11 | 0.1185 | lower |
| center vs nominal, combined edges | 3992 | 65 vs 3 | 65 vs 3 | 3.56e-16 | higher |
| center vs wide, nominal range | 3991 | 1 vs 8 | 0 vs 7 | 0.0156 | lower |
| center vs wide, nominal range | 3992 | 26 vs 6 | 22 vs 2 | 3.59e-5 | higher |

the scene-level tests describe each fixed core only. they do not override the
preregistered requirement that the effect direction agree across independent
training seeds.

raw rollouts are in
`reports/coverage-exp15-{near,nominal,far}-ik-h16-6model-50.json`; the compact
analysis is in `reports/coverage-exp15-summary.json`.

## finding

center weighting is inconclusive. seed 3992 produced a strong center-weighted
policy across all three ranges, but seed 3991 produced an almost completely
failed center-weighted policy. both primary comparisons changed direction
between seeds, so neither effect replicated.

the result also changes the diagnosis from a coverage-only problem to a
training-stability problem. one seed shows that the center-weighted dataset can
support broad control, while the other shows that this recipe does not learn it
reliably under the current 2,500-update protocol.

## decision

- do not adopt center weighting as a reliable coverage fix.
- keep the seed-3992 center checkpoint as evidence of feasibility, not as a
  selected general policy.
- isolate core-training instability before changing coverage again: use more
  independent core seeds and intermediate closed-loop evaluations under a fixed
  dataset and evaluation suite.
- continue to make no coverage claim from validation loss alone.
