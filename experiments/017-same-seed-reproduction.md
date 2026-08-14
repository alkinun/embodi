# 017: same-seed reproduction

## question

is experiment 015's exceptional center-weighted `(model, loader) = (3992,
3992)` cell reproducible under the current training recipe?

## setup

- repeat the exact center-weighted 2,500-update command with model seed 3992
  and loader seed 3992 in a new output directory.
- keep the dataset, expert source, config, batch size 32, accumulation 1,
  workers 0, 250-step warmup, validation split, and software/hardware
  environment fixed.
- do not enable deterministic algorithms; this tests reproducibility of the
  recipe that produced experiments 015 and 016.
- compare semantic model tensor hashes and final validation metrics first.
- if the semantic core hash differs, run the exact experiment 015 closed-loop
  suite: 50 paired scenes in each range, seed 15500, deterministic ik, horizon
  16, and a 500-step limit.

an identical semantic core hash counts as exact reproduction. a different hash
requires behavioral evaluation and is reported without inventing a post-hoc
success threshold.

## result

the repeated run did not reproduce the semantic core hash:

| artifact | original | reproduction | identical |
| --- | --- | --- | --- |
| core | `0e1a7c90256a...` | `1cbde9a89a6d...` | no |
| state adapter | `fc2730ad99f4...` | `fc2730ad99f4...` | yes |
| frozen decoder | `c7d96ceaf580...` | `c7d96ceaf580...` | yes |

only 7 of 381 backbone/expert tensors were identical. the final validation
regression losses were nevertheless similar: 0.000897 original and 0.000877
reproduction. state-adapter losses were exactly 0.325868 for both.

closed-loop outcomes were:

| run | near | nominal | far | total |
| --- | ---: | ---: | ---: | ---: |
| original | 28/28 | 26/35 | 37/39 | 91/102 |
| reproduction | 28/29 | 17/24 | 29/42 | 74/95 |

each cell is `successes/lifts`; totals are out of 150. the runs shared 50
successes, with 41 original-only and 24 reproduction-only successes (jaccard
0.435, paired exact mcnemar p=0.0464). the reproduction remained substantially
stronger than experiment 016's best off-diagonal cell at 14/150.

as a follow-up implementation check, strict pytorch deterministic algorithms
with `CUBLAS_WORKSPACE_CONFIG=:4096:8` accepted the real vlm/expert forward and
backward path. two independent one-step runs then produced identical validation
metrics and semantic core hashes.

raw rollouts and the compact comparison are in
`reports/repro-exp17-{near,nominal,far}-ik-h16-1model-50.json` and
`reports/repro-exp17-summary.json`.

## finding

the current recipe is not bitwise reproducible even with identical model seed,
loader seed, data, and command. nondeterminism is confined to the trained
backbone/expert path in this comparison: the independently trained state adapter
was exactly reproducible.

the broad center-weighted capability did behaviorally reproduce, but its rate
and successful-scene identity changed materially. this strengthens experiment
016's association with model seed 3992 while showing that uncontrolled kernel
nondeterminism is another meaningful source of control variance.

## decision

- use `--deterministic` for future causal training comparisons.
- continue to distinguish bitwise training reproducibility from behavioral
  replication.
- do not run a larger seed study until deterministic same-cell reproduction is
  verified over a full training run.
- preserve model, loader, and validation seeds separately in future reports.
