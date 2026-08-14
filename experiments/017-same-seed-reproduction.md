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

pending.

## finding

pending.

## decision

- distinguish bitwise training reproducibility from behavioral replication.
- do not add determinism controls until this existing recipe is measured.
