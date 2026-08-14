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

pending.

## finding

pending.

## decision

- keep both off-diagonal runs final-only to limit disk use.
- compare closed-loop control rather than selecting by validation loss.
- do not change the center-weighted data recipe from this diagnostic alone.
