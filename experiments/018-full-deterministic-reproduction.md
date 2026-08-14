# 018: full deterministic reproduction

## question

does strict deterministic training produce identical full center-weighted runs,
and what control policy does that deterministic optimization path learn?

## setup

- train two independent center-weighted runs with `--deterministic`.
- use model seed 3992, loader seed 3992, and validation seed 0 for both runs.
- keep the experiment 015/017 dataset, expert source, config, batch size 32,
  accumulation 1, workers 0, 2,500 updates, 250-step warmup, five validation
  episodes, and ten validation batches fixed.
- save only final checkpoints to limit disk use.
- record full run manifests in each output root.
- compare semantic core, embodiment, optimizer, and scheduler state rather than
  serialized file bytes.
- exact semantic equality across both full runs is the primary endpoint.
- if the cores are identical, evaluate only one copy on 50 paired scenes in
  each range with seed 15500, deterministic ik, horizon 16, and a 500-step
  limit.

closed-loop success is secondary. this experiment tests reproducibility first;
it does not require deterministic training to recover experiment 015's
non-deterministic 91/150 checkpoint.

## result

pending.

## finding

pending.

## decision

- do not start broader seed studies unless both full runs match semantically.
- retain final-only checkpoints unless the deterministic policy still shows a
  closed-loop training failure that requires intermediate evaluation.
