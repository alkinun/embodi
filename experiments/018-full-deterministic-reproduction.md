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

the two independent 2,500-update runs were semantically identical:

| artifact | semantic hash | identical |
| --- | --- | --- |
| core | `6a6b16bc1a1b...` | yes |
| embodiment | `6e13ae412909...` | yes |
| optimizer and scheduler | `689c53efbbe7...` | yes |
| model config | semantic comparison | yes |
| run manifest except output path | semantic comparison | yes |

the complete printed train and validation metric trajectories also matched.
both runs ended at validation regression loss 0.000950 and state-adapter loss
0.325868.

one copy was evaluated in closed loop:

| range | successes | lifts | episodes |
| --- | ---: | ---: | ---: |
| near | 0 | 1 | 50 |
| nominal | 9 | 12 | 50 |
| far | 10 | 17 | 50 |
| total | 19 | 30 | 150 |

strict determinism changed the optimization path substantially despite similar
offline loss. the two non-deterministic `(3992, 3992)` runs from experiment 017
scored 91/150 and 74/150, while this deterministic path scored 19/150.

raw rollouts are in
`reports/det-repro-exp18-{near,nominal,far}-ik-h16-1model-50.json`; the compact
result is in `reports/det-repro-exp18-summary.json`.

## finding

strict deterministic mode makes the complete training state exactly
reproducible over 2,500 updates. this resolves the infrastructure problem from
experiment 017.

determinism does not preserve the favorable non-deterministic control basin.
the reproducible seed-3992 policy retains some nominal and far control but loses
all near success. validation loss again fails to reveal this regression.

## decision

- use strict determinism for all new causal comparisons.
- re-establish closed-loop performance within the deterministic protocol before
  changing coverage or returning to decoder work.
- screen additional model seeds under fixed loader seed 3992, then replicate
  any promising seed with the exact same command.
- save intermediate checkpoints for the next screen because final validation
  loss cannot identify a useful control peak.
