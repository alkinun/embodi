# 019: deterministic model-seed screen

## question

can another deterministic robot-side initialization recover strong
center-weighted control, and do useful policies appear before the final
checkpoint?

## setup

- keep the center-weighted dataset and loader seed 3992 fixed.
- use strict deterministic training for every new run.
- reuse experiment 018 model seed 3992 at step 2,500.
- train model seeds 3991 and 3993 for 2,500 updates with checkpoints every 500
  updates.
- keep the expert source, config, optimizer, batch size 32, accumulation 1,
  workers 0, 250-step warmup, and validation split fixed.
- screen steps 500, 1,000, 1,500, 2,000, and 2,500 for seeds 3991 and 3993,
  plus the seed-3992 final.
- evaluate ten paired scenes in each near, nominal, and far range with seed
  19000, deterministic ik, horizon 16, and a 500-step limit.
- rank checkpoints by total success over the 30 scenes; validation loss is not
  a selection metric.

a candidate advances only if it reaches at least 10/30 total successes and at
least one success in every range. ties prefer the earlier checkpoint. an
advancing candidate is confirmed on 50 scenes per range with new evaluation
seed 19500 and the otherwise identical protocol. if no candidate qualifies,
the experiment ends after the screen.

this is a deterministic initialization screen, not an estimate of expected
performance over random seeds.

## result

pending.

## finding

pending.

## decision

- keep all intermediate checkpoints until the screen is complete.
- confirm at most one checkpoint to avoid adaptive reuse of the confirmation
  scenes.
