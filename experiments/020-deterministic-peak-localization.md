# 020: deterministic peak localization

## question

is the deterministic seed-3993 control peak centered at step 1,000, or does a
nearby checkpoint improve broad control?

## setup

- repeat the exact deterministic model-seed-3993, loader-seed-3992 training
  protocol from experiment 019 for 2,500 updates.
- save compact inference checkpoints only at steps 700, 800, 900, 1,000, 1,100,
  1,200, and 1,300; retain a full final checkpoint.
- verify that the repeated step-1,000 and final model states exactly match
  experiment 019 before evaluating.
- screen all seven checkpoints on ten paired scenes per near, nominal, and far
  range with seed 20000, deterministic ik, horizon 16, and a 500-step limit.
- rank by total success over 30 scenes; ties prefer the earlier checkpoint.

a checkpoint other than step 1,000 replaces the current selection only if it
beats step 1,000 by at least 3/30 successes and succeeds in every range. such a
candidate is confirmed against step 1,000 on 50 paired scenes per range with
new seed 20500. otherwise retain step 1,000 without confirmation.

## result

pending.

## finding

pending.

## decision

- do not use validation loss to rank the local checkpoints.
- stop narrowing if the 100-step grid does not produce a materially better
  checkpoint.
