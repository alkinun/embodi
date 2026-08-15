# 022: near failure telemetry

## question

where does the selected deterministic policy fail on near cube positions?

## setup

- evaluate deterministic seed 3993 at step 1,100.
- use 50 paired scenes in each near, nominal, and far range with seed 22500,
  deterministic ik, horizon 16, and a 500-step limit.
- record minimum end-effector-to-cube distance, maximum cube height, minimum
  cube-to-target xy distance, minimum gripper opening, lift, and success.
- compare continuous telemetry by range and by success/lift outcome without
  tuning the policy on these scenes.

## result

pending.

## finding

pending.

## decision

- use the dominant failure stage to choose the next training intervention.
