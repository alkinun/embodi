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

| range | success | lift | failures | median failed approach | failed within 15 mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| near | 9/50 | 9/50 | 41 | 21.1 mm | 10/41 |
| nominal | 27/50 | 28/50 | 23 | 17.4 mm | 7/23 |
| far | 24/50 | 24/50 | 26 | 13.7 mm | 16/26 |

all 41 near failures occurred before lift. the gripper attempted closure in
37/41, but only four failed episodes raised the cube above 55 mm and only two
ever moved it within 55 mm xy of the target. median maximum cube height among
near failures was 23.6 mm.

raw telemetry is in `reports/telemetry-exp22-{near,nominal,far}-ik-h16-50.json`.

## finding

near failure is primarily an approach/alignment problem. the policy usually
commands closure but remains farther from the cube than in the other ranges,
then fails to establish a lift. transport and placement are downstream and
cannot explain the near gap because no near failure reaches the lift stage.

## decision

- target near-side reach/grasp supervision rather than transport or decoder
  changes.
- add a reach-sensitive auxiliary diagnostic or oversample near approach frames
  before collecting more full demonstrations.
