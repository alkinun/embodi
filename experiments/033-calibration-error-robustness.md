# 033: calibration error robustness

## question

how accurately must body-joint zero offsets be calibrated for geometry-first
canonical control to retain the selected policy's nominal task performance?

## setup

- freeze the selected general checkpoint at step 1,100 and use deterministic ik,
  horizon 16, zero canonical noise, and a 500-step limit.
- evaluate 50 paired nominal scenes with seed 33000.
- first compare the existing exact-simulator path with a sensor-roundtrip
  geometry path at zero calibration error.
- then test fixed per-session body-joint offset errors with maximum magnitudes
  0.5, 1, 2, and 5 degrees. for each nonzero magnitude, draw three five-joint
  vectors uniformly from `[-magnitude, magnitude]` with seed 33001 and reuse the
  same 50 scenes.
- the geometry path estimates model state as native state plus the fixed offset,
  performs canonical state construction and ik in that estimated model, then
  subtracts the same offset when sending the native target to the true simulator.
- change no policy, checkpoint, scene, horizon, or controller setting.

continue to the error sweep only if zero-error geometry differs from exact
simulation by at most 2/50 successes and has no significant paired difference at
`p < 0.05`. define the operational offset tolerance as the largest magnitude
whose mean success across three draws retains at least 90% of zero-error geometry
success and whose worst draw loses no more than 5/50 successes. success retention
is primary; lift rate and per-draw paired outcomes are secondary.

## result

the exact-simulator baseline succeeded on 25/50 scenes and lifted on 27/50.
the zero-error sensor-roundtrip geometry path succeeded and lifted on 24/50.
the success difference was 1/50; paired outcomes contained four exact-only and
three geometry-only successes (two-sided exact mcnemar p=1.0), so the
pre-registered admission gate passed.

successes across the three fixed-offset draws were:

| maximum offset | successes by draw | mean | worst | mean retention |
| --- | --- | --- | --- | --- |
| 0.5 degrees | 27, 24, 25 | 25.3 | 24 | 105.6% |
| 1 degree | 26, 29, 27 | 27.3 | 26 | 113.9% |
| 2 degrees | 28, 29, 26 | 27.7 | 26 | 115.3% |
| 5 degrees | 23, 27, 25 | 25.0 | 23 | 104.2% |

mean lifts were 26.3, 27.3, 27.7, and 26.3/50 respectively. paired success
losses and gains relative to zero-error ranged from 4--12 and 7--13 per draw,
showing substantial scene-level turnover despite stable aggregate success. raw
outcomes and checkpoint hashes are in
`reports/calibration-exp33-exact-baseline-h16-50.json`,
`reports/calibration-exp33-geometry-zero-h16-50.json`, and
`reports/calibration-exp33-offset-sweep-h16-50.json`.

## finding

all tested magnitudes pass the pre-registered aggregate criterion: every mean is
above 21.6/50 and every worst draw is at least 19/50. the operational tolerance
is therefore at least 5 degrees for these sampled, fixed per-session offsets;
the experiment does not locate a failure boundary. non-monotonic scores and
paired outcome turnover indicate that offsets move scenes around the selected
policy's narrow control basin rather than producing a simple monotonic loss.

## decision

- do not require tighter than 5-degree body-joint zero calibration based on this
  simulation screen alone.
- do not interpret three random draws as a worst-case guarantee over all joint
  offset vectors, or as evidence that 5-degree physical calibration is safe.
- retain geometry-first deterministic ik as the physical bring-up candidate,
  subject to measured calibration, tcp, direction, limit, and low-speed hardware
  acceptance gates.
- stop synthetic offset refinement until physical measurements identify the
  relevant error range or a broader geometry uncertainty model is specified.
