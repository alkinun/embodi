# 028: native decoder error decomposition

## question

is experiment 027's closed-loop decoder failure already visible as first-action
approximation error, does error grow primarily across the predicted horizon, or
is offline fidelity low enough that rollout distribution shift is the remaining
explanation?

## setup

- freeze the experiment 027 cache and all three final decoder checkpoints.
- use only held-out cache episodes 50 through 59; perform no training or rollout.
- measure action-standard-deviation-normalized rmse at every horizon step, native
  per-joint first-step error, delta-direction agreement, and cross-seed prediction
  disagreement.
- inspect episode-level errors only with the teacher outcomes already stored in
  the cache.

classify local approximation as the primary failure if mean first-step normalized
rmse exceeds `0.15`. otherwise classify horizon degradation if step-16 rmse is at
least twice first-step rmse. if neither condition holds while experiment 027
closed-loop performance remains poor, treat on-policy distribution shift as the
leading diagnosis. these thresholds are diagnostic and do not authorize model
selection or additional evaluation-scene tuning.

## result

across the three decoders, mean held-out first-step normalized rmse was `0.226`,
above the pre-registered `0.15` local-error threshold. mean step-16 rmse was
`0.207`, so error did not double across the executed horizon. after correcting
disagreement to use the checkpoints' action-normalization scale, cross-seed
first-step normalized disagreement was `0.0271`.

| seed | first-step rmse | step-16 rmse | direction agreement |
| ---: | ---: | ---: | ---: |
| 27001 | 0.235 | 0.195 | 86.4% |
| 27002 | 0.224 | 0.207 | 87.1% |
| 27003 | 0.219 | 0.220 | 87.7% |

first-step native rmse was approximately 4.6--5.3 degrees for shoulder lift and
elbow flex, 2.45--2.53 degrees for wrist flex, about 1 degree for shoulder pan,
and about 0.5 native units for wrist roll and gripper. full per-step, per-joint,
per-episode, and checkpoint-hash diagnostics are in
`reports/selected-core-exp28-decoder-fidelity.json`.

## finding

the pre-registered diagnosis is local approximation error. horizon degradation
and seed disagreement are too small to explain experiment 027's control loss.
stable aggregate validation MSE hid motor errors large enough to alter approach
geometry immediately.

## decision

- do not collect more rollout data or blame on-policy shift yet.
- require first-step held-out rmse below `0.15` and shoulder-lift/elbow native
  rmse below 2 degrees before another closed-loop decoder evaluation.
- test whether optimization budget can meet that offline gate before changing
  architecture or objective.
