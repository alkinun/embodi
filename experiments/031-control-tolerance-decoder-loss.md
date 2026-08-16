# 031: control-tolerance decoder loss

## question

can a physical-tolerance-normalized decoder objective correct experiment 030's
loss mismatch and meet a control-relevant held-out fidelity gate?

## setup

- reuse the experiment 030 width-512 / 4-layer decoder, experiment 027 cache,
  training seed 27002, 10,000 updates, and all optimizer settings.
- change only decoder loss normalization: divide native body-joint errors by 2
  degrees and gripper error by 1 native unit before squared error.
- retain the existing 10x first-step weight.
- evaluate only held-out episodes 50--59; run no closed-loop evaluation unless
  the offline gate passes.

the objective was selected after inspecting this same validation split in
experiments 028--030, so the result is exploratory. failure is a conservative
rejection; success would still require confirmation on a fresh cache.

the objective passes if first-step native rmse is at most 2 degrees for every
body joint, at most 1 native unit for the gripper, and first-step delta-direction
agreement is at least 95%. action-std-normalized rmse remains a secondary metric
because experiment 030 showed that it overweights low-variance wrist roll.

## result

the tolerance-normalized decoder achieved first-step body-joint rmse of 0.58,
1.34, 2.12, 1.28, and 0.49 degrees/native units from shoulder pan through wrist
roll; gripper rmse was 0.31 native units. first-step direction agreement was
94.27%.

elbow flex missed the 2-degree gate by 0.12 degrees and direction agreement
missed the 95% gate by 0.73 points. the objective therefore failed and no
closed-loop evaluation was run. full diagnostics are in
`reports/selected-core-exp31-decoder-fidelity-tolerance-w512-l4-10k.json`.

## finding

on the reused validation split, control-tolerance weighting slightly improves
the remaining elbow error but
degrades wrist-roll and aggregate normalized fidelity. loss reweighting alone
does not resolve the local map, despite coming close to both physical gates.

## decision

- keep the native learned path blocked from closed-loop and physical soak.
- retain experiment 030's standard-loss high-capacity decoder as the stronger
  aggregate offline baseline, but do not promote it.
- investigate target formulation and deterministic-ik continuity before another
  decoder training run; stop local capacity/weight tuning here.
