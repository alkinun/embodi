# 030: decoder capacity

## question

can a higher-capacity native decoder clear experiment 029's held-out local
fidelity gate when optimization budget and data are fixed?

## setup

- reuse the experiment 027 cache, selected core, training seed 27002, 10,000
  updates, batch size 32, learning rate `1e-4`, and held-out episodes 50--59.
- change only decoder capacity from width 256 / 3 layers to width 512 / 4 layers.
- perform no closed-loop evaluation unless the offline gate passes.

this is an adaptive exploratory comparison on the same validation episodes used
by experiments 028 and 029. it can reject this capacity change, but cannot
confirm generalization without a fresh cache.

the higher-capacity decoder passes only if first-step normalized rmse is at most
`0.15` and native first-step rmse is at most 2 degrees for both shoulder lift
and elbow flex. otherwise reject capacity alone as the next remedy and examine
the target/objective formulation.

## result

the 512-wide / 4-layer decoder reached final validation loss `0.03334`.
first-step normalized rmse improved from the experiment 029 baseline's `0.165`
to `0.156`, while direction agreement improved from 92.5% to 94.1%.

shoulder-lift rmse was 1.58 degrees and elbow-flex rmse was 2.21 degrees. both
the normalized `0.15` gate and elbow 2-degree gate narrowly failed, so no
closed-loop evaluation was run. diagnostics are in
`reports/selected-core-exp30-decoder-fidelity-w512-l4-10k.json`.

## finding

higher capacity improves the local map on the reused validation split but is not
sufficient. the remaining
normalized error is dominated by wrist roll (`0.347` normalized for only `0.41`
native units), while a control-relevant 2.21-degree elbow error contributes only
`0.122` normalized. action-standard-deviation MSE is misaligned with physical
joint tolerances.

## decision

- keep closed-loop evaluation blocked.
- stop scaling the unchanged normalized-MSE decoder.
- test a fixed physical-tolerance-normalized objective that weights body-joint
  errors by degrees rather than dataset variance.
