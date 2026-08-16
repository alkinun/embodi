# 029: decoder optimization budget

## question

is experiment 028's local native-action approximation error primarily caused by
stopping decoder optimization at 2,500 updates?

## setup

- reuse the exact experiment 027 teacher cache and fixed selected-core decoder
  initialization.
- fix training seed 27002 and train from scratch for 10,000 updates with the
  existing architecture, loss, batch size 32, and learning rate `1e-4`.
- evaluate only held-out cache episodes 50 through 59 with the experiment 028
  fidelity analysis; inspect no closed-loop evaluation scenes.
- compare against seed 27002 at 2,500 updates.

because this intervention was chosen after inspecting the same held-out episodes
in experiment 028, results are exploratory validation, not an independent
generalization estimate. a passing candidate would require a fresh confirmatory
cache before any promotion; a failure remains a conservative rejection.

declare optimization budget sufficient to rescue offline fidelity only if
first-step normalized rmse is at most `0.15` and first-step native rmse is at
most 2 degrees for both shoulder lift and elbow flex. if either gate fails, do
not run closed-loop evaluation; move to architecture or objective changes.

## result

at 10,000 updates, final validation loss was `0.03676`, down from `0.05469` at
2,500 updates. held-out first-step normalized rmse improved from `0.224` to
`0.165`, and direction agreement improved from 87.1% to 92.5%.

| joint | 2,500-step rmse | 10,000-step rmse |
| --- | ---: | ---: |
| shoulder pan | 0.98 deg | 0.84 deg |
| shoulder lift | 4.71 deg | 1.55 deg |
| elbow flex | 4.65 deg | 3.19 deg |
| wrist flex | 2.45 deg | 1.75 deg |
| wrist roll | 0.50 native | 0.40 native |
| gripper | 0.54 native | 0.23 native |

the normalized `0.15` gate and elbow 2-degree gate both failed, so no closed-loop
evaluation was run. full diagnostics are in
`reports/selected-core-exp29-decoder-fidelity-10k.json`.

## finding

optimization budget was a material contributor but not the complete cause. the
existing decoder continues improving well beyond 2,500 updates, yet elbow error
remains too large for the exploratory control-fidelity gate on this reused split.

## decision

- keep closed-loop decoder evaluation blocked.
- do not spend more updates on the unchanged model.
- compare a higher-capacity decoder under the same 10,000-update cache protocol,
  changing no other factor.
