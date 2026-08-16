# simulation benchmark v1

`definition.json` freezes the benchmark semantics before task environments,
scenario manifests, demonstrations, or policies are generated.
`definition.sha256` is the reviewable frozen digest; scenario manifests must bind
that exact digest.

The benchmark tests four separate questions:

1. multi-task control across physically different contact modes;
2. low-data adaptation to a held-out compositional task;
3. transfer between SO-101 and Franka Panda morphologies;
4. the gap between geometry execution and learned state/action paths.

The three pretraining task families are push, lift, and pick/place. Stacking is
held out from broad pretraining and receives only nested adaptation sets of 10,
50, 100, or 500 demonstrations. Each task has one canonical instruction in v1,
so prompt paraphrases cannot count as physical task diversity.

Scenario manifests must enumerate every physical factor explicitly and bind the
exact SHA-256 of `definition.json`. Development scenes may be used for model and
checkpoint selection. Final scenes may be evaluated only after architecture,
checkpoint, control track, execution horizon, and statistical analysis are
frozen.

The geometry track uses native proprioception with deterministic forward and
inverse kinematics. It is the primary representation/control ceiling. The
learned-state and full-learned tracks are reported separately and cannot inherit
the geometry track's validation claim.

Before scenario manifests are frozen, every task/morphology cell must pass the
admission gates in `definition.json`. Individual difficult evaluation scenes
must not be filtered after the manifest is frozen.

The deterministic manifests were frozen on 2026-08-16. `manifests.sha256`
records their immutable digests. Privileged-oracle admission passed all 400
development episodes and all 800 final episodes; the hash-bound reports are in
`reports/`. This admission establishes scenario feasibility only and does not
authorize final-split model selection or policy evaluation.
