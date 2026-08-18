# 040: lead-zero replan continuity

## question

does removing the lead-zero target discontinuity at each H4 replan causally
improve closed-loop control for the joint core or matched specialists?

## setup

- freeze benchmark definition SHA-256
  `9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f`,
  development manifest SHA-256
  `4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576`,
  Experiment 036 summary SHA-256
  `76d7a3b0dc8954fc1fdbf8def99c840881ee3d9b523bf9f3386be5b82be0308f`,
  Experiment 039 summary SHA-256
  `87bd3d30f350e3cda16a8ac91dfeeec5249b46ace97e42de860362d7bef5a8a8`,
  and imported Experiment 039 evaluator SHA-256
  `7617983a71166a9fa261ecb894e2ee6b02bde6d044c8a2b3edd27d17736c791c`.
  perform no training, admission, checkpoint selection, expert action, geometry
  feasibility projection, or final-split evaluation. H4 is fixed by the prior
  diagnostic sequence.
- use exact development indices 0014--0020 in SO-101 and Panda crossed with
  push, lift, and pick/place. the repeating schedule covers all seven factor
  axes once per cell. this slice is intervention-fresh to Experiments 038--039,
  not globally unseen because Experiment 037 evaluated all development cases.
- compare joint and matched conditions at seeds 36001--36003. matched means the
  SO specialist on SO execution and Panda specialist on Panda execution. one
  condition by seed by morphology shard runs 21 pairs: 12 shards, 252 pairs,
  and 504 fresh episodes. reports are
  `benchmark-exp40-{condition}-seed{seed}-{morphology}-alignment.json`.
- retain top 640x480, supplied FK canonical state, `state=None`, regression,
  32-command chunks, H4, 30 Hz, and 500 steps. decode only the next four commands
  against each unchanged anchor. stage reports in ignored
  `reports/exp40-runtime/` and run one locked condition-major, seed-major,
  morphology-major schedule with one runtime signature and nonoverlapping times.
- apply the frozen Experiment 039 initial-input rule: exact state/instruction
  hashes, image maximum difference at most 1 uint8 value, and differing fraction
  at most `1e-4`. use the first arm in counterbalanced order as the shared pair
  anchor. reset and predict twice from that identical snapshot in arm order,
  require chunk difference at most `1e-5`, and execute the exact source-arm chunk
  in both arms. the second prediction is only a repeatability guard.
- decode the first four source-chunk commands once at the unchanged source
  environment and execute those exact native arrays in both arms. after this
  common pre-treatment prefix, require exact equality of task status and a hash
  covering `qpos`, `qvel`, `act`, `ctrl`, simulation time, acceleration
  warm-start, control time, dwell, lifted, success, and the complete ordered
  milestone accumulator. mismatch is infrastructure failure, never an outcome.
  a pair terminating during the prefix remains a valid intent-to-treat no-op pair.
- at the first post-prefix replan, capture both still-identical live inputs and
  reapply the initial-input equivalence rule. use the counterbalanced first arm
  as a shared treatment-onset anchor, predict twice as a repeatability guard, and
  execute the exact source raw chunk unchanged in baseline and transformed in
  alignment. only subsequent replans use independently observed live trajectories.
  this makes the first treatment contrast differ only by the registered transform.
- baseline then continues unchanged. at each alignment replan, compose the
  previous aligned and new chunks into absolute base-frame targets. reference
  old target 4 and boundary new target 0. translation correction is
  `old_position[4] - new_position[0]`; rotation correction is
  `old_rotation[4] @ new_rotation[0].T`. clip both policy gripper targets to
  decoder-effective `[0,1]` before taking their difference.
- modify only leads 0--3 with weights `1, 0.75, 0.5, 0.25`. add weighted
  translation correction; left-multiply by the deterministic SO(3) shortest-path
  correction at that weight; and add weighted effective-gripper correction.
  re-encode against the current anchor with `corrected_position - anchor_position`
  and `anchor_rotation.T @ corrected_rotation`. leads 4--31 must remain bitwise
  unchanged. intended lead 0 therefore equals old target 4; this does not claim
  alignment of new leads 1--3 with old leads 5--7.
- intended equality is transform integrity, not delivered physical continuity.
  require every intended lead-zero error at most `1e-6 m`, `1e-4 degree`, and
  `1e-6` effective gripper. decode through unchanged 20-iteration IK and report
  pre-transform, intended post, and realized FK post overlap plus correction
  magnitudes by lead, condition, seed, morphology, and task.
- retain decoded/prepared counts separately, but record reconstruction, overlap,
  correction, and clipping telemetry only when the corresponding command is
  actually executed. compute clipping rates over executed commands. preserve
  independent raw/final chunk hashes and equality counters for every baseline
  replan rather than relying on object identity.
- delivery requires no treatment failure, exact treatment-onset state, all
  transform thresholds, bitwise-untouched tails, and at most 1% native clipping
  in every arm and shard. condition-level mechanism corroboration requires pooled
  mean realized lead-zero translation and rotation errors each below their
  pre-transform intended means and effective gripper error not increased.
- track Experiment 039 ordered task progression from returned status and existing
  diagnostics. pair by scenario, seed, morphology, and condition. a structured
  transform, decode, or realized-FK treatment failure records its known phase,
  chunk, and lead where applicable; the pair is excluded from endpoint estimates
  and makes support false. capture or policy-prediction errors in either arm are
  infrastructure failures. initial or onset mismatch aborts the shard.
- classify joint and matched separately. support requires equal-weight three-seed
  paired success gain at least 5 points, at least two nonnegative seed gains,
  positive equal-weight task-equal progression gain, passing delivery, and
  corroborated realized continuity. report discordants and subgroup effects
  without inferential, admission, or advancement claims.

## result

pending.

## finding

pending.

## decision

- do not advance or reject a model from this development-only diagnostic.
- keep the final split untouched.
- if supported, repair replan target composition before a new comparison.
- if unsupported despite delivery, move to task-conditioned policy quality.
