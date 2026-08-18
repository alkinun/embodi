# 044: gripper target-state and transition-direction localization

## question

is the fixed joint core's development-only gripper agreement deficit materially
localized to an open or closed target state, an opening or closing setpoint
transition, or an intentional terminal no-op target?

## setup

- perform a registered post hoc secondary analysis of the immutable Experiment
  042 and 043 records. run no training, simulation, expert rollout, processor,
  checkpoint load, or policy inference. do not access the final split. freeze
  the Experiment 044 evaluator SHA-256 `7d40ef70485090ae595a497b4687883baaee26e9349c1899d864416bd18d8c70`
  on a clean merged preregistration
  commit before computing any condition-dependent localized outcome.
- freeze imported Experiment 042 evaluator SHA-256
  `cc4fb718a3f6341dbaaea0026aca193fd86d5b84412ea7061204ed8f4ce24b7d`
  and imported Experiment 043 evaluator SHA-256
  `a1056895ab4926aa1d5822ff6ce1bab6d743fc779b3fb1599da26040356e9794`;
  abort if either implementation differs at aggregation time.
- freeze Experiment 042 summary SHA-256
  `104c16ca2552eb0d2ba7efc65c9228eeecd488d67d765f49d70931aa0d15abf8`,
  Experiment 043 summary SHA-256
  `acdf15900cdb708e8f9b96a5c12f2d70be18f951906b92ade8d2a874b69d15a4`,
  Experiment 042 SO-101 and Panda source SHA-256 values
  `f02815631edb583a5faab4a82926afc650b69d5fc131a60227107bc0abecdd51`
  and `55018696519b930d01e859a19365171a5b70d1fb227777b9bf624daecd0732e5`,
  and Experiment 043 decompressed SO-101 and Panda source SHA-256 values
  `7b2ce863b96ecd873f28cc80a86d8279ec8e58dee3db346d5d82593aee998bb8`
  and `ec3d3534227b86c4bebfa20ceae264f66d86fd1dbaf32cfd3a78818dd20b05ec`.
  also freeze the tracked deterministic Experiment 043 gzip SHA-256 values
  `72f5767a8ff96dfa6d13237ff47e269af0f3848e7d45ebb9a59d924f313feec0`
  and `4dee6a5b629e112b9867d2766e3b03c0a0c0121568691d5522b0836b0b83db06`.
- pool the adjacent development slices from indices 0028--0034 in Experiment
  042 and 0035--0041 in Experiment 043 for the primary `J-F` localization. map
  Experiment 042 `matched` to full-exposure `F`. use the same model seeds
  36001--36003 and give the two slices equal weight within each seed. report
  Experiment 043 `H-F` and `J-H` within every partition as a secondary exact
  decomposition only; do not rerun or reinterpret Experiment 043's mechanism
  classification.
- derive labels only from persisted expert evidence, before reading policy
  error. classify an anchor as `terminal_noop` first when `post_action_phase` is
  `done` and float32 `expert_native` is exactly equal to
  `pre_call_native_state`. for every other anchor, classify the native target as
  open only when it is exactly 16 for SO-101 or 0.08 for Panda, and closed only
  when it is exactly zero. any other nonterminal setpoint is delivery failure.
  classify `release` as openward, `close` as closeward, and every other
  nonterminal phase as steady. these are expert setpoint-transition labels, not
  observed gripper velocity.
- the five exhaustive endpoint partitions are `open_steady`, `open_openward`,
  `closed_steady`, `closed_closeward`, and `terminal_noop`. in each source slice
  require respectively 154, 28, 70, 56, and 56 endpoint anchors and 84, 14, 42,
  28, and 42 distinct scenario-phase units. phase-unit counts overlap across
  endpoint partitions when a phase begins with a command and ends with an
  intentional terminal no-op. require 728 pooled endpoint anchors, 5,460 loaded
  condition records, and 2,184 primary paired `J-F` observations.
- recompute the authoritative lead-zero effective gripper error directly from
  the first persisted output chunk as
  `abs(clip(prediction[9],0,1)-clip(target[9],0,1))`. validate both output hashes,
  the immediate repeat and `1e-5` tolerance, the native and canonical expert
  array hashes, stored effective error, and channel-9 squared loss. repeats are
  validation evidence and are not scored. rely on the frozen summaries and
  byte-complete shard hashes for their prior full registered validation, then
  independently revalidate every outcome-bearing array and selected identity
  used by Experiment 044. report raw absolute error, signed
  prediction-minus-target bias, saturation strictly outside `[0,1]`, and
  normalized channel-9 loss with registered scale 0.5 secondarily.
- aggregate each restricted cell in the frozen order: equal-weight endpoints
  within scenario-phase, phases within scenario, scenarios within task, tasks
  within morphology, morphologies within slice, the two slices within seed, then
  the three paired seeds. report fixed slice, seed, morphology, task, and endpoint
  subgroups. preserve exact decimal `J-F=(H-F)+(J-H)` in each Experiment 043
  partition.
- report material `J-F` deficit in a partition only when `J-F>=0.01`, `J/F>=1.10`,
  and at least two seed ratios are at least 1.10. failure is not equivalence or
  evidence of zero effect.
- preregister four difference-of-differences localization contrasts on common
  support: closed minus open target over lift and pick/place; openward minus
  open-steady within pick/place; closeward minus closed-steady over lift and
  pick/place; and terminal no-op minus all commanded nonterminal records. for
  each, subtract the right cell's `J-F` difference from the left cell's `J-F`
  difference. a negative value localizes in the reverse direction. this is a
  preregistered two-sided rule: the pooled sign selects the candidate direction,
  then strictly same-sign seed and slice interactions test consistency.
- call a contrast localized only when its winning cell passes the partition
  materiality rule, the absolute interaction is at least 0.01 effective opening
  fraction, at least two seed interactions have the registered direction, and
  both source-slice interactions have that direction. call it broad only when
  the interaction also has that direction in both morphologies and the winning
  cell's `J-F` difference is nonnegative in every structurally available slice,
  morphology, task, and endpoint subgroup. openward is intrinsically confined
  to pick/place and can only receive a `pick-place-specific` label.
- classify `invalid_delivery` if any source, reconstruction, label, count, or
  support gate fails after a complete aggregate. a source, reconstruction,
  label, or common-support exception aborts without a model report and is also
  delivery failure, not a model result. otherwise classify
  `not_materially_localized` before any localization label when no
  partition is material, `distributed_material_deficit` when a material cell
  exists but no localization contrast passes, the single passing localization
  with `broad`, `narrow`, or `pick-place-specific` scope, or
  `multiple-localizations` when more than one passes. do not select a new phase,
  threshold, composite, or contrast after reading results.
- this reuses previously inspected development records and is not
  hypothesis-fresh. target state, expert phase, and task remain coupled;
  opening is confined to pick/place release; endpoint records within one expert
  trajectory are not independent; model seeds and checkpoints repeat across
  slices; clipping can hide overshoot; and all states are expert-visited.
  terminal endpoint subgroups remain descriptive because their task composition
  differs structurally. make no causal-mechanism, inferential, population,
  unique-optimality, closed-loop,
  admission, advancement, retraining, or final-split claim.

## result

the zero-inference aggregate completed from clean preregistration commit
`da9a9ca`. its immutable report is `reports/benchmark-exp44-summary.json` at
SHA-256 `956ed4db9de635475e8590fcbf396e2540dae3296f84cc043b6552144609c33b`.
all four source hashes, both imported evaluator hashes, both source-summary
delivery states, every outcome-bearing array reconstruction, 728 endpoint
anchors, 2,184 primary paired observations, 5,460 condition records, exhaustive
labels, zero-inference and zero-training declarations, and the untouched final
split passed validation.

the registered partition-support gate failed. Experiment 042 contained 153
open-steady endpoint anchors and 83 open-steady scenario-phase units rather than
the registered 154 and 84; it contained 57 terminal-noop anchors rather than 56.
Experiment 043 matched the registered counts, and all other partition counts
matched in both slices. the exact discrepancy is SO-101 push scenario
`development-so101-push_to_zone-0028`: its `push` phase starts and transitions to
`done` on the same expert call, so the duplicated first and last endpoints both
correctly receive the terminal-noop label. the source validators require those
same-call endpoint records to be identical.

the registered classification is therefore `invalid_delivery`. the report
contains mechanically computed partition and localization metrics, but they are
not model results and are not interpreted.

## finding

the failure is a preregistered support-table error, not source corruption, label
failure, checkpoint behavior, or a final-split event. the target-first labeling
rule correctly exposed a legitimate structural difference between the two
development slices that the fixed support table incorrectly assumed away.

## decision

- retain `invalid_delivery` and make no exposure, weighting, architecture,
  admission, advancement, or retraining decision from Experiment 044.
- keep the final split untouched.
- any corrected localization assay must use a new experiment number, freeze the
  observed structural support explicitly, and disclose that its source outcomes
  have already been mechanically aggregated; do not retroactively change this
  experiment's gate or interpret its metrics.
