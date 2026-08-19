# 045: fresh-slice gripper target-state and transition localization

## question

on the final factor-complete development slice unused by focused Experiments
038--044, is the fixed joint core's gripper agreement deficit materially
localized to target state, setpoint-transition direction, or terminal no-op
behavior?

## setup

- use exact development indices 0042--0048 crossed with SO-101, Panda, push,
  lift, and pick/place. these seven indices cover each frozen development factor
  axis once. they are fresh to the focused phase/localization assays but not
  globally unseen because Experiment 037 evaluated all development cases. index
  0049 is excluded because it begins a second factor cycle with workspace only.
  do not access validation or final scenarios.
- freeze benchmark definition SHA-256
  `9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f`,
  development manifest SHA-256
  `4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576`,
  Experiment 042 evaluator SHA-256
  `cc4fb718a3f6341dbaaea0026aca193fd86d5b84412ea7061204ed8f4ce24b7d`,
  Experiment 043 evaluator SHA-256
  `a1056895ab4926aa1d5822ff6ce1bab6d743fc779b3fb1599da26040356e9794`,
  Experiment 044 evaluator SHA-256
  `7d40ef70485090ae595a497b4687883baaee26e9349c1899d864416bd18d8c70`,
  and expert SHA-256
  `7fb869a10756864530889f809a8305ee1758bdc51eaeeeda0ff2f451f3b7f8e9`.
  also freeze asset-loader SHA-256
  `cd1516bbb12a9a4fa13e94b24982683bf0d624b056dcd998ceaa24d63488d81d`,
  benchmark-environment SHA-256
  `e3039b41b3e3cb1856688522f6219e8a6b5b797a73ef0bbed985cfc6b646ef52`,
  and morphology SHA-256
  `b8344f1ef1fc5294dd6a61017761973c6294b7a85477ce391fc3142ad043b632`,
  base environment SHA-256
  `66efff951ee618edd389b4a93c0f6a4ed59fbe5775e162f1940735efc4d622d5`,
  and task runtime SHA-256
  `1101bf431f371b500e16bd735732636a43ff5a4aaa41a75eaf90cb719dd7139c`.
- first freeze expert-only structural support with Experiment 045 support evaluator SHA-256 `4cca61781c45eaa2f3b4778ffe48b87c81630166d2b6d9e2e0b2e0eeda45cb14`.
  run the unchanged privileged expert over all 42
  episodes and persist all 364 first/last endpoint identities, native and
  canonical target evidence, transition identity, array hashes, round-trip
  evidence, and target-first partition labels. load no checkpoint, processor, or
  policy and run zero policy inferences. require a clean merged evaluator commit
  and immutable output `reports/benchmark-exp45-support.json`. bind every
  benchmark-referenced robot XML, mesh, and license file by path, size, and
  SHA-256.
- classify each endpoint before policy inference using the unchanged Experiment
  044 target-first rule: exact terminal native no-op first; otherwise exact
  morphology-specific open or zero closed setpoint; `release` openward; `close`
  closeward; all other phases steady. require every endpoint to receive exactly
  one of `open_steady`, `open_openward`, `closed_steady`,
  `closed_closeward`, or `terminal_noop`. do not preregister marginal partition
  counts. same-call first/last endpoints may share a terminal-noop label.
- support delivery requires exact scenario/phase/endpoint order, 42 successful
  expert episodes, 182 scenario-phase units, 364 endpoints, all seven factor
  axes per morphology/task, exhaustive disjoint labels, valid native/canonical
  hashes and round trips, valid same-call duplicates, zero checkpoint opens,
  zero policy inference, zero training, and `final_split_loaded: false`.
- after the support report exists, bind its SHA-256 and freeze Experiment 045 policy evaluator SHA-256 `0de983db18a4d5abfedda6f0e9195be9ac239a761f1eb76f3e72e1dd5b996825`
  on a clean merged commit. before loading any checkpoint, the policy evaluator
  must recapture the expert endpoints and require exact scenario,
  transition, native/canonical state, target, round-trip, and partition evidence.
  rendered image hashes are excluded from cross-run equality because prior
  deterministic recaptures have shown sparse one-level uint8 differences; each
  newly recaptured frame must instead be shared byte-identically across all nine
  F/H/J arms. before checkpoint load, require the policy runtime's
  benchmark-referenced asset inventory to equal the support report exactly. a
  structural or asset mismatch is delivery failure, not a model result.
- freeze and persist the evaluator's enumerated `SOURCE_FILE_SHA256` closure:
  Experiments 041--044 and support, benchmark/environment/expert and geometry
  loading, processor, v3 token config/policy/canonical/backbone/expert/adapter,
  metric/hash helpers, and their local import-time dependencies. validate every
  entry only after both morphology recaptures pass and before checkpoint access.
  independently recompute runtime/package provenance at that boundary; require
  each shard's claimed commit to equal `git rev-parse HEAD`, exist as a commit,
  and contain evaluator and source-closure blobs at the frozen SHA-256 values.
- compare existing full-exposure specialists `F`, half-exposure specialists `H`,
  and joint cores `J` at model seeds 36001--36003 and fixed update 1,200. bind
  checkpoint, trainer, embodiment, config, data-manifest, Experiment 036 summary,
  Experiment 043 training-summary, and Experiment 042 source identities exactly
  as Experiment 043. run no training, selection, architecture change, weighting
  change, or final-split evaluation.
- retain Experiment 043's correct prompt, lead-zero target, immediate repeat,
  native execution, canonical round trip, and six balanced F/H/J query
  permutations. use two sequential non-resumable morphology shards under one
  registered lock. expect 3,276 scored chunks, 3,276 repeats, 6,552 inferences,
  and 1,092 observations for each of `J-F`, `H-F`, and `J-H`. commit source
  shards as deterministic gzip archives bound to decompressed hashes.
- primary endpoint is effective gripper error after clipping prediction and
  target to `[0,1]`. aggregate endpoint, phase, scenario, task, morphology, then
  seed with equal weights. report raw error, signed bias, saturation, normalized
  channel-9 loss, all action channels, task, morphology, endpoint, phase, and
  seed groups secondarily. preserve exact decimal `J-F=(H-F)+(J-H)` in every
  reported cell.
- report each of the five support-manifest partitions. require every condition
  and seed at every frozen endpoint. localization contrasts use exact common
  scenario support from the support manifest: closed minus open target over lift
  and pick/place; openward minus open-steady within pick/place; closeward minus
  closed-steady over lift and pick/place; terminal no-op minus commanded records.
  no outcome-selected subgroup or threshold is permitted.
- call a partition's `J-F` deficit material only when `J-F>=0.01`, `J/F>=1.10`,
  and at least two seed ratios are at least 1.10. call a localization contrast
  supported only when the winning composite side passes the same materiality
  rule, the absolute
  difference-of-differences is at least 0.01, at least two seed interactions have
  the pooled sign, and both morphology interactions have that strict sign.
  zero `F` denominators fail ratio gates. classification first returns
  `not_materially_localized` if none of the five registered partitions is
  material, even if a composite side passes. openward remains
  pick/place-specific. broadness additionally requires the
  winning `J-F` difference to be nonnegative in every structurally available
  morphology, task, and endpoint subgroup.
- classify `invalid_delivery`, `not_materially_localized`,
  `distributed_material_deficit`, one scoped localization, or
  `multiple-localizations`. Experiment 044's mechanically computed outcomes are
  quarantined: they do not set a direction, threshold, composite, or decision in
  this experiment. a failed gate is not equivalence or evidence of zero effect.
- this remains a development-only descriptive assay on expert-visited states.
  target state, phase, and task are coupled; openward exists only in pick/place;
  model seeds are not independent scenario replications. make no inferential,
  population, causal-mechanism, unique-optimality, closed-loop, admission,
  advancement, retraining, or final-split claim.

## result

the expert-only support freeze completed from clean commit `d2d9c28`. the
independently audited report is `reports/benchmark-exp45-support.json` at
SHA-256 `34c3252feab6b7153a17cf14baa8708fccfbb4551ea6fc71d1dc65b73b85cb58`.
all 42 episodes succeeded, all 182 scenario-phase units and 364 endpoints
validated, all seven factor axes appeared once per morphology/task, all 91
benchmark-referenced asset files matched, and no checkpoint, policy inference,
training run, validation split, or final split was accessed.

the frozen endpoint support is 154 open-steady, 28 openward, 70 closed-steady,
56 closeward, and 56 terminal-noop records. the corresponding scenario-phase
support is 84, 14, 42, 28, and 42; these counts overlap where one phase has
different first/last labels. all 14 same-call first/last pairs were exact
terminal-noop duplicates. policy results remain pending the separately frozen
support-bound evaluator.

## finding

the outcome-blind support stage resolves Experiment 044's delivery failure
without inspecting a model: actual structural marginals are now immutable before
checkpoint load, while common scenario support remains exact by construction.

## decision

- bind the policy evaluator to support SHA-256
  `34c3252feab6b7153a17cf14baa8708fccfbb4551ea6fc71d1dc65b73b85cb58`.
- do not load a policy checkpoint until that evaluator is merged.
- keep the training recipe and final split untouched.
