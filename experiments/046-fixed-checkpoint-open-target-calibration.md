# 046: fixed-checkpoint open-target calibration and signed-bias audit

## question

on the last development index unused by focused Experiments 038--045, does the
fixed joint core's open-target deficit reproduce as a morphology-consistent
under-opening bias, and does an Experiment-045-frozen open-target intercept
correction materially attenuate that deficit without retraining?

## setup

- use exact development index 0049 crossed with SO-101 and Panda and the lift
  and pick/place tasks. this index is fresh to the focused assays but not
  globally unseen because Experiment 037 evaluated all development cases. it
  begins a second factor cycle and varies workspace only, so it is a narrow
  confirmation rather than a factor-complete slice. exclude push because it has
  no closed-target comparator in Experiment 045, and exclude stack because it
  was not a pretraining task. do not access validation or final scenarios.
- the open-target subgroup, expected under-opening direction, and intercept
  correction family are outcome-informed by Experiment 045. only transport to
  the new index-0049 expert endpoints and correction performance there are
  prospective; this is not a hypothesis-fresh discovery assay.
- freeze benchmark definition SHA-256
  `9b14210d0c296f53c12dd3c5efec2f022a31ceb4d9bb2a1037bf068f443fd45f`,
  development manifest SHA-256
  `4fec8d65180e94fabd851ecf3468a2f05f5a970ea57c6b301ada859ad8b88576`,
  Experiment 045 support-report SHA-256
  `34c3252feab6b7153a17cf14baa8708fccfbb4551ea6fc71d1dc65b73b85cb58`,
  and Experiment 045 summary SHA-256
  `f5d2b11c297ec3ae54a998d0bb2f78db4da3ad9de95619f0a663ffd3594f008f`.
  bind the SO-101 and Panda Experiment 045 gzip archives at SHA-256
  `1b73eb0908d85c733067b975e536519cd6abe1a18f745ff7eeacf671b2389c1f`
  and `0511f52510607f4840f527dab95d0e116e4abbe5a72dad804a3571c4e7cba2b0`,
  with decompressed SHA-256
  `52eccfc0c9dcda6a63cc6d62e81bc22bcc72ff7c24c1a5042c8f016a63bfdb3d`
  and `5d0a43d8b174ddcae219428436fefd267eeb53de0001b46e69507465ca5a92ea`.
- retain Experiment 045's frozen expert, asset, environment, morphology,
  processor, model-source, checkpoint, trainer, embodiment, config,
  data-manifest, Experiment 036 summary, and Experiment 043 training-summary
  identities. compare the same full-exposure specialists `F`, half-exposure
  specialists `H`, and joint cores `J` at model seeds 36001--36003 and fixed
  update 1,200. run no training, selection, architecture change, loss change,
  weighting change, canonical remapping, or hardware calibration.
- before any index-0049 checkpoint access in Experiment 046, freeze a
  calibration/support evaluator SHA-256, its enumerated source closure, and
  commit on a clean merged commit. at execution, independently recompute and
  persist commit and package/runtime provenance. independently validate the
  Experiment 045 report and source archives, reconstruct every outcome-bearing
  array used below, and require exact agreement with the published summary.
  separately run the unchanged privileged expert over the four index-0049
  episodes and persist exact
  scenario/phase/endpoint order, native and canonical state and target evidence,
  transition identity, array hashes, round trips, target-first labels, and the
  complete benchmark-referenced asset inventory.
- retain Experiment 045's target-first classification exactly: classify exact
  terminal native no-op first; otherwise use the exact morphology-specific open
  or zero closed setpoint, with `release` openward, `close` closeward, and all
  other phases steady. require each endpoint to receive exactly one of
  `open_steady`, `open_openward`, `closed_steady`, `closed_closeward`, or
  `terminal_noop` before policy inference.
- calibration/support delivery requires four successful expert episodes in the
  frozen lift and pick/place phase order, 22 scenario-phase units, 44 first/last
  endpoints, exhaustive disjoint target-first labels, valid native/canonical
  hashes and round trips, and nonempty support for each task, morphology, and
  endpoint marginal used by a gate. require zero checkpoint opens, zero new
  policy inference, zero training, and `final_split_loaded: false`. do not
  preregister marginal partition counts. persist the 18 offsets and immutable
  structural support in one atomic report; any evaluator, source, provenance,
  reconstruction, or delivery mismatch is `invalid_delivery`.
- derive one open-target intercept for each condition, model seed, and
  morphology, for 18 frozen offsets total, using only Experiment 045 indices
  0042--0048 and the lift and pick/place records labeled `open_steady` or
  `open_openward`. for each condition, seed, and morphology, restrict the frozen
  endpoint hierarchy to those records; average endpoint, phase, scenario, then
  task with equal weights; and set `delta = -Hmean(prediction-target)`. the two
  open partitions are not first averaged as equal groups. no index-0049 outcome
  may select, refit, clip, or otherwise change an offset.
- after binding the immutable calibration/support report, freeze the policy
  evaluator and its source closure on a clean merged commit. before checkpoint
  load, recapture index-0049 structural evidence and require exact equality with
  the support report except for rendered-frame hashes; each new frame must be
  byte-identical across all nine F/H/J arms. validate source, package, runtime,
  asset, checkpoint, and source-summary provenance before policy access.
  independently recompute all 18 offsets and support counts directly from the
  bound archives and require exact equality with the calibration/support report
  before checkpoint access; independently validate the completed report and its
  hash before promoting any result artifact.
- retain Experiment 045's correct prompt, lead-zero target, immediate repeat,
  native execution, canonical round trip, six cyclic F/H/J query permutations,
  and two sequential non-resumable morphology shards under one registered lock.
  reset the policy before every query, score the first output, and require the
  repeated chunk to differ by at most `1e-5`. query every frozen index-0049
  endpoint in the two primary open-target partitions; support counts and exact
  inference totals become immutable in the calibration/support report.
- let `N` be the frozen number of primary open-target endpoints. policy delivery
  requires every condition and seed at each endpoint, exactly `9*N` scored
  chunks and immediate repeats, `18*N` policy inferences, and `3*N` observations
  for each of `J-F`, `H-F`, and `J-H`. every output must be finite with shape
  `[1,32,1,10]`. any missing, duplicate, nonfinite, or misordered query, failed
  repeat, count mismatch, or incomplete contrast is `invalid_delivery`.
- for each raw prediction `p`, evaluate the paired corrected prediction
  `p_corrected=clip(p+delta[condition,seed,morphology],0,1)` without a second
  policy query. define raw effective error as
  `abs(clip(p,0,1)-clip(target,0,1))` and corrected effective error as
  `abs(p_corrected-clip(target,0,1))`; apply `Hmean` by averaging endpoint,
  phase, scenario, task, morphology, then seed with equal weights. raw signed
  bias is unmodified `p-target`; corrected residual signed bias is
  `p_corrected-target`. raw absolute error is `abs(p-target)`, raw saturation is
  `p<0 or p>1`, calibrated boundary clipping is
  `p+delta<0 or p+delta>1`, and correction magnitude is
  `abs(p_corrected-clip(p,0,1))`. under-opening and over-opening rates use strict
  negative and positive raw or corrected signed bias separately; exact zero
  belongs to neither. raw normalized channel-9 loss is `((p-target)/0.5)^2` and
  corrected normalized loss is `((p_corrected-target)/0.5)^2`. report mean
  prediction and target secondarily.
- report raw and corrected metrics by condition, seed, morphology, task,
  partition, phase, and first/last endpoint. preserve exact decimal
  `J-F=(H-F)+(J-H)`, attenuation `A_JF=gain_J-gain_F`, and
  `A_JF=A_HF+A_JH` in every reported cell, where
  `gain_C=error_C_raw-error_C_corrected` and
  `A_XY=(X-Y)_raw-(X-Y)_corrected`. every unqualified error, contrast, ratio,
  gain, or attenuation used by a scientific gate below refers to the
  hierarchically aggregated effective clipped error defined above.
- call the focused-assay-fresh raw open-target premise confirmed only when pooled
  `J-F>=0.01`, `J/F>=1.10`, and at least two seed ratios are at least 1.10. zero
  `F` denominators fail ratio gates. call joint-specific signed under-opening
  supported only when pooled `bias_J<=-0.01`, at least two seed `bias_J` values
  and both morphology `bias_J` values are strictly negative, pooled
  `bias_J-bias_F<=-0.01`, at least two seed contrasts are strictly negative, and
  both morphology contrasts are strictly negative.
- call calibration gain supported only when pooled `gain_J>=0.01`, at least two
  seed gains are strictly positive, and both morphology gains are strictly
  positive. call joint-deficit attenuation supported only when pooled
  `A_JF>=0.01`, at least two seed attenuations are strictly positive, and both
  morphology attenuations are strictly positive. the strongest consistency
  label additionally requires nonnegative joint gain and `A_JF` in both tasks,
  both morphologies, and both endpoints.
- classify in precedence order as `invalid_delivery`,
  `focused_assay_fresh_open_target_premise_not_confirmed`,
  `open_target_signed_underbias_not_supported`,
  `signed_underbias_without_calibration_support`,
  `open_target_calibration_supported-narrow`, or
  `open_target_calibration_supported-task_and_morphology_consistent`. a failed
  mandatory support, provenance, reconstruction, repeat, hierarchy, or exact
  decomposition check is `invalid_delivery`. otherwise evaluate the raw premise,
  then both absolute and relative signed-under-opening gates, then require both
  joint calibration gain and joint-deficit attenuation for calibration support,
  and finally apply the strongest consistency gate. a failed scientific gate is
  not equivalence or evidence of zero effect.
- this is a development-only, target-conditional correctability assay on four
  expert-visited scenarios. the offset uses an expert-known open target and is
  not a deployable calibrator. it does not test intermediate or closed targets,
  terminal no-op safety, physical calibration, or canonical endpoint semantics.
  clipping can create apparent gains, so raw and boundary metrics remain
  mandatory. make no population, inferential, causal-training-mechanism,
  unique-optimality, closed-loop, safety, admission, advancement, retraining, or
  final-split claim.

## result

pending.

## finding

pending.

## decision

pending.
