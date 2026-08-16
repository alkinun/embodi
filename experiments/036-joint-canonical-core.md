# 036: joint canonical core

## question

can one morphology-independent canonical core learn SO-101 and Panda manipulation
jointly without sacrificing specialist accuracy, when total training exposure is
held fixed?

## setup

- use the frozen `embodi-sim-v1` train and validation datasets audited in
  `reports/benchmark-data-v1-audit.json`. require generation-manifest SHA-256
  `9d858858c217c2b1200507a503148c99562a06f0beecea054a2b9cc184b840af`
  for train and
  `2735a044d11db30172ae6039f09dfe06adc49462f3b47d5e7b99cb312c354f45`
  for validation.
- compare three conditions: SO-101-only, Panda-only, and joint SO-101/Panda.
  instantiate separate embodiment policy wrappers, native normalization, state
  adapters, and decoders, but use exactly one shared `EmbodiCore` in the joint
  condition.
- train stage `core` only. condition the core on supplied canonical state and do
  not compute native state-adapter or action-decoder losses. keep the VLM vision
  encoder frozen, use the selected 53.12M expert, top camera, 32-step chunks,
  regression objective, and the existing fixed canonical physical scales.
- balance task/morphology cells before collation. each specialist update receives
  30 frame samples: 10 from each of push, lift, and pick/place. each joint update
  receives five samples from each of the six task/morphology cells. this gives
  every condition 36,000 presentations over 1,200 optimizer updates while the
  joint condition receives half as many presentations per cell.
- use learning rate `1e-4`, 120 warmup updates, cosine decay, batch size 1, and
  gradient accumulation over the 30 scheduled samples. use deterministic paired
  model seeds 36001, 36002, and 36003 and loader seeds 36101, 36102, and 36103.
- save and compare fixed update 1,200 only. do not perform checkpoint selection.
  morphology wrappers in the joint condition must reference byte-identical shared
  core payloads.
- evaluate canonical regression loss on a fixed, task-balanced set of 256 frames
  per validation cell. reuse the exact frame indices across every condition and
  seed. freeze `reports/benchmark-exp36-validation-frames.json` at SHA-256
  `32681217e8d6f4142ce754b18b2a3a69527edea9abce3506fb55e7d25c28cc1a`.
  report each task/morphology cell, morphology macro means, the six-cell macro
  mean, train loss, and gradient norm.
- evaluate each specialist both in-domain and on the opposite morphology by
  loading its core into that morphology's wrapper and supplying canonical state.
  do not train or evaluate learned native decoders in this experiment.

the primary endpoint is final six-cell macro validation loss. joint training is
admitted as the shared-core recipe only if its three-seed mean is no more than
10% worse than the matched specialist ensemble macro, it improves over the
opposite-morphology specialist transfer mean by at least 10%, and both conditions
hold in at least two of three paired seeds. morphology/task cells are secondary
failure diagnostics. run no development closed-loop evaluation unless this gate
passes, and do not evaluate the final split.

## result

pending.

## finding

pending.

## decision

pending.
