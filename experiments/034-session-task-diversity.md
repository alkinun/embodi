# 034: session and task diversity

## question

at fixed clip count and exposure, does distributing human pretraining across
more sessions and tasks improve prediction on held-out sessions?

## setup

- use the pinned Xperience-10M revision
  `ce943cf271a758b60240084892d05cf6dc12dd90`, frozen VLM, and selected 53.12M
  action expert.
- compare 1,000 training clips from one session/task against a pooled 1,000 clips
  from five sessions/tasks. preserve the same global 50% meaningful, 25% small,
  and 25% stationary motion mixture; report each session's contribution. session
  and task breadth are confounded in the current cache, so this estimates their
  combined effect rather than either one alone.
- use data seed 81000 and the same 1,000 validation clips from two entirely
  held-out sessions for both conditions. require zero train/validation episode,
  exact-prompt, and main-task overlap.
- use batch size 1, accumulation 32, 1,000 updates, learning rate `1e-4`, 100
  warmup steps, cosine decay, and 32,000 total presentations per condition.
- run paired deterministic model/loader seeds 81001, 81002, and 81003. save and
  evaluate fixed steps 500, 750, and 1,000; do not select any other checkpoint.
- report aggregate validation flow loss, each held-out session separately,
  training loss, motion-bin composition, task/prompt counts, union labeled time,
  and non-overlapping-window capacity from immutable data manifests.
- freeze `reports/xperience-exp34-one-session-split.json` at sha-256
  `64dc080575f97c4d5c65dcffc4157f77b198c151a5630ef2d80ac5401f2c4953`
  and `reports/xperience-exp34-five-session-split.json` at sha-256
  `895c52e8c21230670a5a0088454e7a7ee98b88c97a8f1c03b61fdc4032ff651c`.
  both contain the exact same 1,000 validation records.

the primary endpoint is final aggregate held-out flow loss averaged across the
three seeds. breadth is useful only if the five-session condition improves the
paired final loss in at least two of three seeds and lowers the three-seed mean
by at least 5%. fixed-checkpoint minima and per-session losses are secondary. run
no SO-101 transfer unless the human-only primary gate passes.

## result

pre-registered; execution pending.

## finding

pending.

## decision

pending.
