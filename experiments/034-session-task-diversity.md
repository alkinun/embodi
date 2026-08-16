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

the frozen splits had identical 50/25/25 motion composition and identical
validation records. the one-session split contained one task, 6 prompts, 241.5
union labeled seconds, and 185 non-overlapping windows. the five-session split
contained five tasks, 47 prompts, 579.8 union labeled seconds, and 430
non-overlapping windows.

held-out flow loss at the three fixed checkpoints was:

| condition | seed | step 500 | step 750 | step 1,000 |
| --- | ---: | ---: | ---: | ---: |
| one session | 81001 | 0.4682 | 0.4336 | 0.4401 |
| five sessions | 81001 | 0.3870 | 0.3661 | 0.3579 |
| one session | 81002 | 0.4619 | 0.4502 | 0.4522 |
| five sessions | 81002 | 0.3897 | 0.3436 | 0.3438 |
| one session | 81003 | 0.4811 | 0.4814 | 0.4779 |
| five sessions | 81003 | 0.3811 | 0.3658 | 0.3531 |

mean final loss fell from 0.4567 to 0.3516, a 23.0% reduction. paired final
reductions were 18.7%, 24.0%, and 26.1%, so all three seeds improved. mean final
loss on held-out sessions `7f655.../ep5` and `d659.../ep2` fell from 0.5314 to
0.3997 and from 0.3858 to 0.3060 respectively. mean final training-evaluation
loss was 0.0881 for one session and 0.0973 for five sessions.

post-training evaluation exposed that historical frozen-backbone checkpoints
stored the expert but omitted randomly initialized frozen local state/part
projections. the fixed evaluator reconstructs those runs from their recorded
model seeds and exactly matches training-time final metrics; future checkpoints
now persist the local projections directly. full metrics, hashes, and per-session
results are in `reports/xperience-exp34-summary.json` and the associated
`reports/xperience-exp34-*-seed*.json` files.

## finding

the primary gate passes: breadth improves all three paired seeds and lowers mean
final held-out loss by substantially more than 5%. the higher five-session
training loss with lower held-out loss is consistent with reduced single-session
overfitting. this experiment identifies the combined value of session, task,
prompt, and independent temporal coverage; it does not isolate which component
causes the gain.

## decision

- prioritize new sessions, tasks, scenes, participants, and objects over denser
  windows from existing episodes.
- use pooled global motion balancing and immutable task/session-disjoint splits
  for future human pretraining.
- permit a separately pre-registered SO-101 transfer comparison because the
  human-only gate passed, but do not treat one-task robot transfer as unseen-task
  generalization evidence.
- build a broader task-disjoint human validation set and locked multi-task
  simulation benchmark before making model-size or generalist claims.
