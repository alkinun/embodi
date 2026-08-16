# 035: diversity transfer

## question

does the held-out human-motion gain from broader session/task pretraining
survive expert-only transfer to SO-101 control?

## setup

- compare the one-session and five-session final experts from experiment 034 for
  human seeds 81001, 81002, and 81003. transfer only the action expert.
- use the fixed center-weighted 105-episode SO-101 dataset, top-camera config,
  regression objective, 2,500 updates, batch size 32, learning rate `1e-4`, 250
  warmup steps, and strict deterministic training.
- pair human seed 81001 with fresh robot model/loader seeds 35001/35101, 81002
  with 35002/35102, and 81003 with 35003/35103. within each pair, change only
  the transferred human expert.
- save steps 700, 800, 900, 1,000, 1,100, 1,200, and 1,300. screen every
  checkpoint on ten paired near, nominal, and far scenes with seed 35200,
  deterministic ik, horizon 16, and a 500-step limit.
- select one checkpoint per run by total success over its 30 screen scenes; ties
  prefer the earlier step. both pretraining conditions receive identical
  selection freedom.
- confirm all six selected checkpoints on 50 paired scenes per range with fresh
  seed 35500. reuse each scene across every condition and seed pair.

the primary endpoint is pooled paired success over 450 confirmation scenes per
condition. conclude that breadth transfers only if five-session pretraining wins
in at least two of three seed pairs, gains at least 23/450 pooled successes, and
has two-sided exact mcnemar `p < 0.05`. range-level success, lift, selected step,
and robot validation loss are secondary. this remains a one-task, oracle-state,
deterministic-ik benchmark and is not evidence of unseen-task robot
generalization.

## result

the full training matrix was stopped for compute cost after the first paired
robot seed completed. the completed human-seed-81001 pair used robot model seed
35001, loader seed 35101, and fixed step 1,100. it was screened on the 30 paired
scenes reserved by the pre-registration:

| pretraining | near | nominal | far | total | lifts |
| --- | ---: | ---: | ---: | ---: | ---: |
| one session | 0/10 | 3/10 | 0/10 | 3/30 | 4/30 |
| five sessions | 0/10 | 6/10 | 1/10 | 7/30 | 8/30 |

paired outcomes contained two shared successes, five five-session-only
successes, one one-session-only success, and 22 shared failures. the two-sided
exact mcnemar p value was 0.21875. raw outcomes are in
`reports/xperience-exp35-lightweight-{near,nominal,far}-h16-2model-10.json`; the
compact report is `reports/xperience-exp35-lightweight-summary.json`.

## finding

the fixed-step screen directionally favors broader human pretraining, doubling
nominal success and improving total success by 4/30. the difference is not
statistically significant, both conditions fail every near scene, and one
completed seed pair cannot satisfy the original three-pair confirmation gate.

## decision

- retain experiment 034's five-session expert as the stronger human-pretraining
  result.
- treat improved SO-101 transfer as suggestive rather than confirmed.
- do not use this single-pair screen as evidence of unseen-task robot
  generalization or broad workspace robustness.
