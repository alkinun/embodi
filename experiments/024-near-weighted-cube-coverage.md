# 024: near-weighted cube coverage

## question

can reallocating robot demonstrations toward near cube positions correct the
selected policy's pre-lift approach failures without destroying nominal and far
control?

## setup

- generate 105 successful top-camera demonstrations with seed 11000.
- allocate 53 episodes to near positions `[0.24, 0.28]`, 26 to nominal
  `[0.28, 0.32]`, and 26 to far `[0.32, 0.36]`.
- stratify the final five validation episodes as 3 near, 1 nominal, and 1 far,
  leaving a 50/25/25 training split.
- compare against the center-weighted dataset's 25/50/25 near/nominal/far
  training split at the same total data budget.
- train with expert-only initialization, model seed 3993, loader seed 3992,
  strict determinism, and the experiment 020 optimizer protocol.
- save checkpoints at steps 700 through 1,300 in 100-step increments.
- screen the seven checkpoints on ten paired scenes per range with seed 24000,
  deterministic ik, horizon 16, and a 500-step limit.
- select by near success; ties prefer higher total success, then the earlier
  checkpoint.
- confirm the selected near-weighted checkpoint against the fixed center-weighted
  step-1,100 baseline on 50 paired scenes per range with seed 24500.

the primary endpoint is paired near success. nominal plus far success is the
retention endpoint. adopt the intervention only if it improves near by at least
5/50 successes and retains at least 85% of the baseline's combined nominal and
far successes.

## result

pending.

## finding

pending.

## decision

- adopt near weighting only if it meets both the gain and retention criteria.
