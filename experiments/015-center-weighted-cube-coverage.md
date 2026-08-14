# 015: center-weighted cube coverage

## question

can a center-weighted cube distribution improve edge success without repeating
the nominal forgetting from uniform-wide training?

## setup

- compare nominal-only, uniform-wide, and center-weighted datasets.
- use 105 successful demonstrations per dataset and the same top camera.
- reuse the experiment 011 nominal-only and uniform-wide datasets.
- allocate center-weighted episodes as 53 nominal, 26 near, and 26 far.
- stratify the center-weighted split as 50/25/25 training episodes and 3/1/1
  validation episodes.
- use generator seed 11000 for all datasets.
- use the same 1,000-clip human expert checkpoint.
- train cores with seeds 3991 and 3992 for 2,500 updates at batch size 32.
- hold initialization, config, split size, optimizer, and schedule fixed.
- use deterministic ik with execution horizon 16.
- evaluate 50 paired scenes in each range with seed 15500 and a 500-step limit.

the primary comparisons are center-weighted versus nominal-only on combined
near and far success, and center-weighted versus uniform-wide on nominal
success. an effect counts as replicated only if its direction agrees across
both training seeds. per-seed paired outcomes will be reported without pooling
cores as independent scenes.

## result

pending.

## finding

pending.

## decision

- keep the protocol fixed until all six cores finish evaluation.
- make no coverage claim from validation loss alone.
- report mixed seed directions as inconclusive.
