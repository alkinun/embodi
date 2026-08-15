# 021: control-peak interpolation

## question

does the step-1,100 control peak occupy a connected region between adjacent
checkpoints, and can weight interpolation improve its robustness?

## setup

- use deterministic model seed 3993 checkpoints at steps 1,000, 1,100, and
  1,200 from experiment 020.
- linearly interpolate all floating core and embodiment tensors.
- create alpha 0.25, 0.50, and 0.75 interpolants from step 1,000 to 1,100 and
  from step 1,100 to 1,200.
- screen the six interpolants and three endpoints on ten paired scenes per near,
  nominal, and far range with seed 21000, deterministic ik, horizon 16, and a
  500-step limit.
- rank by total success over 30 scenes; ties prefer the checkpoint closest to
  step 1,100, then the lower effective step.

an interpolant replaces step 1,100 only if it exceeds the endpoint by at least
3/30 successes and succeeds in every range. such a candidate is confirmed
against step 1,100 on 50 paired scenes per range with new seed 21500. otherwise
retain step 1,100 without confirmation.

## result

screen successes over ten scenes per range were:

| effective step | source | near | nominal | far | total |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,000 | endpoint | 3 | 3 | 3 | 9/30 |
| 1,025 | 1,000→1,100, alpha 0.25 | 4 | 5 | 5 | 14/30 |
| 1,050 | 1,000→1,100, alpha 0.50 | 5 | 4 | 4 | 13/30 |
| 1,075 | 1,000→1,100, alpha 0.75 | 5 | 4 | 4 | 13/30 |
| 1,100 | endpoint | 1 | 7 | 7 | 15/30 |
| 1,125 | 1,100→1,200, alpha 0.25 | 4 | 5 | 7 | 16/30 |
| 1,150 | 1,100→1,200, alpha 0.50 | 3 | 4 | 4 | 11/30 |
| 1,175 | 1,100→1,200, alpha 0.75 | 5 | 2 | 3 | 10/30 |
| 1,200 | endpoint | 3 | 3 | 2 | 8/30 |

the best interpolant exceeded step 1,100 by only 1/30, below the preregistered
3/30 replacement threshold. no confirmation was run.

raw rollouts are in
`reports/interp-exp21-{near,nominal,far}-screen-ik-h16-9model-10.json`; the
compact result is in `reports/interp-exp21-summary.json`.

## finding

the control peak occupies a connected local weight-space region. every
interpolant between steps 1,000 and 1,100 scores 13-14/30, and the first
interpolant beyond step 1,100 scores 16/30. performance then declines toward
step 1,200.

linear interpolation smooths the apparent checkpoint-to-checkpoint
discontinuity but does not materially improve the confirmed step-1,100 policy.
the temporal instability is therefore not evidence of completely disconnected
solutions along this local path.

## decision

- retain step 1,100 as the selected deterministic checkpoint.
- stop checkpoint interpolation; it maps the basin but does not improve the
  endpoint enough to justify confirmation.
- diagnose near-range pre-lift failures before changing training again.
