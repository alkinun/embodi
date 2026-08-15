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

pending.

## finding

pending.

## decision

- treat interpolation as a diagnostic, not additional independent training.
- stop this line if interpolation does not materially improve the endpoint.
