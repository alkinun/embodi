# 027: selected-core native decoder viability

## question

can the selected general canonical policy retain its deterministic-ik control
quality when its canonical chunks are executed through a learned native-action
decoder, with low optimization variance across seeds?

## setup

- freeze the selected general checkpoint at
  `outputs/det-peak-exp20-center-m3993-l3992/step-0001100`.
- collect one deterministic-ik teacher cache from 60 nominal-range episodes with
  seed 27000, horizon 16, and a 500-step limit.
- split by episode: 50 train and 10 held out.
- reuse the exact cache and fixed decoder initialization for training seeds
  27001, 27002, and 27003.
- train only the native decoder for 2,500 updates with batch size 32 and learning
  rate `1e-4`; evaluate final checkpoints only.
- evaluate deterministic ik and all three learned decoders on the same 100 unseen
  nominal scenes with seed 27100, horizon 16, and a 500-step limit.
- perform no checkpoint, cache, scene, or seed selection.

the primary endpoint is mean learned-decoder closed-loop success across the three
seeds versus deterministic ik. declare the selected core's learned native path
viable only if mean success is no worse than deterministic ik by more than 5/100
and population standard deviation is at most 5 percentage points. require the
worst decoder seed to be no worse than deterministic ik by more than 10/100 as a
secondary robustness gate. paired scene outcomes, lift rate, validation loss,
and action clipping telemetry are secondary diagnostics and cannot override the
primary rule.

## result

the deterministic-ik path achieved 66/100 successes and 66 lifts. final learned
decoders achieved:

| training seed | validation loss | success | lifts |
| ---: | ---: | ---: | ---: |
| 27001 | `0.05396` | 18/100 | 19/100 |
| 27002 | `0.05469` | 21/100 | 23/100 |
| 27003 | `0.05760` | 20/100 | 23/100 |

mean learned success was 19.67/100 with population standard deviation 1.25
points. paired deterministic-only versus learned-only successes were 54 vs 6,
50 vs 5, and 49 vs 3 (`p < 2.2e-10` for each exact paired test). only 6 scenes
were solved by all learned decoders, while 36 were solved by at least one.

the fixed teacher cache contained 1,263 chunks from 60 episodes, with 36 teacher
successes and 37 lifts. raw evaluations are in
`reports/selected-core-exp27-{deterministic-ik-h16-100,learned-decoder-h16-3seed-100}.json`;
the compact result is in `reports/selected-core-exp27-summary.json`.

## finding

decoder optimization is reproducible, but the learned native path loses about
46/100 successes relative to deterministic ik. low and stable held-out MSE is
therefore not evidence of closed-loop decoder fidelity for the selected core.

## decision

- reject the learned native decoder as a physical-soak candidate.
- retain deterministic ik as the canonical-control reference only.
- diagnose temporal/action error and distribution shift before collecting more
  decoder data or training a physical policy.
