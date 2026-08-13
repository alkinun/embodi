# 008: xperience data scaling

## question

does more xperience data improve human loss and so-101 transfer?

## setup

- eight episodes from eight sessions at revision
  `ce943cf271a758b60240084892d05cf6dc12dd90`.
- 18,574 valid anchors. one episode had invalid labels.
- fixed whole-episode split.
- motion-balanced budgets of 100, 1,000, and 10,000 clips.
- frozen vlm and fixed 53.12m expert.
- 1,000 human-pretraining updates for every data budget.
- expert-only transfer with 1,500 so-101 updates on 100 demonstrations.
- 30 paired scenes, seed 63000, deterministic ik, horizon 8.

## result

| human clips | human validation loss | robot validation loss | lift | success |
| ---: | ---: | ---: | ---: | ---: |
| 100 | `0.4380` | `0.000502` | 14/30 (47%) | 11/30 (37%) |
| 1,000 | `0.2217` | `0.000375` | 18/30 (60%) | 15/30 (50%) |
| 10,000 | `0.1950` | `0.000545` | 14/30 (47%) | 13/30 (43%) |

human loss improved at each budget. control peaked at 1,000 clips. the 10,000
clip run had only 3.2 effective passes. this screen is optimization-limited.

an earlier run used `image_do_rescale=true` and scored 0/30. it is invalid. the
config now locks `image_do_rescale=false`.

## finding

more data improves human loss. robot transfer needs matched exposure.

## decision

use 1,000 clips at the fixed 1,000-update budget. rerun 10,000 clips with a
longer matched-exposure schedule.

control results are in
`reports/so101-xperience-multi8-data-scaling-rescale-h8-30.json`.
