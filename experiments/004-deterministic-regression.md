# 004: deterministic regression

## question

was flow integration hiding transfer behind sampling error?

## setup

compare flow inference with deterministic trajectory regression.

## result

vector-field error dominated integration error. regression produced useful
control. the first transfer gain did not replicate.

| training steps | baseline | full transfer |
| ---: | ---: | ---: |
| 2,500, first run | 10% | 46% |
| 5,000 | 36% | 34% |

two more 2,500-step seeds reversed the first result.

## finding

regression removes avoidable sampling noise. one seed is not enough.

## decision

use regression for so-101 studies. use multiple seeds and paired scenes.
