# 004: Deterministic Regression

## Question

Was flow integration hiding useful transfer behind sampling error?

## Result

Sampling diagnostics showed vector-field error dominated numerical integration
error. Replacing flow inference with deterministic trajectory regression
produced nontrivial control, but the full-generalist advantage was seed-sensitive:

| Training steps | Baseline | Full generalist |
| ---: | ---: | ---: |
| 2,500, first run | 10% | 46% |
| 5,000 | 36% | 34% |

Two additional 2,500-step seeds reversed the initial result.

## Decision

Use deterministic regression for controlled SO-101 transfer studies. Evaluate
multiple training seeds and paired simulator scenes.
