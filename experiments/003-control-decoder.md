# 003: Control And Decoder

## Question

Why did accurate canonical predictions fail to produce robot success?

## Findings

- Native privileged controller: 20/20 success.
- Exact canonical commands through deterministic IK: 20/20 success.
- Original learned decoder: 0/20 success.
- Residual decoder on 100 demonstrations: 15/20 with exact canonical commands.
- Configured per-chunk `2 degree` arm limits reduced the IK teacher to 0/20.
- Removing those mismatched limits recovered learned-controller success.

Training the decoder on deterministic-IK labels at states visited by the policy
raised learned baseline control to 62% and 56% in two core runs.

## Decision

- Disable the stale per-chunk action limits.
- Train robot decoders on the policy-prediction manifold, not only exact
  demonstration labels.
- Keep decoder evaluation closed-loop; decoder MSE is not a reliable selector.
