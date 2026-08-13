# 005: Expert-Only Transfer

## Question

Is useful Xperience transfer stored in the VLM adapters, the action expert, or
both?

## Comparison

- Baseline: fresh robot-domain adapters and fresh action expert.
- Full generalist: Xperience adapters and Xperience action expert.
- Backbone only: Xperience adapters and fresh action expert.
- Expert only: fresh robot-domain adapters and Xperience action expert.

## Result

Seed 3991, 50 paired scenes:

| Initialization | Success |
| --- | ---: |
| Baseline | 56% |
| Full generalist | 22% |
| Backbone only | 40% |
| Expert only | 88% |

Expert-only transfer replicated on seed 3992 at 60%, versus 42% baseline and
28% full generalist. Aggregate expert-only deterministic-IK success was 74%.

After policy-manifold decoder training, the best expert-only learned controller
reached 83/100 success versus its matched baseline at 56/100. Expert transfer
won 33 paired scenes uniquely; baseline won 6 uniquely. Decoder training itself
remains seed-sensitive.

## Finding

The reusable knowledge is in the approximately 53M-parameter action expert.
Human-domain VLM adapters cause negative visual-domain transfer.

## Decision

- Freeze the complete VLM during egocentric pretraining.
- Pretrain and transfer the action expert only.
- Initialize robot-domain VLM adapters/state path fresh.
- Train a small decoder per robot.
