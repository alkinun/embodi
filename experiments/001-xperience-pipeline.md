# 001: Xperience Pipeline

## Question

Can Xperience egocentric hand motion be converted into the same canonical action
space used by SO-101?

## Setup

- 100 balanced clips from one Xperience episode.
- `primary_effector / pose_scalar`, 10 channels.
- Initial-root coordinates, relative rotation6d, normalized hand opening.
- 32 steps at 30 Hz.

## Result

- 2,932 valid anchors passed synchronization and geometry checks.
- Training loss: `1.0529 -> 0.1053` over 2,000 steps.
- Held-out loss: `1.0686 -> 0.1642`.

## Decision

The data and canonical conversion pipeline are usable. This is a 100-clip
pipeline experiment, not evidence about large-scale Xperience pretraining.
