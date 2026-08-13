# 001: xperience pipeline

## question

can xperience hand motion use the same canonical space as so-101?

## setup

- 100 balanced clips from one xperience episode.
- `primary_effector / pose_scalar`, 10 channels.
- initial-root coordinates, relative rotation6d, and normalized hand opening.
- 32 steps at 30 hz.

## result

- 2,932 valid anchors passed synchronization and geometry checks.
- training loss: `1.0529 -> 0.1053` over 2,000 steps.
- held-out loss: `1.0686 -> 0.1642`.

## finding

the conversion pipeline works.

## decision

use the pipeline for larger studies. do not treat this 100-clip run as scaling
evidence.
