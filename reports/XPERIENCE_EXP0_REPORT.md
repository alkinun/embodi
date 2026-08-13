# Xperience Experiment 0

The experiment history has been consolidated into concise records under
[`experiments/`](../experiments/README.md).

The final finding is documented in
[`005-expert-only-transfer.md`](../experiments/005-expert-only-transfer.md):
pretrain and transfer the action expert, keep the VLM frozen during egocentric
pretraining, initialize robot-domain adapters fresh, and train one decoder per
robot.

JSON metrics and diagnostic artifacts remain in this directory.
