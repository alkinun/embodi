# embodi

embodi is a vision-language-action model for cross-embodiment motion learning.

the model uses `LiquidAI/LFM2.5-VL-450M` for vision and language. an action
expert predicts canonical motion. a small decoder maps that motion to robot
commands.

the current recipe is simple:

- freeze the vlm during human pretraining.
- pretrain and transfer only the action expert.
- start robot perception adapters and state paths from scratch.
- train one decoder for each robot.

see the [experiment index](experiments/index.md) for evidence and decisions and
the [data research plan](docs/data-research.md) for scaling and generalization
criteria.

the selected checkpoints are canonical policies validated in simulation with
deterministic ik. they are not physical-arm controllers. physical setup and
acceptance gates are documented in [docs/physical-so101.md](docs/physical-so101.md).

## setup

```bash
uv sync --frozen --all-extras
```

`uv` uses python 3.12 from `.python-version`. `uv.lock` pins the environment.
the lfm integration needs transformers 5.1.x.

## checks

```bash
uv run pytest
uv run embodi-train --smoke-test
uv run embodi-sim --smoke-test
```

## model

```text
images + instruction + canonical state
                  |
             frozen vlm
                  |
            action expert
                  |
       canonical motion tokens
                  |
          robot decoder
                  |
        native robot commands
```

the selected expert has 53.12M parameters. it uses `width=512`, `layers=12`,
and `heads=8`.

each controllable part uses one 10-channel token:

| channels | meaning |
| --- | --- |
| `0:3` | translation |
| `3:9` | rotation6d |
| `9` | scalar, such as gripper opening |

part kinds are `pose`, `pose_scalar`, and `scalar`. channel masks can disable
unused values. part names condition state and action tokens.

```json
{
  "parts": [
    {"name": "left_hand", "kind": "pose_scalar"},
    {"name": "right_hand", "kind": "pose_scalar"},
    {"name": "base", "kind": "pose", "channel_mask": [true, true, false, true, true, true, true, true, true, false]}
  ]
}
```

pose actions use translation deltas in meters. rotation is relative and uses
the first two matrix columns. pose states use absolute values in a stable
reference frame.

canonical values use fixed physical scales. native state and action statistics
belong to each embodiment. checkpoints split transferable `core.pt` from
robot-specific `embodiment.pt`.

## batch contract

```text
observation.state              float [b, native_state_dim]
canonical_state               float [b, p, 10]
action                         float [b, 32, native_action_dim]
canonical_action               float [b, 32, p, 10]
canonical_part_mask            bool  [b, p]
canonical_channel_mask         bool  [b, p, 10]
canonical_part_kind            int64 [b, p]
canonical_part_name_features   float [b, p, 32]
action_group_mask              bool  [b, embodiment_groups]
action_is_pad                  bool  [b, 32]
```

`canonical_action` is required for every stage. decoder training also needs
`action`. decoder batches must use one embodiment.

## xperience cache

accept the controlled-access agreement first. then run `hf auth login`.

```bash
uv run python scripts/cache_xperience.py \
  --episodes 1 \
  --seed 0 \
  --max-download-gb 10 \
  --output-dir datasets/xperience-10m
```

use `--dry-run` to inspect sizes. the cache records the resolved revision. it
downloads only the required hdf5 labels and left stereo video.

`configs/xperience-exp0.json` freezes the backbone. human pretraining updates
only the action expert. it does not create a decoder.

## robot training

generate a current format-v3 dataset:

```bash
uv run embodi-generate \
  --episodes 500 \
  --cameras top \
  --root datasets/embodi-sim-pickplace \
  --repo-id embodi/sim-so101-pickplace
```

train a scratch baseline:

```bash
uv run embodi-train \
  --dataset embodi/sim-so101-pickplace \
  --dataset-root datasets/embodi-sim-pickplace \
  --config configs/so101-exp0-top.json \
  --stage core \
  --output-dir outputs/so101-baseline \
  --steps 2500
```

for transfer, add these arguments:

```text
--init-core outputs/<human-run>/final
--init-core-component expert
```

train decoders on policy predictions. reuse one cached teacher dataset across
comparisons. see `scripts/distill_so101_decoder.py`.

physical lerobot recordings require calibrated canonical labels before training:

```bash
uv run embodi-init-so101-calibration \
  --lerobot-calibration /path/to/so101_follower_main.json \
  --output configs/so101-physical-calibration.json

uv run embodi-convert-so101 \
  --source-root datasets/physical-so101-raw \
  --source-repo-id embodi/physical-so101-raw \
  --output-root datasets/physical-so101 \
  --output-repo-id embodi/physical-so101 \
  --calibration configs/so101-physical-calibration.json \
  --lerobot-calibration /path/to/so101_follower_main.json
```

do not use the example calibration unchanged. measure joint signs, model zero
offsets, and physical gripper endpoints on the assembled follower, validate FK,
then set `physical_validation_complete` to `true`.
Record with `embodi-record-so101 --robot.max_relative_target=5`, not the stock
LeRobot recorder, so the dataset stores post-clipping commands and immutable
file provenance.
Physical training admission verifies the completed conversion inventory and
rejects modified data, incomplete conversion, schema mismatches, or invalid stats.

## evaluation

run closed-loop evaluation:

```bash
uv run python scripts/evaluate_so101.py \
  outputs/<robot-run>/final \
  --episodes 100 \
  --seed 65000 \
  --max-steps 500 \
  --execution-horizon 16 \
  --output reports/evaluation.json
```

offline loss is not enough. use paired scenes and multiple training seeds.

## simulation server

```bash
uv run embodi-sim --checkpoint outputs/<robot-run>/final
```

open `http://localhost:8080`. the server binds to localhost by default. pass an
explicit `--host` only when remote access is required and protected.

the first run caches pinned apache-2.0 mujoco assets in `~/.cache/embodi`.

## dataset preview

```bash
uv run embodi-preview \
  --dataset embodi/sim-so101-pickplace \
  --dataset-root datasets/embodi-sim-pickplace \
  --episode 0 \
  --output outputs/dataset-preview.jpg
```

generated datasets use lerobot v3. `meta/embodi.json` stores part descriptors.
the descriptors must match the model config.

Frozen multi-task benchmark demonstrations are generated into separate native
datasets for each morphology:

```bash
uv run --extra train --extra sim embodi-generate-benchmark \
  benchmarks/sim-v1/definition.json \
  benchmarks/sim-v1/manifests/train.json \
  --root datasets/embodi-sim-v1-train \
  --repo-prefix embodi/sim-v1-train

uv run --extra train --extra sim embodi-generate-benchmark \
  benchmarks/sim-v1/definition.json \
  benchmarks/sim-v1/manifests/validation.json \
  --root datasets/embodi-sim-v1-validation \
  --repo-prefix embodi/sim-v1-validation
```

Use `--resume` only for an incomplete root created by the exact same command.
Generation consumes frozen scenarios in order and aborts rather than replacing
an oracle failure. `benchmark-dataset.json` binds source hashes, episode/scenario
mapping, clipping diagnostics, assets, and every generated payload file.

Train one embodiment at a time and provide its matching frozen validation root:

```bash
uv run embodi-train \
  --dataset embodi/sim-v1-train-so101 \
  --dataset-root datasets/embodi-sim-v1-train/so101 \
  --validation-dataset embodi/sim-v1-validation-so101 \
  --validation-dataset-root datasets/embodi-sim-v1-validation/so101 \
  --config configs/so101-exp0-top.json \
  --stage core \
  --output-dir outputs/sim-v1-so101-core
```

Benchmark lineage validation rejects modified payloads, missing scenario mappings,
validation data used as training data, and benchmark training without the frozen
validation dataset.
