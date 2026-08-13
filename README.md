# embodi

a vision-language-action model for cross-embodiment motion learning.

it uses `LiquidAI/LFM2.5-VL-450M` for vision and language, predicts variable
canonical part-token chunks with an embodiment-agnostic action expert, then maps
each part block to native robot commands with small per-robot decoders.

the current transfer recipe freezes the vlm during egocentric pretraining,
pretrains only the action expert, initializes robot-domain perception adapters
fresh, and trains a decoder per robot. see [`experiments/`](experiments/README.md).
the selected expert has 53.12m parameters (`width=512`, `layers=12`, `heads=8`).
decoder training uses one cached 40-episode policy-manifold dataset; three
optimizer seeds reached 89–90% simulator success.

## setup

```bash
uv sync --frozen --all-extras
```

uv creates `.venv` with python 3.12 from `.python-version`. `uv.lock` pins the
complete environment.

the lfm integration is pinned to transformers 5.1.x because its multimodal
hidden-state interface is version-sensitive.

## test

```bash
uv run pytest
uv run embodi-train --smoke-test
```

## xperience pretraining

Cache only the Xperience modalities required by experiment 0. Selection is
deterministic, the resolved Hugging Face revision is recorded, and the command
aborts before download when the bounded subset exceeds the size limit:

```bash
uv run python scripts/cache_xperience.py \
  --episodes 1 \
  --seed 0 \
  --max-download-gb 10 \
  --output-dir datasets/xperience-10m
```

Use `--dry-run` to inspect exact episode sizes first. Access to
`ropedia-ai/xperience-10m` requires accepting its controlled-access agreement
and authenticating with `hf auth login`. Only `annotation.hdf5` and
`stereo_left.mp4` are cached; rerunning reuses the local Hugging Face cache.

`configs/xperience-exp0.json` sets `freeze_backbone=true`. human pretraining
updates the action expert only and does not create an embodiment decoder.

## robot adaptation

transfer only the pretrained action expert. the robot keeps the foundational
vlm but initializes its trainable adapters and state path fresh:

```bash
uv run embodi-train \
  --dataset embodi/sim-so101-pickplace \
  --dataset-root datasets/embodi-sim-pickplace \
  --config configs/so101-exp0-top.json \
  --correct-image-rescale \
  --stage core \
  --init-core outputs/xperience-pretraining/final \
  --init-core-component expert \
  --init-embodiment outputs/so101-decoder/final \
  --output-dir outputs/so101-expert-transfer \
  --steps 50000
```

train the robot decoder without changing the action expert:

```bash
uv run embodi-train \
  --dataset embodi/sim-so101-pickplace \
  --dataset-root datasets/embodi-sim-pickplace \
  --config configs/so101-exp0-top.json \
  --correct-image-rescale \
  --stage decoder \
  --init-core outputs/xperience-pretraining/final \
  --init-core-component expert \
  --output-dir outputs/so101-decoder
```

checkpoints split transferable `core.pt` from robot-specific `embodiment.pt`.
use `--init-core-component expert` for cross-domain transfer.

## model

```text
images + instruction + canonical part state
                 |
      foundational lfm2.5-vl backbone
                 |
      post-gqa multimodal features
                  |
 embodiment-agnostic action expert
                  |
  32 x p x 10 canonical part tokens
                  |
       per-robot action decoder
                  |
    32 x native_action_dim command chunk
```

the foundational vlm is frozen during egocentric pretraining. only the action
expert is pretrained and transferred. robot training starts fresh lora/state
adapters for the robot visual domain. deterministic kinematics can supply
canonical state directly; each embodiment checkpoint owns its state adapter,
native normalization, safety behavior, and action decoder.

the core has no fixed part count. every controllable part uses one width-10
physical token, and a robot config composes any number `p` of those blocks. the
expert uses shared `10 -> width -> 10` projections, axial temporal/part
attention, and masks, so changing `p` does not change any core parameter shape.

each descriptor has a unique name, a physical kind, and a channel mask. names
such as `left_hand`, `right_hand`, `base`, or `camera_head` are encoded
deterministically and condition both state and action tokens. supported kinds
are `pose` (translation + rotation6d), `pose_scalar` (pose plus one scalar), and
`scalar`. masks specialize the primitive: a planar base can disable z, a head
can disable translation, and an arm without a gripper uses `pose`.

```json
{
  "parts": [
    {"name": "left_hand", "kind": "pose_scalar"},
    {"name": "right_hand", "kind": "pose_scalar"},
    {"name": "base", "kind": "pose", "channel_mask": [true, true, false, true, true, true, true, true, true, false]}
  ]
}
```

a pose action is reference-frame translation delta in meters and relative
rotation represented by the first two rotation-matrix columns. the optional
scalar is a normalized physical value such as gripper opening. a pose state has
absolute position/orientation in the same stable reference frame. generated
mujoco arm data computes both through fk before applying each native command.
native joint actions are never accepted as implicit canonical labels.

stored actions are frame-local. the training loader reconstructs each future ee
target from its matching canonical state and re-anchors the complete chunk to
the observation-time pose. canonical values use schema-fixed physical scales,
not per-robot dataset statistics, so a core checkpoint retains the same meaning
when another embodiment is attached.

canonical normalization uses fixed physical scales shared by every robot. only
native state/action statistics come from an embodiment dataset. inference
returns native actions in physical units; inactive absolute-position groups
explicitly hold their current state.

## batch

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

`canonical_action` is required for every stage. `action` is additionally required
for decoder and refinement. descriptors default to the config. padded core
batches can supply per-sample descriptors with any `p`; decoder/refinement
batches remain homogeneous because native dimensions and normalization are
embodiment-specific. sequential multi-robot pretraining loads the same
`core.pt` under different robot configs.

## simulation

run the trained checkpoint in the so-101 mujoco pick-and-place scene:

```bash
uv run embodi-sim
```

open `http://localhost:8080` or `http://<machine-ip>:8080`. the server binds to
`0.0.0.0`, streams the top camera with a wrist-camera inset, and exposes an
episode reset button. actions execute at 30 hz while the next chunk is inferred.
the default execution horizon is the full 32-step horizon used during training.

the first run downloads the pinned apache-2.0 mujoco menagerie so-101 assets to
`~/.cache/embodi`. verify mujoco without loading the policy with:

```bash
uv run embodi-sim --smoke-test
```

## synthetic data

verify the privileged ik expert:

```bash
uv run embodi-generate --evaluate-only --episodes 100
```

generate 500 successful top-only demonstrations locally:

```bash
uv run embodi-generate \
  --episodes 500 \
  --cameras top \
  --root datasets/embodi-sim-pickplace \
  --repo-id embodi/sim-so101-pickplace
```

use `--cameras top wrist` with a separate root to generate a two-camera
dataset. training must request the same camera names and order.

preview evenly spaced frames from a generated episode:

```bash
uv run embodi-preview --episode 0 --output outputs/dataset-preview.jpg
```

failed oracle attempts are discarded. successful episodes are stored at 30 hz
in lerobot v3 format with images, native state/action, `[p,10]` canonical action
and state blocks, part masks, and the task instruction. `meta/embodi.json` stores
the stable name/kind/channel descriptors and must match the training config.
datasets generated before model format version 3 must be regenerated or
converted through robot-specific kinematics before token-core training.
