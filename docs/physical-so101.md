# physical so101 bring-up

the first physical milestone is one stationary follower arm, one leader arm,
and one fixed head camera. current selected checkpoints were evaluated with
simulation kinematics and must not be sent directly to hardware.

## safety boundary

- mount the follower rigidly before connecting power.
- keep the workspace empty and support the arm before disabling torque.
- use a physical power-cut emergency stop within the operator's reach.
- never rely on a python process as the only emergency stop.
- do not run policy motion during initial calibration or observation checks.
- do not execute open-loop action chunks on first bring-up.

`embodi-so101-preflight` is observation-only. it opens the motor bus, disables
torque, validates calibration, reads positions, and disconnects with torque off.
there is intentionally no autonomous hardware rollout command yet.

## before arrival

```bash
uv sync --frozen --extra train --extra sim --extra test --extra hardware
uv run embodi-so101-preflight \
  --canonical-calibration configs/so101-physical-calibration.example.json
uv run pytest
```

the example calibration proves file parsing only. it is not valid physical
kinematic calibration.

## ports and calibration

find each controller port separately:

```bash
uv run lerobot-find-port
```

use stable `/dev/serial/by-id/...` paths after identifying the ports. only run
motor setup if the assembled arm documentation requires assigning motor ids:

```bash
uv run lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/<leader>

uv run lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/<follower>
```

calibrate the leader and follower under distinct, stable ids:

```bash
uv run lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/<leader> \
  --teleop.id=so101_leader_main

uv run lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/<follower> \
  --robot.id=so101_follower_main
```

preserve the generated calibration files and record their sha-256 hashes. do
not edit or replace them after collecting a dataset without creating a new
dataset revision.

initialize the ignored canonical calibration from the exact follower file:

```bash
uv run embodi-init-so101-calibration \
  --lerobot-calibration /path/to/so101_follower_main.json \
  --output configs/so101-physical-calibration.json
```

this only binds provenance. the generated file remains intentionally invalid
for hardware and conversion until the measured fields are updated and
`physical_validation_complete` is set to `true` after the FK checks below.

with the follower torque off, measure and enter:

- the sign of each body joint relative to the pinned model;
- each model-zero offset in degrees;
- the native gripper value at physical closure;
- the native gripper value at the chosen fully open pose.

validate nominal forward kinematics against measured full tool poses in the
physical base frame at no fewer than three separated configurations. include
translation and orientation, define the physical tcp to match `gripperframe`,
and include wrist-roll-sensitive poses. the converter produces encoder-fk
labels; it cannot correct backlash, flex, assembly error, base-axis mismatch,
or an incorrect tool center. only then set `physical_validation_complete=true`.

for this gate, place the base origin at the model's base yaw axis on the mounting
plane, use +z upward, +x from the base toward the nominal workspace, and +y to
complete a right-handed frame. use at least five separated poses spanning the
working volume; every measured tcp must be within 15 mm and 5 degrees of model
fk, including at least two poses that isolate wrist-roll orientation.

## torque-off follower gate

with the arm physically supported:

```bash
uv run embodi-so101-preflight \
  --connect-hardware \
  --port=/dev/serial/by-id/<follower> \
  --robot-id=so101_follower_main \
  --canonical-calibration=configs/so101-physical-calibration.json \
  --raw-only \
  --reads=500
```

the first `--raw-only` pass verifies calibration-file identity, on-motor
calibration, torque state, native values, and disconnect behavior without
trusting unfinished canonical signs or offsets. after completing the full-pose
measurements above, rerun the same command without `--raw-only`.
the JSON report includes native per-joint minima and maxima for both passes and
calibrated model-coordinate ranges for the validated pass; retain both reports
with the calibration measurements.

go criteria:

- all six expected motors respond;
- saved and on-motor calibration agree;
- the validated pass maps all 500 readings inside model and gripper limits;
- torque remains off;
- disconnect leaves the arm torque-free.

## camera gate

discover the fixed head camera and use its stable by-id path:

```bash
uv run lerobot-find-cameras opencv
```

record at 640x480 rgb and 30 hz. mount the camera rigidly before collecting any
training data. save sample images and verify orientation, field of view, and the
entire reachable workspace manually.

## teleoperation

start with low-speed, supervised leader-follower teleoperation. no policy is
involved:

```bash
uv run lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/<follower> \
  --robot.id=so101_follower_main \
  --robot.max_relative_target=5 \
  --robot.disable_torque_on_disconnect=true \
  --robot.cameras='{"top":{"type":"opencv","index_or_path":"/dev/v4l/by-id/<camera>","width":640,"height":480,"fps":30,"fourcc":"MJPG"}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/<leader> \
  --teleop.id=so101_leader_main \
  --fps=30 \
  --display_data=false
```

verify joint direction, gripper direction, reachable limits, backlash, camera
latency, and emergency power removal before recording demonstrations.
if teleoperation or recording presents an unexpected calibration prompt, stop;
do not accept or rewrite calibration in a powered-motion workflow. rerun
calibration followed by the torque-off preflight instead.

## recording

stock lerobot recording does not include embodi canonical labels. record an
immutable raw dataset first:

```bash
uv run embodi-record-so101 \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/<follower> \
  --robot.id=so101_follower_main \
  --robot.max_relative_target=5 \
  --robot.disable_torque_on_disconnect=true \
  --robot.cameras='{"top":{"type":"opencv","index_or_path":"/dev/v4l/by-id/<camera>","width":640,"height":480,"fps":30,"fourcc":"MJPG"}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/<leader> \
  --teleop.id=so101_leader_main \
  --dataset.repo_id=embodi/physical-so101-raw \
  --dataset.root=datasets/physical-so101-raw \
  --dataset.no_stamp=true \
  --dataset.single_task="Pick up the cube and place it in the box." \
  --dataset.fps=30 \
  --dataset.num_episodes=20 \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --play_sounds=false \
  --display_data=false
```

keep `--robot.max_relative_target=5` while recording. the stock lerobot 0.6.1
recorder discards the value returned after relative-target clipping, so do not
substitute `lerobot-record`: the embodi wrapper stores the command actually sent
and writes `meta/embodi-recording.json` with its calibration and safety provenance.
resume is intentionally unsupported because existing episodes cannot be proven to
share that provenance; create a new immutable raw dataset for each recording run.
the wrapper is pinned to lerobot 0.6.1, refuses hub upload during collection, and
writes a completed manifest only after local finalization. the manifest binds the
repo id, fps, schema, frame and episode counts, and every metadata, parquet, and
video file by size and sha-256. conversion rejects any later modification.

## conversion

```bash
uv run embodi-convert-so101 \
  --source-root datasets/physical-so101-raw \
  --source-repo-id embodi/physical-so101-raw \
  --output-root datasets/physical-so101 \
  --output-repo-id embodi/physical-so101 \
  --calibration configs/so101-physical-calibration.json \
  --lerobot-calibration /path/to/so101_follower_main.json
```

`--lerobot-calibration` must be the unchanged follower calibration used for
recording; its sha-256 must match `source_calibration_sha256` in the canonical
calibration. conversion is fail-closed: it requires 30 hz data, exact six-joint
names, exactly one 640x480 rgb `top` video, finite values, calibrated model and
gripper limits, post-clipping sent-action provenance, an unchanged immutable
recording manifest, a separate output root, and writes canonical provenance to
`meta/embodi.json`.

the converted dataset also receives a completed `meta/conversion.json` inventory
covering its metadata, canonical parquet data, and videos. treat the output as
immutable. `embodi-train` verifies this inventory, physical calibration status,
feature schemas, cameras, normalization statistics, and dataset counts before
constructing loaders; modified or partially converted physical data is rejected.
the admitted data manifest is copied into every checkpoint and must match on
resume; physical checkpoints without verified dataset lineage cannot be resumed.
physical datasets contain only the fixed `top` camera, so every physical training
invocation must use a top-camera config such as `configs/so101-exp0-top.json` or
pass `--cameras top`; the default two-camera model configuration is intentionally
rejected.

## policy-motion gate

do not add autonomous motion until all of these pass:

- deterministic scripted joint motions stay inside 0.5 degrees per command;
- gripper motion stays inside 1 percentage point per command;
- a startup-centered 5-degree session envelope cannot be exceeded;
- camera, inference, or motor failure latches a stop and disables torque;
- the complete native-state to learned-decoder path succeeds in simulation;
- a five-minute observation-only policy soak produces finite actions with low
  safety-filter clipping;
- a physical power-cut stop is held by an operator.

the current selected checkpoints are not eligible for the observation-only soak:
they were validated through canonical policy plus deterministic simulator ik,
not the learned physical-native decoder. no hardware soak command is exposed
until a physical dataset has trained or validated that complete native path and
a clipping acceptance threshold has been pre-registered.

the first policy test should be 5 hz, horizon one, at most ten seconds, in an
empty padded workspace. do not begin with autonomous pick-and-place.
