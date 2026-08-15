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

## torque-off follower gate

with the arm physically supported:

```bash
uv run embodi-so101-preflight \
  --connect-hardware \
  --port=/dev/serial/by-id/<follower> \
  --robot-id=so101_follower_main \
  --canonical-calibration=configs/so101-physical-calibration.json \
  --reads=500
```

go criteria:

- all six expected motors respond;
- saved and on-motor calibration agree;
- all 500 readings are finite and inside conservative limits;
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
  --robot.cameras='{"top":{"type":"opencv","index_or_path":"/dev/v4l/by-id/<camera>","width":640,"height":480,"fps":30,"fourcc":"MJPG"}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/<leader> \
  --teleop.id=so101_leader_main \
  --fps=30 \
  --display_data=false
```

verify joint direction, gripper direction, reachable limits, backlash, camera
latency, and emergency power removal before recording demonstrations.

## recording

stock lerobot recording does not include embodi canonical labels. record an
immutable raw dataset first:

```bash
uv run embodi-record-so101 \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/<follower> \
  --robot.id=so101_follower_main \
  --robot.max_relative_target=5 \
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

## canonical calibration

copy `configs/so101-physical-calibration.example.json` to an ignored local file
named `configs/so101-physical-calibration.json`. before conversion, measure:

- the sign of each body joint relative to the pinned model;
- each model-zero offset in degrees;
- the native gripper value at physical closure;
- the native gripper value at the chosen fully open pose;
- the sha-256 of the lerobot follower calibration file.

validate the nominal forward kinematics against measured tool positions in at
least three separated poses. the current converter produces encoder-fk labels;
it cannot correct backlash, flex, assembly error, or an incorrect tool center.

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
names, finite values, calibrated model limits, post-clipping sent-action provenance,
a matching trusted recording manifest, a separate output root, and writes
canonical provenance to `meta/embodi.json`.

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

the first policy test should be 5 hz, horizon one, at most ten seconds, in an
empty padded workspace. do not begin with autonomous pick-and-place.
