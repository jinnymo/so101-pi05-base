# 06 — Inference and hardware evaluation

How to run the `so101-pi05-base` checkpoint in closed loop on a physical SO-101 arm.

Everything here was run on one machine with one arm. Where a number was not measured it is marked
`not recorded` rather than estimated.

Contents:

1. [Prerequisites](#1-prerequisites)
2. [Environment](#2-environment)
3. [Patches on top of upstream VLASH](#3-patches-on-top-of-upstream-vlash)
4. [Checkpoint layout](#4-checkpoint-layout)
5. [Robot and camera configuration](#5-robot-and-camera-configuration)
6. [Runtime options](#6-runtime-options)
7. [Validated operating configuration](#7-validated-operating-configuration)
8. [Evaluation procedure](#8-evaluation-procedure)
9. [Observed behavior and limits](#9-observed-behavior-and-limits)
10. [Safety](#10-safety)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

| Item | What was used |
|---|---|
| Robot | SO-101 follower arm, 6 DoF, Feetech STS3215 servos, USB serial (`/dev/ttyACM0`) |
| Cameras | 2 USB UVC cameras: one overhead view of the workspace, one wrist view |
| GPU | NVIDIA RTX 3090 Ti, 24 GB, driver 590.48.01 |
| OS | Linux (kernel 6.x), V4L2 |
| Checkpoint | step 40000, `model.safetensors` 7,481,485,688 bytes |

The policy is about 3.62B parameters in bfloat16, so the weights alone occupy roughly 7 GiB of
VRAM. Peak VRAM during inference was not recorded; a 24 GB card had ample headroom with
`torch.compile` enabled.

Time budget for a first run, from a clean machine:

| Step | Time |
|---|---|
| Create the environment, install VLASH | 15-30 min (dominated by the torch download) |
| Download the checkpoint (7.0 GiB) | depends on link speed |
| Identify camera device nodes, write the config | 10-20 min |
| Dry run | ~1 min per attempt after startup |
| Startup, no `compile_model` | about 25 s |
| Startup, `compile_model: true` | about 2-3 min (torch.compile warmup) |

The two startup figures were measured on this machine with pi0.5 checkpoints of the same size; they
were not timed separately for this checkpoint.

---

## 2. Environment

### 2.1 Install

```bash
conda create -n so101-eval python=3.10
conda activate so101-eval
conda install ffmpeg=7.1.1 -c conda-forge

git clone https://github.com/mit-han-lab/vlash
cd vlash
git checkout 22cbabfee0f57874987c75a35a7dac129e695db0
pip install -e .
pip install -U torch torchvision torchcodec
```

`pip install -e .` pulls `lerobot[feetech,smolvla]==0.4.1` transitively, which brings the SO-101
driver, the OpenCV camera backend and the LeRobot dataset utilities that VLASH's inference loop
uses. An editable install is recommended because the patches in section 3 are edits to VLASH source
files.

The checkout is pinned to the commit the training image builds from — `ARG VLASH_COMMIT` in
`training-docker/Dockerfile` holds the same value. The patches in section 3 are whole-file
replacements written against that revision, so on a moved upstream tree they would revert
unrelated changes or fail against a shifted API.

### 2.2 Versions actually used

Resolved versions in the environment that produced the results in section 9:

| Package | Version |
|---|---|
| python | 3.10.20 |
| torch | 2.7.1+cu126 |
| torchvision | 0.22.1 |
| torchcodec | 0.5 |
| lerobot | 0.4.1 |
| transformers | 4.53.3 |
| peft | 0.18.0 |
| numpy | 2.2.6 |
| opencv-python | 4.13.0.92 |
| draccus | 0.10.0 |
| ffmpeg | 7.1.1 (conda-forge) |

VLASH's `pyproject.toml` pins `transformers` to a git commit rather than a release tag; 4.53.3 is
what that commit resolved to here. If a later resolution breaks, pin `transformers==4.53.3`.

### 2.3 Tokenizer

pi0.5 tokenizes the language instruction with the PaliGemma tokenizer. Upstream VLASH loads it as:

```python
AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
```

`google/paligemma-3b-pt-224` is a gated repository. Before the first run, accept its terms on the
Hub and authenticate:

```bash
hf auth login
```

`hf` is the current Hugging Face Hub CLI. The older `huggingface-cli` entry point is deprecated
and was reduced to a stub that prints a notice and exits non-zero, so older instructions using it
no longer work.

For a machine with no network access, download the tokenizer files once on a connected machine and
point the policy at the local directory (see the tokenizer patch in section 3.4). The directory
needs these files:

```
added_tokens.json  config.json  generation_config.json
preprocessor_config.json  special_tokens_map.json
tokenizer.json  tokenizer.model  tokenizer_config.json
```

`model.safetensors.index.json` may also be present; the weights themselves are not needed, only the
tokenizer.

### 2.4 Environment variables

| Variable | Purpose | If unset |
|---|---|---|
| `VLASH_PALIGEMMA_PATH` | local tokenizer directory (only exists with the patch in 3.4) | the patched build falls back to a hardcoded container path; if that path does not exist, `transformers` treats the string as a repo id and raises `HFValidationError` |
| `HF_HUB_OFFLINE=1` | forbid all Hub lookups | the Hub is contacted on every load; on a machine without credentials for the gated tokenizer repo this fails at load time |
| `TRANSFORMERS_OFFLINE=1` | same, for the `transformers` cache path | as above |
| `VLASH_DRY_RUN=1` | compute and print actions without commanding the motors (patch 3.2) | actions are sent to the arm |
| `VLASH_SAFE_DEG` | abort threshold on action magnitude (patch 3.2), default 180 | 180 |

Setting `HF_HUB_OFFLINE=1` with an incomplete cache does not fall back to a download; the load
fails with an offline-mode error. Populate the cache first, then go offline.

---

## 3. Patches on top of upstream VLASH

The checkpoint itself is a plain pi0.5 checkpoint and needs no custom code to load. Four details of
the run loop do need changes. Patch 3.1 is **required** if you have fewer than three cameras, which
is the normal case for SO-101. Patches 3.2 and 3.3 were used for the evaluation in section 9 and are
strongly recommended. Patch 3.4 is only for offline installs.

Three of the four (3.1, 3.2, 3.3) are edits to `vlash/run.py`; 3.4 is in
`vlash/policies/pi05/modeling_pi05.py`. Both files ship already patched in
`training-docker/patched/`, so an alternative to applying the diffs by hand is to copy those two
over an editable VLASH checkout. They are the same files the training image uses — `run.py` is in
that image for exactly this reason, since training never calls it.

### 3.1 Allow a camera subset (required for fewer than 3 cameras)

Upstream `validate_robot_cameras` demands an exact match between the robot's camera keys and the
policy's image features. The policy has three slots; a two-camera arm therefore fails at startup
with `Robot camera names must exactly match policy image feature names!` before any inference
happens. The fix is to accept a subset and let the empty-slot machinery fill the rest.

```diff
--- a/vlash/run.py
+++ b/vlash/run.py
@@ def validate_robot_cameras(robot, policy_config):
-    # Strict match required
-    if robot_image_features != policy_camera_features:
+    extra = robot_image_features - policy_camera_features
+    if extra:
         raise ValueError(
-            "Robot camera names must exactly match policy image feature names!\n"
-            f"Robot cameras (with prefix): {sorted(robot_image_features)}\n"
+            "Robot has cameras the policy does not expect!\n"
+            f"Unexpected robot cameras: {sorted(extra)}\n"
             f"Policy image features: {sorted(policy_camera_features)}\n"
-            "Please ensure camera configuration matches the trained model."
+            "Rename robot.cameras keys to match policy slot names."
         )
+    missing = policy_camera_features - robot_image_features
+    if missing:
+        logging.warning(
+            "Policy camera slot(s) not provided by robot: %s "
+            "- auto-filled as masked dummies when policy.empty_cameras > 0.",
+            sorted(missing),
+        )
```

A camera key the policy does not know is still rejected, which is the useful half of the original
check: a typo in a slot name should not silently degrade into a masked slot.

### 3.2 Dry run and action magnitude abort

Upstream sends every action straight to the arm. Two guards make bring-up survivable: a mode that
computes actions without commanding the motors, and a hard abort when an action leaves a plausible
range (which is what a bad checkpoint path, wrong normalization or NaN looks like at the joint
level).

```diff
--- a/vlash/run.py
+++ b/vlash/run.py
@@ def run_loop(...):
         action = async_manager.get_action(observation_frame)
 
+        import os as _os
+        _dry_run = _os.environ.get("VLASH_DRY_RUN", "0") == "1"
+        _safe_range = float(_os.environ.get("VLASH_SAFE_DEG", "180"))
+
+        action_dict = action if isinstance(action, dict) else None
+        if action_dict is not None:
+            unsafe = {k: v for k, v in action_dict.items() if abs(float(v)) > _safe_range}
+            if unsafe or (step_count % 10 == 0):
+                print(f"[step {step_count}] action: "
+                      f"{dict((k, round(float(v), 1)) for k, v in action_dict.items())}")
+            if unsafe:
+                print(f"UNSAFE action (>{_safe_range}): {unsafe} - aborting")
+                break
+
         if (step_count + 1) % action_quant_ratio == 0:
-            robot.send_action(action)
+            if not _dry_run:
+                robot.send_action(action)
```

On the units: with LeRobot's default `use_degrees: false`, SO-101 arm joints are normalized to
-100..100 and the gripper to 0..100, so the threshold is a magnitude bound in normalized units
despite the variable name. 180 is a loose bound that only catches gross divergence; 100 is a tight
bound that catches anything out of the calibrated range. The evaluation used 180.

### 3.3 Chunk boundary blend

With overlap enabled, each new chunk starts from an observation captured a few steps earlier, so its
first action can differ from the last action of the outgoing chunk. Commanded as-is this is a step
discontinuity at every chunk boundary. Blending the first `overlap_steps` actions of the new chunk
into the last action of the old one with a linear ramp removes the jump.

```diff
--- a/vlash/run.py
+++ b/vlash/run.py
@@ class VLASHAsyncManager.get_action
         elif self.should_switch_chunk():
-            self.current_chunk = self.next_chunk.cpu().numpy() if self.next_chunk is not None else None
+            new_chunk = self.next_chunk.cpu().numpy() if self.next_chunk is not None else None
+            if (
+                new_chunk is not None
+                and self.current_chunk is not None
+                and self.overlap_steps > 0
+            ):
+                last_action = self.current_chunk[self.n_action_steps - 1]
+                blend_n = min(self.overlap_steps, len(new_chunk))
+                for i in range(blend_n):
+                    w_new = (i + 1) / (blend_n + 1)
+                    new_chunk[i] = last_action * (1 - w_new) + new_chunk[i] * w_new
+            self.current_chunk = new_chunk
             self.next_chunk = None
```

This is a real-time-chunking style inpainting on the boundary only. It has no effect when
`inference_overlap_steps` is 0.

### 3.4 Local tokenizer path (offline installs only)

```diff
--- a/vlash/policies/pi05/modeling_pi05.py
+++ b/vlash/policies/pi05/modeling_pi05.py
-        self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
+        _paligemma_path = os.environ.get(
+            "VLASH_PALIGEMMA_PATH",
+            "google/paligemma-3b-pt-224",
+        )
+        self.language_tokenizer = AutoTokenizer.from_pretrained(_paligemma_path)
```

Keeping the Hub id as the default means the patch is a no-op unless the variable is set. The build
used for the evaluation defaulted to a container path instead, which is the source of the
`HFValidationError` symptom listed in 2.4.

---

## 4. Checkpoint layout

### 4.1 What must sit in one directory

`policy.path` must point at the directory that **directly contains** these files:

```
<CKPT_ROOT>/40000/
    config.json            2,292 bytes    policy architecture and feature shapes
    model.safetensors      7,481,485,688 bytes    weights + normalization buffers
    train_config.json      6,830 bytes    training run metadata (not required to load)
```

For this full fine-tune, the normalization statistics are **inside** `model.safetensors` as buffers,
not in a separate stats file:

```
normalize_inputs.buffer_observation_state.mean   [6]
normalize_inputs.buffer_observation_state.std    [6]
normalize_targets.buffer_action.mean             [6]
normalize_targets.buffer_action.std              [6]
unnormalize_outputs.buffer_action.mean           [6]
unnormalize_outputs.buffer_action.std            [6]
```

Verify before the first run that they are present and finite. An uninitialized buffer is `inf` and
turns every action into NaN:

```python
from safetensors import safe_open

path = "<CKPT_ROOT>/40000/model.safetensors"
keys = [
    "normalize_inputs.buffer_observation_state.mean",
    "normalize_inputs.buffer_observation_state.std",
    "unnormalize_outputs.buffer_action.mean",
    "unnormalize_outputs.buffer_action.std",
]
with safe_open(path, framework="pt") as f:
    for k in keys:
        t = f.get_tensor(k)
        print(k, tuple(t.shape), bool(t.isfinite().all()))
```

All four lines must end in `True`.

LoRA-style checkpoints are different: the adapter weights live in a `lora_adapters/` subdirectory
and the statistics in a separate `normalize_buffers.pt`, both of which must sit in the same
directory as `config.json`. That case does not apply to this checkpoint but the path discipline
below does.

### 4.2 Warning: a directory level out of place breaks the run silently

**The loader does not search. It loads exactly what `policy.path` points at.**

This has already cost a full debugging cycle. A checkpoint whose real layout was
`{step}/pretrained_model/lora_adapters/` was copied by a sync tool that dropped the
`pretrained_model` level. `policy.path` still pointed one level too high. Nothing raised: the loader
found no adapter directory, fell through to loading the stock pi0.5 base, and ran. On the arm the
result was a policy that had learned nothing about the task and moved erratically. The training
code, the data, the normalization statistics and the cameras were all audited before the cause
turned out to be one directory level.

Before every run, list the directory instead of trusting the path:

```bash
CKPT=<CKPT_ROOT>/40000
ls -l "$CKPT/config.json" "$CKPT/model.safetensors"
```

If either line is missing, fix `CKPT` rather than launching. Different distribution paths produce
different depths: an extracted `model.tar.gz` typically puts the files under
`{step}/pretrained_model/`, while a directory sync puts them directly under `{step}/`.

Failure signatures:

| Symptom | Cause |
|---|---|
| `config.json not found`, or a draccus `ParsingError ... got {}` | `policy.path` is not the directory holding `config.json` |
| Runs, but the motion has nothing to do with the trained behavior; log shows the adapter directory was not found | one directory level off on an adapter-style checkpoint; only the base loaded |
| Actions are NaN or inf | normalization buffers missing or uninitialized |

---

## 5. Robot and camera configuration

The run is driven by one YAML file. It ships as `inference/eval.yaml`; the listing below is that
file, annotated. Every command in this document is written from the package root, so the two paths
that appear in them — `inference/v4l2_launch.py` and `inference/eval.yaml` — resolve as written.
The `<...>` placeholders have to be filled in before the first run.

```yaml
robot:
  type: so101_follower
  port: <serial_port>                   # follower arm, e.g. /dev/ttyACM0; `ls /dev/ttyACM*`
  id: <robot_id>                        # selects <calibration_dir>/<robot_id>.json
  calibration_dir: <path/to/calibration>
  cameras:
    base_0_rgb:                         # overhead / external view
      type: opencv
      index_or_path: <overhead_node>    # e.g. /dev/video0; identify by capturing a frame (5.4)
      fourcc: MJPG
      width: 640
      height: 480
      fps: 30
    left_wrist_0_rgb:                   # wrist view
      type: opencv
      index_or_path: <wrist_node>       # e.g. /dev/video2; swapping the two raises nothing
      fourcc: MJPG
      width: 640
      height: 480
      fps: 30

policy:
  path: <CKPT_ROOT>/40000               # directory holding config.json + model.safetensors
  empty_cameras: 1                      # fills the missing right_wrist_0_rgb slot
  device: cuda
  compile_model: true                   # required by inference_overlap_steps > 0
  n_action_steps: 20                    # re-plan every 20 actions; checkpoint stores 50
  num_inference_steps: 15               # checkpoint stores 10

single_task: "Pick up the green cube block and put it inside the white cup."

fps: 30
control_time_s: 600
display_data: false
play_sounds: false
action_quant_ratio: 1
inference_overlap_steps: 8
```

The four values that differ from the checkpoint's own `config.json` — `compile_model`,
`n_action_steps`, `num_inference_steps` and `inference_overlap_steps` — are the operating point,
and they are collected in section 7 with the reasoning for each.

Every key under `policy:` other than `path` is turned into a config override and applied on top of
the checkpoint's own `config.json`. Anything not listed keeps the value stored in the checkpoint.

### 5.1 Camera keys must be the policy slot names

The run loop compares `robot.cameras` keys against the policy's image features by name. pi0.5 has
exactly three fixed slots:

```
base_0_rgb            external / overhead view
left_wrist_0_rgb      wrist view
right_wrist_0_rgb     second wrist view
```

A key called `top` or `wrist` does not bind to a slot; with patch 3.1 in place it is rejected as an
unexpected camera, and without that patch the exact-match check rejects the whole configuration.
Name the camera entries after the slots and assign the physical views the way the training data was
mapped: external view to `base_0_rgb`, a wrist view to `left_wrist_0_rgb`.

### 5.2 Filling unused slots with `empty_cameras`

`empty_cameras` is the number of slots the policy should fill itself. Set it to the number of policy
slots you are not providing:

| Physical cameras | Slots provided | `empty_cameras` |
|---|---|---|
| 3 | base + left wrist + right wrist | 0 |
| 2 | base + left wrist | 1 |
| 1 | base | 2 |

For each missing slot the policy appends a black image (all -1 after the SigLIP range shift) with
its attention mask set to `False`, so the placeholder contributes no attention keys and the language
and action tokens never attend to it. This is exactly the arrangement used at training time: SO-101
data has no second wrist camera, so that slot was masked out for every SO-101 episode.

Two consequences worth knowing:

- Leave the **same** slots empty that training left empty. Fill in order (base, then a wrist) and
  leave the trailing slots to `empty_cameras`.
- Because position ids are computed as the cumulative sum of the padding mask, masked placeholders
  consume no position index; they do not shift the valid images.

The checkpoint's `config.json` stores `empty_cameras: 0`, so a two-camera setup must set it
explicitly in the YAML. A verification run with two cameras produced three images with masks
`[True, True, False]` and a finite `(1, 6)` action, which is the expected result.

### 5.3 Calibration

LeRobot resolves the calibration file as `<calibration_dir>/<id>.json` and decodes it into
`dict[str, MotorCalibration]`. The schema is one entry per joint:

```json
{
  "shoulder_pan":   {"id": 1, "drive_mode": 0, "homing_offset": -1594, "range_min": 1045, "range_max": 3418},
  "shoulder_lift":  {"id": 2, "drive_mode": 0, "homing_offset": -1657, "range_min":  845, "range_max": 3214},
  "elbow_flex":     {"id": 3, "drive_mode": 0, "homing_offset":  1566, "range_min":  904, "range_max": 3121},
  "wrist_flex":     {"id": 4, "drive_mode": 0, "homing_offset": -1462, "range_min":  875, "range_max": 3220},
  "wrist_roll":     {"id": 5, "drive_mode": 0, "homing_offset": -1488, "range_min":    0, "range_max": 4095},
  "gripper":        {"id": 6, "drive_mode": 0, "homing_offset":  1548, "range_min": 2027, "range_max": 3496}
}
```

Values are that particular arm's; use your own.

Two practical points:

- **Extra top-level keys break the parse.** The decoder maps every top-level key to a
  `MotorCalibration`, so a metadata block added by other tooling (creation time, hardware id, and
  so on) raises a decoding error naming that key. Strip anything that is not a joint before use.
- **First connect prompts.** If the values stored in the motors differ from the file, LeRobot prints
  `Press ENTER to use provided calibration file associated with the id ..., or type 'c' and press
  ENTER to run calibration:`. Pressing ENTER writes the file's calibration into the motors, which is
  what you want. Typing `c` starts a full re-calibration and overwrites the file.

If the arm's calibration does not match the arm the data was recorded on, joint zero points shift
and the policy's actions are offset by a constant. That is not something normalization can absorb.

### 5.4 Camera backend and format

Three issues cost real time here.

**OpenCV backend.** On Linux, LeRobot's `get_cv2_backend()` returns `cv2.CAP_ANY`. With some UVC
cameras that backend reports `set(fourcc)` and `set(width)` as failed and then dies during capture
with `VIDIOC_QBUF Bad file descriptor`. `cv2.CAP_V4L2` works. Rather than editing site-packages,
replace the function at import time in a launcher. That launcher ships as
`inference/v4l2_launch.py`:

```python
#!/usr/bin/env python
"""Launcher that forces LeRobot's OpenCV camera backend to V4L2.

LeRobot's get_cv2_backend() returns CAP_ANY on Linux, which fails on some UVC
cameras with set(fourcc/width)=False followed by VIDIOC_QBUF: Bad file descriptor.
CAP_V4L2 works. Replacing the function at import time avoids modifying
site-packages and is reversible.

Usage: python v4l2_launch.py run <config.yaml> [--overrides...]
"""
import cv2
import lerobot.cameras.utils as _utils
import lerobot.cameras.opencv.camera_opencv as _cam


def _v4l2_backend():
    return int(cv2.CAP_V4L2)


_utils.get_cv2_backend = _v4l2_backend
_cam.get_cv2_backend = _v4l2_backend

from vlash.cli import main   # noqa: E402  (import after the patch is applied)

if __name__ == "__main__":
    main()
```

Both module references have to be replaced; the camera module imports the symbol directly.

**Pixel format.** Set `fourcc: MJPG` explicitly. Several USB cameras default to a raw format that
cannot sustain 640x480 at 30 fps over USB bandwidth.

**Which node is which camera.** Do not assume `/dev/video0` is the overhead camera. Capture a frame
from each node and look at it. Swapping the overhead and wrist views does not raise anything; the
policy just degrades. For stability across reboots and replugs, add udev rules that create stable
symlinks (`/dev/cam_top`, `/dev/cam_wrist`) keyed on vendor, product and serial, and use those in
`index_or_path`.

---

## 6. Runtime options

### 6.1 How the loop works

At `fps: 30` the control loop runs one step every 33.3 ms. The policy predicts a chunk of
`chunk_size` (50) actions per inference; the loop executes `n_action_steps` of them and then
re-plans. Observations are captured only when needed - at startup and at the step where the next
inference is launched - not every step.

With `inference_overlap_steps: N`, the next inference is issued `N` steps before the current chunk
runs out. The predicted chunk stays on the GPU until the boundary, so GPU work overlaps with the
remaining `N` steps of execution and the transfer and synchronization happen at the switch. The
inference call is issued from the control loop, so its CPU-side launch cost lands on that one step;
that is why VLASH refuses `inference_overlap_steps > 0` unless `compile_model` is true.

The chunk that is launched early is conditioned on the *last action of the current chunk* rather
than the live joint state - the state the arm is predicted to be in when the new chunk starts. This
is VLASH's future-state-aware trick and it is what makes overlap stable instead of jittery.

### 6.2 Option reference

| Option | Where | Meaning | Tradeoff |
|---|---|---|---|
| `policy.n_action_steps` | policy | actions executed per chunk before re-planning; must be <= `chunk_size` (50) | this is the re-planning period. At 30 fps, 20 means a fresh plan every 667 ms. Lower = more reactive to changes in the scene, more compute; higher = smoother but sluggish to react |
| `inference_overlap_steps` | run | steps before the chunk ends at which the next inference starts; 0 = synchronous | larger = more compute window and a longer blend ramp, but the new chunk is conditioned on an older observation. A quarter to a half of `n_action_steps` is a reasonable band. Must be <= `n_action_steps`, and requires `compile_model: true`. Scaled internally by `action_quant_ratio` |
| `policy.compile_model` | policy | `torch.compile` the policy | adds 2-3 min of warmup at startup, then faster inference. Mandatory when overlap > 0 |
| `policy.num_inference_steps` | policy | flow matching denoising steps per chunk | more = more precise integration, slower; fewer = faster, coarser. The checkpoint stores 10; the evaluation used 15 |
| `fps` | run | control loop target rate | should match the rate the training data was collected at (30). A different loop rate changes what one action step means in time and stretches or compresses every trajectory |
| `policy.device` | policy | `cuda` | - |
| `action_quant_ratio` | run | send every Nth action | 2 makes the arm traverse a chunk twice as fast at the cost of resolution. The evaluation used 1 |
| `control_time_s` | run | wall-clock runtime, then the loop exits | 15-30 s for a dry run, 60-600 s for a live task |
| `display_data` | run | stream observations to Rerun | useful for debugging, costs loop time |
| `play_sounds` | run | spoken start/stop cues | off for evaluation |
| `VLASH_SAFE_DEG` | env (patch 3.2) | abort if any action exceeds this magnitude | 180 is loose, 100 is tight in normalized joint units |
| `VLASH_DRY_RUN` | env (patch 3.2) | compute actions, do not command motors | see section 8 |

`n_action_steps` and `num_inference_steps` are stored in the checkpoint's `config.json` as 50 and 10.
To reproduce the evaluation they must be overridden explicitly, in YAML or on the command line.

### 6.3 Making it more reactive

If the arm looks like it is committed to a stale plan:

1. Lower `n_action_steps` (the stored 50 -> 20 -> 12). This is the dominant knob; it is the
   re-planning period.
2. Enable overlap at roughly half of `n_action_steps`, with `compile_model: true`. The boundary
   blend of patch 3.3 absorbs the jitter that frequent re-planning would otherwise introduce.
3. If inference still does not fit the window, lower `num_inference_steps`.

Do not go to `n_action_steps: 1`. Re-planning every step only works if a full inference finishes
inside 33 ms; when it does not, the loop stutters and the motion becomes jerkier than with a longer
chunk.

---

## 7. Validated operating configuration

These are the settings under which the hardware result in section 9 was obtained. They are one
working point, not a tuned optimum.

| Setting | Value |
|---|---|
| Checkpoint | step 40000 |
| `inference_overlap_steps` | 8 |
| `policy.compile_model` | true |
| `policy.n_action_steps` | 20 |
| `policy.num_inference_steps` | 15 |
| `fps` | 30 |
| `action_quant_ratio` | 1 |
| `VLASH_SAFE_DEG` | 180 |
| Cameras | 2 physical, `empty_cameras: 1` |

At 30 fps that is a re-plan every 667 ms with a 267 ms overlap window and a blend ramp over the
first 8 actions of each chunk. The reading is a balance against what the checkpoint stores:
`num_inference_steps: 15` above the stored 10 for precision, `n_action_steps: 20` well below the
stored 50 for reactivity, overlap 8 for smoothness.

The stored `n_action_steps` is 50, equal to `chunk_size` — the checkpoint's default is to execute
an entire chunk before re-planning, which at 30 fps is a plan every 1.67 s. That is the value
being overridden here, and it is the single largest difference between the stored configuration
and the one that was validated.

As a command line on top of the YAML in section 5:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLASH_SAFE_DEG=180
export VLASH_DRY_RUN=0

python inference/v4l2_launch.py run inference/eval.yaml \
  --policy.path=<CKPT_ROOT>/40000 \
  --single_task="Pick up the green cube block and put it inside the white cup." \
  --inference_overlap_steps=8 \
  --policy.compile_model=true \
  --policy.n_action_steps=20 \
  --policy.num_inference_steps=15 \
  --fps=30 \
  --control_time_s=600
```

Without the V4L2 launcher the same command is `vlash run inference/eval.yaml ...`.

---

## 8. Evaluation procedure

### 8.1 Dry run first, always

`VLASH_DRY_RUN=1` runs the whole pipeline - cameras, policy, action decoding - and prints the
action dict every 10 steps without calling `send_action`. Use it to confirm, before the arm can move:

- the checkpoint loaded from the directory you intended,
- the cameras opened and are the right way round,
- actions are finite and inside the calibrated range,
- the loop holds its rate.

Turn compile and overlap off for this, so startup is 25 s instead of 2-3 minutes; neither has
anything to do with what the dry run checks.

```bash
export VLASH_DRY_RUN=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python inference/v4l2_launch.py run inference/eval.yaml \
  --policy.path=<CKPT_ROOT>/40000 \
  --policy.compile_model=false \
  --inference_overlap_steps=0 \
  --control_time_s=30
```

Expected output shape:

```
[step 0] action: {'shoulder_pan.pos': -1.4, 'shoulder_lift.pos': -96.2, ...}
[step 10] action: {...}
```

What to look for: six keys ending in `.pos`, values that move smoothly between prints, nothing
saturating at the abort threshold, no `nan`. `nan` means the normalization buffers did not load
(section 4.1). Values that make no sense for the scene, with everything else healthy, usually mean
the wrong directory level (section 4.2).

The arm is still connected and torque is still enabled in dry run; `configure()` re-enables torque
on connect. The arm holds position rather than going limp, so treat it as live hardware even in dry
mode.

### 8.2 Live run

Start synchronous and short, still overriding compile and overlap off:

```bash
export VLASH_DRY_RUN=0
python inference/v4l2_launch.py run inference/eval.yaml \
  --policy.path=<CKPT_ROOT>/40000 \
  --policy.compile_model=false \
  --inference_overlap_steps=0 \
  --control_time_s=30
```

Then drop the two overrides and let the YAML stand as written, which is the configuration in
section 7. Startup is about 25 s without compile and about 2-3
minutes with it, most of it `torch.compile` warmup - the policy is warmed with three dummy inference
passes before the arm is touched.

### 8.3 Prompt selection

`single_task` is the language instruction that goes into the VLM. Use the exact phrasing from the
training data when measuring task success; use different phrasing when probing generalization. A
prompt that does not match anything in the checkpoint's training distribution produces plausible but
wrong behavior rather than an error, so keep the two cases separate when interpreting results.

The training task string for a checkpoint can be read from `train_config.json` in the checkpoint
directory.

### 8.4 Checkpoint sweep

Lowest training loss does not mean best closed-loop behavior. Checkpoints were saved every 2000
steps, so the run produced 20 of them and the mid-to-late range is the interesting one.

**Only step 40000 is published.** The other nineteen are not part of the release, so this sweep
is available only if you ran the training yourself and still have your own archive. Nothing below
can be done against the released weights alone.

Hold everything constant except `policy.path`:

```bash
for STEP in 12000 16000 20000 24000 28000 32000 36000 40000; do
  echo "=== step $STEP ==="
  python inference/v4l2_launch.py run inference/eval.yaml \
    --policy.path=<CKPT_ROOT>/$STEP \
    --single_task="$PROMPT" \
    --inference_overlap_steps=8 \
    --policy.compile_model=true \
    --policy.n_action_steps=20 \
    --policy.num_inference_steps=15 \
    --control_time_s=120
done
```

Reset the scene to the same initial layout between checkpoints, run a fixed number of trials each,
and record successes rather than impressions. Note that `compile_model: true` pays the warmup cost
once per checkpoint, so a sweep of eight checkpoints spends roughly 20 minutes on compilation alone;
running the sweep synchronously (`--inference_overlap_steps=0 --policy.compile_model=false`) trades
motion quality for turnaround.

**This sweep has not been run.** Only step 40000 has been evaluated on hardware.

---

## 9. Observed behavior and limits

Closed-loop evaluation on the physical arm, single session.

| Item | Value |
|---|---|
| Checkpoint | step 40000 |
| Prompt | "Pick up the green cube block and put it inside the white cup." |
| Demonstrations of this task in the training data | 14 |
| Observed success rate | about 90% |
| Number of trials behind that rate | not recorded |
| Settings | section 7 |

The task was run repeatedly, with the cube moved between trials and under varying lighting, and
succeeded across those variations. **The number of trials was not recorded**, so 90% is an observed
rate over an unrecorded denominator: no confidence interval, a qualitative report rather than a
measurement.

The remaining roughly 10% is not spread evenly across conditions. It concentrates on one:
instructions naming several cube colors and asking for them to be placed in the cup in a specified
order. That instruction form does not appear in this task's 14 demonstrations, which are all
single-color pick-and-place under one identical prompt, so multi-color sequential ordering is out of
distribution for this checkpoint on this task. Section 9.1 reaches the same condition from the
generalization side.

The task appears in the training set as 14 episodes out of the 16,687 the run was fed. The result is
mainly a statement about how little task-specific data an adapted base needs, not about the task.
No runaway motion was observed on any joint.

### 9.1 Generalization probe

Substituting an instruction absent from the training data - "black cube" in place of "green cube":

- **The manipulation skill transferred.** The arm performed pick-and-place under the novel phrasing.
- **Color grounding transferred.** The named color was handled first, so the language-to-object
  binding was not simply ignored.
- **Scoping and termination did not transfer.** After finishing the named subset the policy kept
  going with the other cubes instead of stopping. With the trained phrasing ("green cube") it
  stopped correctly.

Caveat on the first two points: black had high visual salience against the mat, so salience rather
than language may explain the ordering. A follow-up with multi-color sequential instructions ("blue,
then green, then black, each into the cup") was run to settle this and was **inconclusive** - the
overhead camera was positioned such that color identification was occluded, so perception, not
grounding, dominated the result. This is not evidence that grounding failed; it is an experiment
that has to be repeated with an unoccluded overhead view.

### 9.2 Limits

- Only step 40000 has been validated on hardware. No checkpoint sweep has been run, and the lowest
  training loss does not imply the best closed-loop behavior.
- One task, one physical setup, one session. This is not a benchmark and the success rate should not
  be read as a general capability number. The trial count behind it was not recorded.
- Task scoping and termination is the weakest observed behavior. Anything that depends on "do
  exactly the named subset, then stop" should be fine-tuned rather than prompted.
- Trained on single-arm 6-DoF data only. Bimanual setups use a different action space.
- Compared with a task-specialized fine-tune on the same task, the base's trajectories are longer
  and execution is slower; it is a general policy, not a specialist.
- Inference latency per chunk, loop timing statistics and peak VRAM were not recorded.

---

## 10. Safety

The arm holds a 6-DoF servo chain with enough torque to damage itself, the workspace and hands. The
policy has no notion of obstacles beyond what it sees.

Before a live run:

- Clear the workspace of anything outside the intended scene, including cables in the swept volume.
- Keep the power supply switch within reach. Cutting power is the fastest stop.
- Have both stop paths ready: `ESC` or the right arrow key if there is a display attached (the
  keyboard listener is disabled in headless environments and prints a warning saying so), `Ctrl-C`
  otherwise. Both fall into the `finally` block that disconnects the robot and releases the cameras.
- Set `control_time_s` to the shortest value that covers the trial. The loop exits by itself, which
  is a backstop when a stop key is missed.

During bring-up:

- Always dry run a new checkpoint or a new prompt before commanding motors.
- Keep `VLASH_SAFE_DEG` set. It aborts the loop on the first action whose magnitude leaves the
  plausible range. With normalized units, 100 is a tight bound and 180 only catches gross
  divergence.
- Consider LeRobot's own `max_relative_target` on the robot config; it clips how far a commanded
  position may sit from the present position, which bounds the size of a single jump. It costs a
  `Present_Position` read every step, so the loop rate drops.
- Torque is enabled on connect and stays enabled through a dry run. The arm holds position; it does
  not go limp. `disable_torque_on_disconnect` defaults to true, so the arm does relax on a clean
  disconnect - support it before stopping if it is holding a load in a raised pose.
- Watch the first chunk after a checkpoint switch. If the loaded checkpoint is not the one you meant
  (section 4.2), the failure appears as motion, not as an exception.

Hardware faults seen on this rig, in case they surface as apparent policy failures:

- All six motors failing `sync_read` at once, right after motion begins, is usually a power problem:
  torque draws current, voltage sags, the bus drops. Check the supply before the code.
- A single motor reporting no status packet at connect is a cable or daisy-chain fault on that
  motor.
- Servo overload: power off for 10 s, power on.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `HFValidationError` on tokenizer load | tokenizer path unset and the default path does not exist | set `VLASH_PALIGEMMA_PATH` to a local tokenizer directory, or authenticate to the gated Hub repo |
| Offline-mode error on load | `HF_HUB_OFFLINE=1` with an incomplete cache | populate the cache online once, then go offline |
| `Robot camera names must exactly match policy image feature names!` | fewer cameras than policy slots on unpatched VLASH | apply patch 3.1 and set `empty_cameras` |
| `Robot has cameras the policy does not expect` | camera key is not a policy slot name | rename the key to `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb` |
| `When inference_overlap_steps > 0, policy.compile_model must be True` | async without compile | set `--policy.compile_model=true` |
| `n_action_steps must be greater than or equal to overlap_steps` | overlap too large | lower `inference_overlap_steps` below `n_action_steps` |
| `n_action_steps (N) cannot be greater than chunk_size (50)` | `n_action_steps` over 50 | lower it |
| `config.json not found`, draccus `ParsingError ... got {}` | `policy.path` one level off | `ls` the directory, point at the level holding `config.json` |
| Actions are NaN or inf | normalization buffers missing or uninitialized | verify with the snippet in 4.1 |
| Runs but the behavior is unrelated to training | wrong checkpoint level loaded, or `single_task` does not match the trained instruction | check the directory level (4.2), check `train_config.json` for the training task string |
| `set(fourcc/width)` returns false, then `VIDIOC_QBUF Bad file descriptor` | OpenCV resolved to `CAP_ANY` | force `cv2.CAP_V4L2` (5.4) |
| Camera opens but frames stall at 30 fps | raw pixel format over USB | set `fourcc: MJPG` |
| `Not a video capture device` | wrong `/dev/videoN` | enumerate capture devices, verify by capturing a frame |
| Decoding error naming an unknown key while loading calibration | non-joint metadata in the calibration JSON | strip every top-level key that is not a joint |
| `Mismatch between calibration values in the motor and the calibration file` prompt | motors and file disagree, or `id` points at the wrong file | press ENTER to write the file into the motors; check `robot.id` |
| Headless warning about keyboard input | no display attached | stop with `Ctrl-C`; `ESC` is unavailable |
| Process left running after a crash | - | `pkill -f "vlash run"`; the `finally` block disconnects the robot and releases the cameras |
