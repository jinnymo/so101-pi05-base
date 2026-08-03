---
license: gemma
license_link: https://ai.google.dev/gemma/terms
base_model: lerobot/pi05_base
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - vision-language-action
  - vla
  - lerobot
  - pi0.5
  - pi05
  - so-101
  - so-arm101
  - flow-matching
  - full-fine-tune
---

# so101-pi05-base

A domain-adapted base checkpoint: Physical Intelligence's **pi0.5** vision-language-action
model, fully fine-tuned on SO-101 / SO-100 single-arm manipulation data.

- Base model: [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base)
- Training / inference stack: [VLASH](https://github.com/mit-han-lab/vlash) (`PI05Policy`)
- Robot: SO-101 / SO-100, single arm, 6 DoF
- Checkpoint: step 40000, `model.safetensors` 7,481,485,688 bytes (about 7.0 GiB)
- Training data: [`dongyoonkim/so101-pi05-base-dataset`](https://huggingface.co/datasets/dongyoonkim/so101-pi05-base-dataset)
  (the licensed subset of it)

## Intended use

This checkpoint is a **domain adaptation base, not a finished policy**. It is meant to be the
starting point of a task-specific fine-tune - LoRA or full - on demonstrations of the task you
actually want to run.

What the adaptation provides is a prior over SO-101 kinematics, action scale and the camera
viewpoints these robots are usually recorded from, so that a task fine-tune converges from fewer
demonstrations than one started from `lerobot/pi05_base`. What it does not provide is dependable
behavior on any particular task, because no particular task was the training objective.

Running an untuned VLA checkpoint on a real arm is unsafe. That holds for the backbones this work
started from and it holds for this one: without a task fine-tune the output is erratic, and an
erratic action stream on a physical arm damages the arm or whatever it reaches. Evaluate in dry
run first, keep the joint-angle abort in place, and stay within reach of the stop.

## Model description

pi0.5 is roughly 3.62B parameters in two parts:

| Component | Role |
|---|---|
| PaliGemma VLM (`gemma_2b` backbone + SigLIP vision encoder) | encodes camera images and the language instruction |
| Action expert (`gemma_300m`) | generates the action chunk by flow matching, integrated over `num_inference_steps` denoising steps |

Interface:

| Field | Value |
|---|---|
| Action dimension | 6 (SO-101 joints) |
| Chunk size | 50 |
| State conditioning | enabled (state enters through adaRMS conditioning, not the prompt) |
| Image resolution | 224 x 224, resize with aspect-preserving pad |
| Normalization | state / action `MEAN_STD`, visual `IDENTITY` |

### Camera slots and masking

pi0.5 uses exactly three fixed image slots, following the openpi convention:

```
observation.images.base_0_rgb          external / overhead view
observation.images.left_wrist_0_rgb    wrist view
observation.images.right_wrist_0_rgb   second wrist position
```

A setup with fewer than three cameras fills the unused slots with a black dummy image and
sets that slot's attention mask to `False`, so the placeholder contributes no attention keys.
The model therefore runs on 1, 2 or 3 cameras.

Training data was mapped into the same three slots by keyword: cameras whose key looks
wrist-mounted go to the wrist slots, everything else goes to `base_0_rgb`, and a slot that is
still empty after that is filled from whatever cameras are left over before being masked. That
last rule is the reason the slot names should not be read literally. Every source is single-arm,
so no source has a genuine second wrist camera, but `right_wrist_0_rgb` is far from always a
placeholder: in the released subset 14 sources (1,151 episodes) carry a real camera stream in
that slot, and 42 sources (4,704 episodes) have a non-wrist camera sitting in one of the two
wrist slots.

**Slot names denote positions in the pi0-family layout, not a guaranteed viewpoint semantics.**
Assign your external view to `base_0_rgb` and your wrist view to `left_wrist_0_rgb`, which is by
far the most common assignment in training, but do not expect the model to hold a strict "this
slot is always a wrist view" invariant.

Camera count is not a weight anywhere in sampling, loss or normalization, so single-camera and
three-camera episodes contribute equally.

## Training recipe

| Item | Value |
|---|---|
| Method | full fine-tune, no LoRA, all 3.62B parameters trainable |
| Precision | bfloat16 (the action expert's adaRMS conditioning dense layers stay fp32: 97 of 824 tensors, 3.4% of parameters) |
| Effective batch | 256 (per-GPU 8 x 8 GPUs x grad accumulation 4) |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 1e-10, grad clip norm 1.0 |
| Learning rate | peak 5e-5, cosine decay to 2.5e-6 |
| Warmup | 1000 steps |
| Steps | 40000 (about 1.19 epochs over the training set) |
| Normalization | MEAN_STD for state and action |
| Dataloader workers | 8 |
| RTC delay simulation | `max_delay_steps` 8 |
| Hardware | 8 x A100 80GB, about 40 hours wall clock, clean exit |

Training loss went from 0.10 to 0.0065 and flattened near the end (standard deviation of the
last 300 steps: 0.0006). Gradient norm settled around 0.058 after warmup. No crashes and no
NaN losses.

The `config.json` shipped with the checkpoint carries the policy-level optimizer defaults
(`optimizer_lr`, `optimizer_weight_decay`, `scheduler_decay_steps`), which are not the values used
for this run; the same defaults appear again inside the `policy` block of `train_config.json`. The
run used the top-level `optimizer` and `scheduler` blocks of `train_config.json`, which is what
`use_policy_training_preset: false` selects. The table above is authoritative.

## Training data

The training set is a single unified LeRobot v2.1 dataset built from public SO-101 / SO-100
datasets on the Hugging Face Hub plus a small set of recordings made by the author.

Collection and filtering:

```
11,270 repositories crawled on the Hub
  ->  1,724 with SO-101 / SO-100 metadata
  ->    408 after screening (>= 50 episodes, camera blacklist, simulation split)
  ->    307 passing action-convention checks (dim 6, joint names, degrees, follower frame)
  ->    307 downloaded (305 automatically, 2 recovered with git-lfs)
  ->    296 after precision analysis and quality exclusions (11 removed)
  ->    205 after camera classification and manual duplicate removal (91 removed)
  ->    211 with 6 datasets recorded by the author added
  ->    181 kept after action-fingerprint deduplication (30 dropped as re-uploads,
             supersets or train/val overlaps)
```

Action convention was checked before anything else: datasets whose `action` was recorded in
the leader arm's calibration frame, in delta or raw units, or for a bimanual arm, were
rejected rather than normalized, since normalization statistics absorb scale but not a shifted
zero point.

Unified dataset after merging:

| Item | Value |
|---|---|
| Episodes | 17,137 |
| Frames | 8,690,531 |
| Tasks | 430 |
| Videos | 51,411 |
| Format | LeRobot v2.1, three camera slots plus a per-slot mask column |

**Episodes actually used for this run: 16,687.** 450 episodes from 10 fps sources were
excluded after a timestamp inconsistency was found mid-run (see Issues below).

### Released data subset

The training data is released only in part. Only datasets whose source repository declares a
license are included; repositories with no declared license were withheld, verified through
the Hub API.

| License | Origin | Datasets | Episodes |
|---|---|---|---|
| apache-2.0 | Hugging Face Hub | 149 | 13,587 |
| mit | Hugging Face Hub | 1 | 50 |
| apache-2.0 | recorded by the author | 6 | 332 |
| **Released total** | | **156** | **13,969** |
| withheld, no declared upstream license | Hugging Face Hub | 25 | 3,168 |

"Recorded by the author" is a provenance category, not a license: those six datasets are
released under the Apache License 2.0 like the rest.

**This model cannot be exactly reproduced from the released data.** The withheld portion is
3,168 episodes, 18.5% of the unified dataset, and it is spread across tasks and camera
configurations rather than concentrated in one category. A run on the released subset alone
trains on a different, smaller distribution. The per-dataset source repository list and its
licenses ship with
[`dongyoonkim/so101-pi05-base-dataset`](https://huggingface.co/datasets/dongyoonkim/so101-pi05-base-dataset)
as `ATTRIBUTION.md`.

## Evaluation

Closed-loop evaluation on physical hardware, 2026-06-26. This is a qualitative observation on one
task and one setup, not a benchmark.

| Item | Value |
|---|---|
| Robot | SO-101 follower arm |
| Cameras | 2 physical (overhead to `base_0_rgb`, wrist to `left_wrist_0_rgb`), third slot masked via `empty_cameras=1` |
| Checkpoint | step 40000 |
| Prompt | "Pick up the green cube block and put it inside the white cup." |
| Demonstrations for this task in training data | 14 |
| Observed success rate | about 90% |
| Number of trials | not recorded |
| Run settings | `inference_overlap_steps` 8, `compile_model` true, `n_action_steps` 20, `num_inference_steps` 15, `fps` 30 |

The task was run repeatedly, with the cube moved between trials and under varying lighting, and
succeeded across those variations. **The exact number of trials was not recorded**, so 90% is an
observed rate over an unrecorded denominator, with no confidence interval.

The remaining roughly 10% is not spread evenly across conditions. It concentrates on one:
instructions naming several cube colors and asking for them to be placed in the cup in a specified
order. That instruction form is absent from the training data for this task, whose 14
demonstrations are all single-color pick-and-place under one identical prompt. Multi-color
sequential ordering is out of distribution for this checkpoint on this task, and it is where the
failures are.

The task appears in the training data with 14 demonstrations out of 8.69M frames, so the result is
mainly a statement about how little task-specific data an adapted base needs before it does
something reasonable, not about the task itself.

### Generalization observations

Substituting an instruction that does not appear in training data ("black cube" instead of
"green cube") produced:

- The manipulation skill transferred: the arm performed pick-and-place under the novel phrasing.
- Color grounding transferred: the named color was handled first, so the language-to-object
  binding was not simply ignored.
- Task scoping and termination did not transfer: after finishing the named subset the policy
  continued with other cubes instead of stopping. With the trained phrasing ("green cube") it
  stopped correctly.

Caveat on the first two points: black had high visual salience against the mat, so
salience rather than language may explain the ordering. A follow-up with multi-color
sequential instructions was **inconclusive**, because the overhead camera was positioned such
that color identification was occluded. Grounding strength is therefore an open question, and
since multi-color sequential instructions are also where the failures reported above
concentrate, how much of that failure is grounding and how much is camera placement is
unresolved.

### An observed recovery behavior

In one run the cube ended up beside the cup, in a position where the cup occluded it from the
overhead camera. The arm attempted the grasp three times and missed each time. It then backed
away, re-approached from a lower angle that cleared the occlusion, and completed the task.

Retry-and-re-approach is not in the 14 demonstrations of this task, which are single clean
approaches under one prompt. That fact on its own says nothing about where the behavior came
from.

**Where it plausibly comes from.** Recovery is a stated property of the pi0 pretraining mixture.
The pi0 paper describes its diverse pretraining data as "providing a variety of scenes,
corrections, and recovery behaviors that might not be present in more narrow specialized data",
and says the resulting model "still has a repertoire of recoveries and corrections that it can
deploy in the case of a mistake" ([arXiv:2410.24164](https://arxiv.org/abs/2410.24164)). pi0.5 is
trained on an extended version of that dataset
([arXiv:2504.16054](https://arxiv.org/abs/2504.16054)), so it inherits the property. The reading
consistent with this is that the behavior originates in pi0.5's pretraining and that a full
fine-tune on a broad multi-source SO-101 corpus **preserved** it rather than produced it - a full
fine-tune can overwrite pretrained behavior, and this one apparently did not.

**An independent measurement exists on this robot.** Recovery rate for pi0.5 fine-tuned on SO-101
has been measured by a third party: 30.77%, against 20.51% for Wall-X, 6.45% for ACT and 3.23% for
SmolVLA, over 20 evaluation episodes per model-task pair with 100 demonstrations per task
([arXiv:2606.08881](https://arxiv.org/abs/2606.08881)). ACT was trained on the same SO-101
demonstrations as pi0.5 there, so the gap between them is attributable to pretraining rather than
to the task data. One recovery after three failed grasps is not in tension with a reported rate in
that range. That benchmark is a different setup, a different task set and a different checkpoint,
so it is context for the observation, not a measurement of this checkpoint.

**A contrasting observation, not an ablation.** A LoRA fine-tune of the same pi0.5 base on 150
episodes of a single task did not show this behavior. That is one comparison under one set of
conditions - task, data volume and adaptation method all differ at once. Its direction matches
what RETAIN reports for policies fine-tuned on limited demonstrations of one task: they "often
overfit to the specific demonstrations - not only losing their prior abilities to solve a wide
variety of generalist tasks but also failing to generalize within the new task itself"
([arXiv:2512.08333](https://arxiv.org/abs/2512.08333)). That paper measures loss of general
capability, not retry behavior specifically, so it supports the direction of the observation and
not the mechanism.

**What is not established.**

- It has **not** been verified that no retry or recovery frames exist anywhere in the
  17,137-episode corpus. "Absent from the demonstrations" holds for this task's 14
  demonstrations, not for the corpus.
- The number of trials was not recorded, so this is one observation and not a rate: no recovery
  rate is claimed for this checkpoint.
- The behavior may be a product of closed-loop dynamics and flow matching's stochastic sampling
  rather than a retained skill. The same SO-101 benchmark classifies the underlying pattern as
  recovery when the retry succeeds and as a repetition loop when it does not, which is what a
  sampling-driven account would predict. Action representation is also known to change how
  strongly a policy retries: Octo reports that an absolute versus relative *gripper* action
  representation alters this ([arXiv:2405.12213](https://arxiv.org/abs/2405.12213)).
- A single observation is not a capability. Treating one run as evidence of a discrete new ability
  is the failure mode Schaeffer et al. document for claimed capability jumps
  ([arXiv:2304.15004](https://arxiv.org/abs/2304.15004)).
- No video of this observation is published.

Practical implication: a narrow single-task LoRA on top of this checkpoint can suppress this kind
of recovery, since its demonstrations show one clean approach and nothing else. If recovering from
a failed grasp matters for your task, put failed-and-recovered attempts in the fine-tuning data
rather than assuming the base behavior survives.

### Limitations

- This is a base for fine-tuning, not a deployable task policy. See Intended use.
- Only step 40000 has been validated on hardware. A checkpoint sweep across the
  mid-to-late checkpoints has not been run, and lowest training loss does not imply best
  closed-loop behavior. An earlier checkpoint may perform better.
- The evaluation covers one task on one physical setup, over an unrecorded number of trials.
  The success rate is not a general capability number.
- Task scoping and termination is the weakest observed behavior. Anything relying on "do
  exactly the named subset, then stop" should be fine-tuned rather than prompted.
- Trained on single-arm 6-DoF data only. Bimanual configurations use a different action space
  and are out of scope.
- The training pool includes a small number of simulation datasets, so some prior comes from
  rendered rather than real observations.
- Per-dataset absolute calibration zero points were not equalized across sources. Normalization
  statistics absorb scale, not a shifted zero; a downstream fine-tune on the target arm is the
  intended correction path.

## Verified environment

The release `README.md` carries the canonical tables, one for the training container and one for
the machine the evaluation ran on. In short: training on 8 x A100 80GB SXM4, evaluation on a
single RTX 3090 Ti 24 GB with driver 590.48.01, both on Python 3.10, torch 2.7.1+cu126,
torchvision 0.22.1, torchcodec 0.5, ffmpeg 7.1, lerobot 0.4.1, transformers 4.53.3 and VLASH at
commit `22cbabfee0f57874987c75a35a7dac129e695db0`. Other versions may work but have not been
verified.

## Quick start

Confirm the checkpoint downloaded intact before wiring up a robot. The check needs no GPU and no
VLASH — it reads the safetensors header and `config.json` with the standard library alone.

```bash
pip install huggingface_hub
hf download dongyoonkim/so101-pi05-base --local-dir ckpt/so101-pi05-base
python scripts/verify_checkpoint.py ckpt/so101-pi05-base
```

`hf` is the current Hugging Face Hub CLI. `huggingface-cli` is deprecated and on
huggingface_hub 1.x prints a notice and exits non-zero.

`scripts/verify_checkpoint.py` ships with this release and the release `README.md` shows its
full output. It should report 3,618,890,548 parameters — the whole 3.62B model — the three
camera slots, and `empty_cameras: 0`, which is the field a robot with fewer than three cameras
has to override. It exits non-zero on a truncated or mismatched checkpoint.

## Usage

**Upstream VLASH will not run this checkpoint with fewer than three cameras without a patch.**
Upstream's `validate_robot_cameras` requires the robot's camera keys to match the policy's image
features *exactly*, and it compares them against the three slots read from the checkpoint by
`PreTrainedConfig.from_pretrained`. That comparison happens before `empty_cameras` is applied, so
setting `empty_cameras` in your config does not satisfy it. On a clean clone the two-camera
configuration shown below dies at startup with

```
ValueError: Robot camera names must exactly match policy image feature names!
```

The patch relaxes the check to accept a subset of the policy's slots, while still rejecting camera
keys the policy does not know. The hardware evaluation above was run with it applied. The patched
file ships with this release as `training-docker/patched/run.py` and also carries the
chunk-boundary blend; copy it over `vlash/run.py` after installing. See "Patches to the training
and inference stack" below for the full list.

```bash
git clone https://github.com/mit-han-lab/vlash
cd vlash
git checkout 22cbabfee0f57874987c75a35a7dac129e695db0   # the revision the patched files were written against
pip install -e .

# required for fewer than three cameras
cp <path-to-this-release>/training-docker/patched/run.py vlash/run.py
```

The checkpoint itself comes down with the `hf download` in Quick start above.

The policy loads its tokenizer from `google/paligemma-3b-pt-224`, which is a gated repository.
Accept its terms on the Hub and run `hf auth login` before the first inference, or pre-download
it.

Config for a two-camera SO-101:

```yaml
robot:
  type: so101_follower
  port: /dev/ttyACM0
  id: <your_robot_id>
  calibration_dir: <path/to/lerobot/calibration>
  cameras:
    base_0_rgb:                 # overhead / external view
      type: opencv
      index_or_path: /dev/video0
      fourcc: MJPG
      width: 640
      height: 480
      fps: 30
    left_wrist_0_rgb:           # wrist view
      type: opencv
      index_or_path: /dev/video2
      fourcc: MJPG
      width: 640
      height: 480
      fps: 30

policy:
  path: ckpt/so101-pi05-base
  empty_cameras: 1              # fills the unused right_wrist_0_rgb slot
  device: cuda
  compile_model: true
  n_action_steps: 20
  num_inference_steps: 15

single_task: "Pick up the green cube block and put it inside the white cup."
fps: 30
control_time_s: 600
inference_overlap_steps: 8
```

```bash
vlash run config.yaml
```

Points that matter:

- **Camera keys must be the slot names.** The keys under `robot.cameras` are matched against
  the policy's image features by name. A key like `top` or `wrist` will not bind to a slot.
- **Set `empty_cameras` to the number of missing slots**: 1 for a two-camera setup, 2 for one
  camera. The checkpoint's `config.json` stores `empty_cameras: 0`, so this must be set
  explicitly. It only takes effect once the camera-validation patch is in place; without the
  patch the run aborts before `empty_cameras` is read. Assign the external view to `base_0_rgb`
  and a wrist view to `left_wrist_0_rgb`, matching how the training data was mapped.
- **`inference_overlap_steps > 0` requires `compile_model: true`**; VLASH rejects the
  combination otherwise. Overlap must be smaller than `n_action_steps`.
- The checkpoint stores `n_action_steps: 50` and `num_inference_steps: 10`. The values in the
  table above (20 and 15) are the ones used in the hardware evaluation and must be set
  explicitly to reproduce it.
- Set `fps` to the control rate the data was collected at (30 here). A different loop rate
  changes the time meaning of each action step.

### Camera pitfalls

- On Linux, LeRobot's OpenCV camera backend resolves to `CAP_ANY`, and some UVC cameras then
  fail on `set(fourcc/width)` and raise `VIDIOC_QBUF Bad file descriptor`. Forcing
  `cv2.CAP_V4L2` fixes it. Patching `lerobot.cameras.utils.get_cv2_backend` and
  `lerobot.cameras.opencv.camera_opencv.get_cv2_backend` at import time works without
  modifying site-packages.
- Set `fourcc: MJPG` explicitly. Several USB cameras default to a raw format that cannot
  sustain 640x480 at 30 fps over USB bandwidth.
- Verify which physical camera is on which device node by capturing a frame, not by assuming
  the enumeration order. Swapping the overhead and wrist views silently degrades the policy.

## Issues resolved during training

Four problems appeared in this run. They are recorded here because none of them were caught by
a short smoke test; all surfaced in the first few hundred steps of the full run or later.

**1. DataLoader file descriptor exhaustion.** With 8 workers per process across 8 GPUs on a
dataset of 51,411 videos, the default `nofile` limit of 1024 is exceeded and the run deadlocks
around step 70 with `Too many open files`. Raise the limit on the container or shell, for
example `docker run --ulimit nofile=1048576:1048576`.

**2. Timestamp tolerance violations on 10 fps sources.** LeRobot checks that a frame's parquet
timestamp matches the decoded video timestamp within a tolerance. Some 10 fps sources have a
first video PTS of 0.1 s while the parquet timestamps start at 0, and some carry `fps: 30` in
`info.json` while the video is 10 fps. Either mismatch aborts the run mid-training. The fix is
to align parquet timestamps to the video's actual first PTS (`ts - ts[0] + pts0`) and to trust
the video's real frame rate over the metadata. The 450 affected episodes were excluded from
this run rather than repaired in time for it.

**3. Unbounded video decoder cache causing RAM growth.** LeRobot's `VideoDecoderCache` keeps
one decoder per video path with no eviction. Across tens of thousands of distinct videos this
accumulates: anonymous memory grew about 1.6 GB per step and the process was eventually killed.
Bounding the cache as an LRU with a size limit and closing the file handle on eviction reduced
growth to about 0.015 GB per step. Setting `num_workers=0` does not fix it, since the leak is
then simply in the main process, and it idles the GPU. When diagnosing this, separate cgroup
`anon` (unreclaimable) from `file` (page cache) memory; total RSS alone is misleading.

**4. Corrupt video frames.** A few source videos have frames that fail to decode. Wrapping the
decode in a try/except with a fallback to frame 0 keeps a single bad frame from ending a
40-hour run.

## Patches to the training and inference stack

The stack used to produce and evaluate this checkpoint is upstream VLASH plus the 13 changes
below, spread over four files: `train.py` (4), `modeling_pi05.py` (3), `run.py` (4) and LeRobot's
`video_utils.py` (2). All four files ship with this release under `training-docker/patched/`, each
carrying a header naming its upstream and its changes, and the same list appears in `NOTICE`.

**Training** (1 to 8). Items 1 to 3 are no-ops on single-source datasets that carry no mask
columns; items 4 to 6 are inactive unless the S3 environment variables are set; items 7 and 8
apply to any long run over many distinct videos.

1. `modeling_pi05.py`, policy `__init__`: drop keys ending in `_mask` from
   `config.input_features`. Mask columns are float32 and would otherwise be classified as state
   features and rescaled by normalization, destroying the 0/1 values.
2. `modeling_pi05.py`, `prepare_images`: read `observation.images.{slot}_mask` from the batch
   per sample and use it as the attention mask, falling back to all-ones when the column is
   absent. This is what makes camera presence vary per episode within a batch.
3. `train.py`, dataset construction: when the dataset has `_mask` columns, apply a 224
   resize-with-pad image transform (`_resize_with_pad` / `_resize_with_pad_then_aug`). Source
   datasets have mixed resolutions (640x480 and 1920x1080 among others), which cannot be
   collated into one tensor otherwise. Doing it as a transform avoids re-encoding any video.
4. `train.py`, checkpointing on the LoRA branch: save the adapters only, skipping the merged
   model weights and the training state, and export the normalization buffers to
   `normalize_buffers.pt` so an adapter-only checkpoint stays deployable. Not used by this run,
   which is a full fine-tune.
5. `train.py`, checkpointing: upload a rolling resume checkpoint and a permanent archive
   checkpoint to S3 and prune the local copies, driven by the `SLOT_ID`, `S3_CKPT_BASE` and
   `RESUME_FREQ` environment variables. Inactive when those variables are unset.
6. `train.py`, W&B: pin the run id and the job name to the slot id and force `resume="allow"`
   when an S3 slot is configured, so a restarted preemptible instance appends to the same run
   instead of opening a new one.
7. `video_utils.py` (LeRobot), `VideoDecoderCache`: bound the decoder cache as an LRU of
   `VLASH_DECODER_CACHE_MAX` entries (default 64) and close the file handle on eviction. The
   upstream cache is unbounded; without this the process grows without limit across a corpus of
   51,411 videos (see Issues, item 3).
8. `video_utils.py` (LeRobot), `decode_video_frames_torchcodec`: on a decoder error, log the
   offending video path and fall back to frame 0 instead of aborting the run (see Issues,
   item 4).

**Inference** (9 to 12). Patch 9 is required for any setup with fewer than three cameras.

9. `run.py`, `validate_robot_cameras`: accept a subset of the policy's camera slots instead of
   demanding an exact match, and keep rejecting camera keys the policy does not know, so a typo
   in a slot name still fails loudly instead of silently degrading into a masked slot.
10. `run.py`, run loop: a dry-run mode (`VLASH_DRY_RUN=1`) that computes actions without
    commanding the motors, and an abort when a commanded action exceeds a magnitude bound
    (`VLASH_SAFE_DEG`, 180 in the evaluation) - which is what a wrong checkpoint path, wrong
    normalization or a NaN looks like at the joint level.
11. `run.py`, chunk switch: linearly blend the first `overlap_steps` actions of a new action chunk
    into the last action of the outgoing chunk. Without it there is a step discontinuity at every
    chunk boundary when asynchronous inference is enabled. No effect when
    `inference_overlap_steps` is 0.
12. `run.py`, `load_and_compile_policy`: load adapter-only checkpoints, combining the base
    weights with a LoRA merge, and restore the dataset statistics from `normalize_buffers.pt`.
    The counterpart of item 4; not exercised by this checkpoint, which is a merged full
    fine-tune.

**Offline installs** (13).

13. `modeling_pi05.py`, tokenizer load: read the PaliGemma tokenizer path from
    `VLASH_PALIGEMMA_PATH` instead of hardcoding the Hub id `google/paligemma-3b-pt-224`. The
    default keeps the Hub id, so this is a no-op unless the variable is set.

## License

The weights are released under the **Gemma Terms of Use**, not Apache-2.0. pi0.5 is built on
PaliGemma, which is a Gemma model, so this checkpoint is a Gemma derivative and inherits those
terms. The upstream base model `lerobot/pi05_base` is released under the same terms.

- Gemma Terms of Use: https://ai.google.dev/gemma/terms
- Gemma Prohibited Use Policy: https://ai.google.dev/gemma/prohibited_use_policy

If you redistribute these weights or anything derived from them, the Gemma Terms require you to:

1. Pass the use restrictions in Section 3.2 of the Gemma Terms of Use on to your recipients as
   enforceable provisions.
2. Give every recipient a copy of the Gemma Terms of Use.
3. Attach notices to any files you modify stating that you modified them.
4. Include a `NOTICE` text file with your distribution containing exactly this line:

```
Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms
```

The `NOTICE` file in this repository contains that line, and the `LICENSE` file in this repository
is a full copy of the Gemma Terms of Use retrieved from the URL above. Use of this model is
subject to the Gemma Prohibited Use Policy.

### Code

The training and inference code this model depends on is Apache-2.0 and is separate from the
weight license:

- [openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence) - pi0.5 reference implementation
- [VLASH](https://github.com/mit-han-lab/vlash) (MIT HAN Lab) - training and real-time inference stack
- [LeRobot](https://github.com/huggingface/lerobot) (Hugging Face) - dataset format, robot drivers, video decoding

Apache-2.0 requires that redistributions of that code carry a copy of the license, retain
attribution notices, and state which files were changed. The modifications made here are listed
in "Patches to the training and inference stack" above and in `NOTICE`. Apache-2.0 places no
conditions on model weights produced by running the code.

### Data

The released data subset comes from source datasets under apache-2.0 and mit. Apache-2.0
Section 4 requires attribution to be carried forward and a copy of the license to be distributed;
both ship with
[`dongyoonkim/so101-pi05-base-dataset`](https://huggingface.co/datasets/dongyoonkim/so101-pi05-base-dataset)
as `ATTRIBUTION.md` and `LICENSE`.

## Related

Released with this checkpoint:

- [`dongyoonkim/so101-pi05-base-dataset`](https://huggingface.co/datasets/dongyoonkim/so101-pi05-base-dataset),
  the licensed subset of the corpus this checkpoint was trained on.
- [jinnymo/so101-pi05-base](https://github.com/jinnymo/so101-pi05-base) - the full package: the
  dataset-construction pipeline, the training container, and seven documents covering the build,
  the training run, troubleshooting, inference and LoRA fine-tuning. The files this card refers to
  under `scripts/` and `training-docker/patched/` are mirrored here for convenience; the repository
  holds the rest.

Built on:

- [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) - the weights this fine-tune
  started from.
- [VLASH](https://github.com/mit-han-lab/vlash), [openpi](https://github.com/Physical-Intelligence/openpi),
  [LeRobot](https://github.com/huggingface/lerobot) - the code stack, listed under "License / Code"
  above.

Author's other work on the same robot:

- [jinnymo/lerobot-v3-v2-converter](https://github.com/jinnymo/lerobot-v3-v2-converter) - two-way
  LeRobot v2.1 / v3.0 dataset converter, used to normalize the v3.0 sources in the training corpus.
- [jinnymo/gr00t-n17-lora](https://github.com/jinnymo/gr00t-n17-lora) - restores LoRA fine-tuning
  on NVIDIA GR00T N1.7, which upstream removed, by monkey-patching Isaac-GR00T.
- [`dongyoonkim/grootn17-lora-so101-eraser-tier1`](https://huggingface.co/dongyoonkim/grootn17-lora-so101-eraser-tier1),
  the adapter that wrapper produced, and
  [`dongyoonkim/so101-eraser-90ep-wrist`](https://huggingface.co/datasets/dongyoonkim/so101-eraser-90ep-wrist),
  the 90-episode dataset it was trained on. A different base model and a task fine-tune rather than
  a domain adaptation, so the two are not comparable.

## References

- Black et al. [pi0: A Vision-Language-Action Flow Model for General Robot
  Control](https://arxiv.org/abs/2410.24164), arXiv:2410.24164. Source of the
  corrections-and-recoveries property of the pretraining mixture.
- Physical Intelligence. [pi0.5: a Vision-Language-Action Model with Open-World
  Generalization](https://arxiv.org/abs/2504.16054), arXiv:2504.16054. The base model of this
  checkpoint, trained on an extended version of the pi0 dataset.
- Yu and Qiu. [Benchmarking Vision-Language-Action Models on SO-101: Failure and Recovery
  Analysis](https://arxiv.org/abs/2606.08881), arXiv:2606.08881. Real-hardware SO-101 benchmark
  with a failure taxonomy and recovery-aware metrics; source of the 30.77% / 20.51% / 6.45% /
  3.23% recovery rates quoted above.
- Yadav et al. [Robust Finetuning of Vision-Language-Action Robot Policies via Parameter
  Merging](https://arxiv.org/abs/2512.08333) (RETAIN), arXiv:2512.08333. On policies fine-tuned
  to limited demonstrations of one task overfitting to them and losing prior generalist ability.
- Octo Model Team. [Octo: An Open-Source Generalist Robot
  Policy](https://arxiv.org/abs/2405.12213), arXiv:2405.12213. On the gripper action
  representation changing how strongly a policy retries.
- Schaeffer et al. [Are Emergent Abilities of Large Language Models a
  Mirage?](https://arxiv.org/abs/2304.15004), arXiv:2304.15004. On claimed discontinuous
  capability jumps in large models being largely an artifact of the evaluation metric rather
  than a property of the model.

## Citation

```bibtex
@misc{kim2026so101pi05base,
  title  = {so101-pi05-base: a pi0.5 base checkpoint fully fine-tuned on SO-101 data},
  author = {Dongyoon Kim},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/dongyoonkim/so101-pi05-base}}
}
```

Underlying model:

```bibtex
@article{physicalintelligence2025pi05,
  title   = {{$\pi_{0.5}$}: a Vision-Language-Action Model with Open-World Generalization},
  author  = {{Physical Intelligence}},
  journal = {arXiv preprint arXiv:2504.16054},
  year    = {2025}
}
```

## Acknowledgements

- **Physical Intelligence** for pi0.5 and the openpi reference implementation.
- **Google** for PaliGemma and Gemma.
- **MIT HAN Lab** for VLASH, whose asynchronous inference made the closed-loop evaluation
  practical at 30 Hz. The linear blend across chunk boundaries is not an upstream feature; it was
  added for this work (patch 11 above).
- **Hugging Face** for LeRobot and for hosting `lerobot/pi05_base`.
- The SO-101 and SO-100 community members who published their teleoperation datasets on the
  Hub. Almost all of this checkpoint's training distribution came from recordings other people
  uploaded of their own robots. The released data subset lists each source repository.
