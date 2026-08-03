# 07 — LoRA fine-tuning on top of this checkpoint

Documents 02 through 05 describe building this base. This one describes using it: taking the
released checkpoint as the starting point and training a LoRA adapter on demonstrations of your
own task, then running that adapter on the arm. The base is a domain adaptation, not a policy —
it knows the SO-101, its joint units and the follower-frame action convention, and it does not
know your task.

Scope note, stated once. The LoRA runs quoted here were made in this same stack, with the same
patched files, but on the stock `lerobot/pi05_base` — they predate this checkpoint. **No LoRA
fine-tune on top of the released base has been trained or evaluated.** Every number below is
labelled with what it came from, and where nothing was measured it says so.

Contents:

1. [Data](#1-data)
2. [Hardware](#2-hardware)
3. [Configuration](#3-configuration)
4. [Running the fine-tune](#4-running-the-fine-tune)
5. [Hyperparameters](#5-hyperparameters)
6. [The normalization statistics trap](#6-the-normalization-statistics-trap)
7. [Inference with an adapter](#7-inference-with-an-adapter)
8. [Choosing a checkpoint](#8-choosing-a-checkpoint)
9. [Cost](#9-cost)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Data

### 1.1 How many episodes

There is no measured curve. Three data points, none of which is a recommendation on its own:

| Evidence | Episodes | What it actually shows |
|---|---|---|
| The task evaluated in 06 §9 | 14 | Those 14 episodes were **inside** the base's own 16,687-episode training corpus, not a separate fine-tune. It bounds how much task data an adapted base needs to absorb a task, not how much a LoRA run needs |
| Earlier LoRA run, 1 camera | 90 | 20000 steps, 4.14 epochs, loss 0.10 -> 0.007. On stock pi0.5 |
| Earlier LoRA run, 2 cameras | 50 | Same recipe, effective batch 10. Wall clock not recorded |

Fifty episodes of one task is the smallest thing that was actually trained here. Below that
nothing has been tried in this stack. Note the direction of the risk: a narrow single-task LoRA
overfits to the demonstrations it has, which `model/README.md` discusses under the recovery
behavior — if your task needs the arm to retry after a failed grasp, the retries have to be in
the data, because a clean-approach-only dataset trains that behavior out.

### 1.2 Format

LeRobot v2.1, the layout the training stack reads:

```
<dataset>/
  meta/info.json               codebase_version "v2.1", fps, features, video_path template
  meta/episodes.jsonl
  meta/tasks.jsonl
  meta/stats.json              required; see below
  meta/episodes_stats.jsonl
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.<camera>/episode_000000.mp4
```

A v3.0 dataset has to be converted first. `pipeline/05_convert_v3_to_v2.py` drives the converter
listed under Related in the top-level README; 02 §5 covers it.

`meta/stats.json` is not optional. Without it `dataset.meta.stats[key]` is `None` and the run
dies at policy construction. If your recorder did not write one,
`training-docker/scripts/inject-stats <dataset_root>` computes it from the parquet files and
skips silently when one is already there.

### 1.3 Camera slots — get this wrong and the run is wasted

pi0.5 has three fixed image slots and the base was trained with the assignment in 02 §6:

```
base_0_rgb            external view (overhead, front, side — the corpus mixes them)
left_wrist_0_rgb      wrist view
right_wrist_0_rgb     never a real second wrist camera in this corpus; masked in most episodes
```

Slots are bound **by dataset feature name**, and the name is what carries the meaning. A dataset
whose cameras are called `cam_top` and `cam_wrist` will train — nothing raises — but the wrist
stream is no longer the stream the base learned to treat as a wrist view, and the adapter spends
its capacity relearning a mapping the base already had. That is the whole reason to fine-tune on
this base rather than on stock pi0.5.

Two ways to get the names right. Either record with LeRobot camera keys already named after the
slots, which is what `inference/eval.yaml` does for the run side, or rename an existing dataset.
The names appear in four places and all four have to move together:

```python
#!/usr/bin/env python3
"""Rename LeRobot v2.1 camera keys onto the pi0.5 slots.

Usage: rename_slots.py <dataset_root> old=new [old=new ...]
Bare camera names, without the observation.images. prefix.
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
mapping = {f"observation.images.{a}": f"observation.images.{b}"
           for a, b in (arg.split("=", 1) for arg in sys.argv[2:])}

def rekey(d):
    return {mapping.get(k, k): v for k, v in d.items()}

info = json.loads((root / "meta/info.json").read_text())
info["features"] = rekey(info["features"])
(root / "meta/info.json").write_text(json.dumps(info, indent=4))

stats = root / "meta/stats.json"
if stats.exists():
    stats.write_text(json.dumps(rekey(json.loads(stats.read_text())), indent=4))

eps = root / "meta/episodes_stats.jsonl"
if eps.exists():
    out = []
    for line in eps.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            rec["stats"] = rekey(rec["stats"])
            out.append(json.dumps(rec))
    eps.write_text("\n".join(out) + "\n")

for chunk in sorted((root / "videos").glob("chunk-*")):
    for old, new in mapping.items():
        if (chunk / old).is_dir():
            (chunk / old).rename(chunk / new)

print("renamed:", ", ".join(f"{k} -> {v}" for k, v in mapping.items()))
```

```bash
python rename_slots.py <dataset> cam_top=base_0_rgb cam_wrist=left_wrist_0_rgb
```

Work on a copy. Assign the external view to `base_0_rgb` and a wrist view to
`left_wrist_0_rgb`, in that order, and leave the trailing slot to `empty_cameras` — the same
fill order the base was trained with.

**Then set `policy.empty_cameras` to `3 - <number of cameras>` in the training config.** The
policy appends that many black placeholders with their attention mask set to `False`, which is
exactly the arrangement every SO-101 episode in the base corpus was trained under (06 §5.2 has
the mechanics). Leaving it at 0 with two cameras trains the policy on a two-image prefix, and
then the adapter and the base disagree about the shape of the input.

### 1.4 Actions must be in the follower frame

The base was trained on `action` recorded in the follower arm's calibration frame, and 02 §3 is
the screen that enforced it across 211 source datasets. A recording whose `action` carries the
*leader* arm's calibration zero has a constant per-joint offset against the state it produces.
Normalization absorbs a scale difference; it does not absorb that offset. In closed loop the
offset accumulates and the arm diverges, while the training loss looks entirely normal — the
policy learned the offset data faithfully.

The cheap check, from 02 §3: for each joint take the mean absolute difference between `action[t]`
and `observation.state[t]`. A follower-frame recording commands roughly what it achieves, so the
offset is small; the screening threshold used there was 6 degrees for suspicion and 15 for
rejection. Some recorders write the leader position into `action` by default, so this is worth
checking on your own recordings and not only on downloaded ones.

### 1.5 Pre-flight checks

Both of these correspond to failures that killed paid runs (04 §2, 05 §2):

1. **Every episode's parquet `timestamp` starts where its video starts.** LeRobot asserts they
   agree within 1e-4 s. A 10 fps recording whose first video PTS is 0.1 s while the parquet
   starts at 0 fails at whatever step first samples that episode, not at step 0.
2. **`info.json` `fps` matches the actual videos.** A dataset assembled from mixed sources can
   carry `fps: 30` in metadata over 10 fps video, which produces the same failure.

Check every episode, not a sample. In the corpus these affected 2.6% of episodes and a random
sample missed them.

---

## 2. Hardware

### 2.1 What decides whether it fits

One GPU. `vlash train` counts visible GPUs and launches DDP across all of them, but nothing
about a LoRA run needs more than one, and a single card avoids the failure modes in 04 §5.

Three things set memory: the number of cameras, the per-GPU batch size, and nothing else you can
reach. Each camera runs the vision tower once and adds an image's worth of tokens to the prefix,
so a second camera roughly doubles the vision activations. Gradient accumulation does not change
memory — it is the knob that buys effective batch size back after you lower the real one.

**There is no gradient-checkpointing escape hatch.** `gradient_checkpointing` exists as a field
on the pi0.5 policy config and is never read anywhere in the training stack; setting it true
changes nothing. Older notes that treat it as a memory lever are wrong.

### 2.2 Measured

| Configuration | Card | Result |
|---|---|---|
| 1 camera, batch 5, accum 2 (effective 10), 20000 steps, 90 episodes | RTX 3090 24 GB | Ran. 5.75 h |
| 2 cameras, batch 5, accum 2 | RTX 3090 24 GB | **Out of memory** |
| 2 cameras, batch 2, accum 5 (effective 10) | RTX 3090 24 GB | The configuration used after the OOM. Peak memory and wall clock not recorded |
| 2 cameras, batch 10, accum 1 (effective 10) | L40S 48 GB | Ran. Wall clock not recorded |

So: 24 GB is enough for one camera at the recipe below, and is not enough for two cameras at the
same per-GPU batch. Whether 24 GB holds two cameras at batch 2 was never measured cleanly — the
run was launched at that setting and did not report OOM, which is weaker evidence than a peak
memory figure.

### 2.3 What to rent

**Recommendation: a 48 GB card, and among those, the L40S — because it is the only 48 GB card a
2-camera run in this stack has actually been trained on.** That is the entire argument. It is
not a claim that the L40S is the fastest or the cheapest 48 GB card for this workload; nothing
here benchmarked one 48 GB card against another.

48 GB rather than 24 GB buys the thing that matters at this scale, which is not speed: it lets
you keep the per-GPU batch at the recipe's value with two cameras instead of dropping to batch 2
and multiplying accumulation, and it removes the OOM restart from the middle of a run you are
paying for by the hour.

Prices per GPU-hour, **retrieved 2026-08-03**, all from provider pricing pages or live API
queries. Marketplace prices move hourly; re-check before renting.

| GPU | VRAM | Provider / tier | $/hr |
|---|---|---|---|
| L40S | 48 | Vast.ai on-demand, cheapest of 8 listings | 0.401 |
| L40S | 48 | Vast.ai on-demand, median of 8 | 0.801 |
| L40S | 48 | RunPod Community / Secure | 0.79 / 0.99 |
| L40S | 48 | AWS `g6e.xlarge` on-demand | 1.861 |
| RTX A6000 | 48 | RunPod Community / Secure | 0.33 / 0.49 |
| A40 | 48 | RunPod Community / Secure | 0.35 / 0.44 |
| L40 | 48 | RunPod Community / Secure | 0.69 / 0.82 |
| RTX 6000 Ada | 48 | RunPod Community / Secure | 0.74 / 0.77 |
| RTX 3090 | 24 | Vast.ai on-demand, median of 64 | 0.137 |
| RTX 4090 | 24 | RunPod Community / Secure | 0.34 / 0.69 |
| RTX 5090 | 32 | RunPod Community / Secure | 0.69 / 0.99 |
| A100 PCIe | 80 | RunPod Community / Secure | 1.19 / 1.39 |
| H100 SXM | 80 | RunPod Community / Secure | 2.69 / 2.99 |

The A6000 and the A40 hold the same 48 GB for roughly half the L40S price. Their throughput on
this workload was **not measured**, so whether the cheaper hour is also the cheaper run is
unknown. If you are price-sensitive and willing to time a smoke run yourself, they are the
obvious things to try; measure `updt_s` at step 100 and compare.

80 GB cards are not needed. A LoRA run trains a few percent of the parameters and holds optimizer
state only for those, which is what forced 80 GB per GPU on the base's full fine-tune (04 §5).

Two constraints that are not about the GPU:

- **Put the instance near your data.** 04 §1 has the measurement that makes this a rule rather
  than a preference: a host on the wrong side of an ocean from the bucket sustained 0.87 MiB/s.
  A task dataset is small enough that this is survivable, unlike the 198 GB corpus, but you are
  still paying GPU rent while it transfers.
- **Idle time bills at the same rate.** Terminate as soon as the checkpoints are somewhere that
  survives the instance.

---

## 3. Configuration

`training-docker/configs/base.yaml` is the full fine-tune config, and
`training-docker/scripts/train-base` is hardwired to it: it renders that one template through
`envsubst` and has no flag that turns LoRA on. **A LoRA run therefore does not use the wrapper.**
Write a config and call `vlash train` with it directly.

The config below is `base.yaml` with the LoRA block from the runs in §2.2 restored and the
pretrained path pointed at this release. Save it as `lora.yaml`.

```yaml
policy:
  type: pi05
  pretrained_path: /workspace/base/so101-pi05-base   # this release's checkpoint
  push_to_hub: false                 # avoids the lerobot validation error when repo_id is unset
  dtype: bfloat16
  device: cuda
  state_cond: true
  compile_model: false
  fuse_qkv: false                    # interacts with the LoRA merge; false is the tested path
  fuse_gate_up: false
  empty_cameras: 1                   # 3 - (number of cameras you provide)

dataset:
  repo_id: my_task
  root: /workspace/datasets/my_task
  video_backend: torchcodec

job_name: my_task_lora
output_dir: /workspace/checkpoints/my_task_lora     # must not already exist
batch_size: 2
grad_accum_steps: 5
steps: 20000
num_workers: 4
seed: 1000

use_policy_training_preset: false
optimizer:
  type: adamw
  lr: 5.0e-5
  betas: [0.9, 0.95]
  weight_decay: 1.0e-10

scheduler:
  type: cosine_decay_with_warmup
  num_warmup_steps: 1000
  peak_lr: 5.0e-5
  decay_lr: 2.5e-6
  num_decay_steps: 20000            # keep equal to steps

save_checkpoint: true
save_freq: 2000
log_freq: 10

wandb:
  enable: false
  project: unused                   # a value is required even when disabled
  disable_artifact: true            # an adapter-only checkpoint has no model.safetensors to upload

max_delay_steps: 8

lora:
  enable: true
  backend: peft
  r: 16
  alpha: 16
  dropout: 0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, out_proj, fc1, fc2]
  extra_trainable_modules:
    - action_in_proj
    - action_out_proj
    - time_mlp_in
    - time_mlp_out
    - state_proj
    - state_mlp_in
    - state_mlp_out
    - embeddings
    - input_layernorm
    - post_attention_layernorm
```

Set `batch_size` and `grad_accum_steps` from §2.2 for your camera count and card, keeping their
product at 10. `num_decay_steps` must track `steps`, or the cosine schedule ends somewhere other
than where training does.

`output_dir` must not exist. LeRobot raises `FileExistsError` on a directory that is already
there and `resume` is false, which is a guard against silently overwriting a previous run.

---

## 4. Running the fine-tune

### 4.1 Get the base

```bash
pip install huggingface_hub
hf download dongyoonkim/so101-pi05-base --local-dir base/so101-pi05-base
python scripts/verify_checkpoint.py base/so101-pi05-base
```

Do not skip the verify step. A truncated download of a 7 GB file is the kind of thing that shows
up eight hours into a rented run.

### 4.2 In the container

Build the image from `training-docker/` as `03-training-environment.md` describes. It carries the
four patched files, the PaliGemma tokenizer and a conda environment already on `PATH`, so
`vlash` runs without any activation step.

```bash
docker run --gpus all \
  --shm-size=8g \
  --ulimit nofile=1048576:1048576 \
  -v $PWD/base/so101-pi05-base:/workspace/base/so101-pi05-base:ro \
  -v $PWD/datasets/my_task:/workspace/datasets/my_task:ro \
  -v $PWD/checkpoints:/workspace/checkpoints \
  -v $PWD/lora.yaml:/workspace/lora.yaml:ro \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  <YOUR_REGISTRY>/pi05-so101-train:latest \
  vlash train /workspace/lora.yaml
```

| Flag | Why |
|---|---|
| `--shm-size=8g` | Docker defaults `/dev/shm` to 64 MB; DataLoader workers pass tensors through it and the run dies with `Bus error`. 8 GB is what the single-GPU runs used |
| `--ulimit nofile` | Cheap insurance. Descriptor exhaustion (05 §1) is a function of workers times videos and a task dataset is far smaller than the corpus, but the failure mode is a silent hang |
| `-e PYTORCH_CUDA_ALLOC_CONF` | What `train-base` sets for the full fine-tune; reduces allocator fragmentation |
| mounts read-only | The base and the dataset are inputs. Only `/workspace/checkpoints` is written |

The image sets `HF_HUB_OFFLINE=1` and `VLASH_PALIGEMMA_PATH`, so nothing is fetched from the Hub
at run time.

If your dataset has no `meta/stats.json`, generate it before training:

```bash
docker run --rm -v $PWD/datasets/my_task:/data <YOUR_REGISTRY>/pi05-so101-train:latest \
  /opt/scripts/inject-stats /data
```

### 4.3 Without the container

Same environment as the inference install in 06 §2.1, with all four patched files in place rather
than the two that inference alone needs. `peft` arrives with the stack.

```bash
conda create -n so101-lora python=3.10
conda activate so101-lora
conda install ffmpeg=7.1.1 -c conda-forge

git clone https://github.com/mit-han-lab/vlash
cd vlash
git checkout 22cbabfee0f57874987c75a35a7dac129e695db0
pip install -e .
pip install -U torch torchvision torchcodec

PKG=$(python -c "import lerobot, pathlib; print(pathlib.Path(lerobot.__file__).parent)")
cp <package>/training-docker/patched/train.py           vlash/train.py
cp <package>/training-docker/patched/run.py             vlash/run.py
cp <package>/training-docker/patched/modeling_pi05.py   vlash/policies/pi05/modeling_pi05.py
cp <package>/training-docker/patched/video_utils.py     "$PKG/datasets/video_utils.py"
```

All four are whole-file replacements written against that commit; on a moved upstream tree they
revert unrelated changes. `video_utils.py` goes into the installed `lerobot`, not into the vlash
tree, and must be copied after `pip install -e .`.

The PaliGemma tokenizer is gated. Accept the terms on the Hub and `hf auth login`, or point
`VLASH_PALIGEMMA_PATH` at a local copy (06 §2.3).

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
vlash train lora.yaml 2>&1 | tee train.log
```

### 4.4 Smoke run first

Copy the config, set `steps: 20`, `save_freq: 10` and a different `output_dir`, and run it. It
takes minutes and it proves the dataset loads, the base loads, the LoRA layers attach, an
optimizer step completes, and — the part that actually matters — that a checkpoint lands on
disk in the layout §7 needs.

Then run §6's check on the checkpoint the smoke run wrote, before starting the real run. A smoke
test that only proves the process exits 0 has not proved anything about the artifact.

What to watch once the real run starts, from the log at `log_freq: 10`: `updt_s` should be flat,
`data_s` near zero, and the loss should be falling inside the first few hundred steps. The
90-episode run went 0.10 to 0.007 over 20000 steps.

---

## 5. Hyperparameters

The values in §3, with what is known about each.

### Do not change without a reason

| Key | Value | Why |
|---|---|---|
| `lora.r` / `lora.alpha` | 16 / 16 | Effective LoRA scaling is `alpha / r`, so equal values mean a scale of 1. This pair is what both runs used |
| `lora.target_modules` | the ten listed | Attention and MLP projections across both the VLM and the action expert. This is the upstream VLASH list and what both runs used; narrower lists were not tried |
| `lora.extra_trainable_modules` | the ten listed | These are trained in full, not low-rank: the action in/out projections, the time and state MLPs, embeddings and layer norms. They are small and they are what makes an adapter able to move the action distribution at all |
| `policy.dtype` | bfloat16 | What the base was trained in (04 §4) |
| `max_delay_steps` | 8 | Trains under a random inference delay in [0, 8], which is what makes the asynchronous inference in 06 §6 behave. Set it to 0 only if you will run synchronously |
| `policy.fuse_qkv`, `fuse_gate_up` | false | Both interact with the LoRA merge. False is the tested path; true was not tried with LoRA enabled |
| `use_policy_training_preset` | false | Otherwise the policy's built-in optimizer preset overrides the blocks above |

### Change deliberately

| Key | Value used | What moving it does |
|---|---|---|
| `batch_size` / `grad_accum_steps` | product 10 | Per-GPU batch sets memory; accumulation buys the effective batch back. Halve one and double the other to fit a smaller card. Effective batch 10 is what both runs used and it is far below the base's 256, which is appropriate — this is a small dataset |
| `steps` | 20000 | 20000 at effective batch 10 was 4.14 epochs over 90 episodes. Scale it toward that epoch count for a different dataset size rather than copying the step number |
| `num_decay_steps` | = `steps` | Must move with `steps` |
| `optimizer.lr` / `scheduler.peak_lr` | 5e-5 | The standard LoRA fine-tune rate, and the same rate the base's full fine-tune used. An adapter tolerates more than a full fine-tune does, so this is a floor rather than a ceiling, but nothing higher was tried here |
| `scheduler.num_warmup_steps` | 1000 | 5% of 20000. Keep it proportional |
| `save_freq` | 2000 | Ten checkpoints over 20000 steps, which is what §8's sweep needs. Adapter checkpoints are about 571 MB each, so ten of them is under 6 GB — there is no disk reason to save less often |
| `num_workers` | 4 | Raise it if `data_s` in the log is not near zero |
| `policy.empty_cameras` | `3 - cameras` | §1.3 |

Two values that differ from the full fine-tune, deliberately or otherwise:

- `optimizer.grad_clip_norm` is set to 1.0 in `base.yaml` and is **not set** in the LoRA config,
  which leaves it at the framework default of 10.0. That is what the runs in §2.2 used. It was
  not a considered choice and it was not compared against 1.0.
- `weight_decay` 1e-10 is decay effectively off, for the same reason as the full fine-tune: over
  a handful of epochs of adaptation it only pulls pretrained weights toward zero.

---

## 6. The normalization statistics trap

**Read this before the first run, not after the first NaN.** It costs a run, and on hardware the
guard that would normally catch a bad action does not catch this one.

### 6.1 Symptom

The adapter trains, the loss curve looks correct, the checkpoint saves, and then at inference
every predicted action is `nan`. In a dry run (06 §8.1) the printed action dict is all `nan`.

The action magnitude abort in `run.py` **does not fire on this.** It tests
`abs(float(v)) > threshold`, and any comparison against `nan` is false, so a fully NaN action
passes the safety check and reaches `robot.send_action`. What happens on the bus from there is
not something this stack defines. Dry run first; the guard is not a substitute.

### 6.2 Cause

pi0.5 normalizes `observation.state` on the way in and un-normalizes `action` on the way out,
using mean/std buffers held on the policy:

```
normalize_inputs.buffer_observation_state.mean / .std
normalize_targets.buffer_action.mean / .std
unnormalize_outputs.buffer_action.mean / .std
```

They are created filled with `torch.inf` so that uninitialized statistics are detectable, and the
stack does **not** assert on them — the helper that produces the "is infinity" message exists in
`vlash/policies/normalize.py` and nothing calls it. `inf` therefore propagates to `nan` in
silence.

A full fine-tune has no problem here: the buffers are `nn.Parameter`s, so they are inside
`model.safetensors` and travel with the weights (06 §4.1 lists the six keys). A LoRA checkpoint
does not contain them. PEFT's `save_pretrained` writes the LoRA matrices and the
`modules_to_save` modules, both of which live under `policy.model`; the normalize buffers hang
off the policy itself and are not part of either set. An adapter saved by stock VLASH is
therefore missing exactly the six numbers per feature that keep its output finite.

### 6.3 What this release does about it

Both halves of the fix are in the patched files, so if you train and run with them there is
nothing to do:

- `training-docker/patched/train.py`, on the LoRA branch of the checkpoint save, dumps the
  input and output buffers to `normalize_buffers.pt` next to the adapter.
- `training-docker/patched/run.py`, after merging the adapter, loads that file and copies the
  values into all three buffer sets. When the file is missing it prints
  `normalize_buffers.pt missing (...) - inference may produce NaN` and continues.

That warning line in the run log is the thing to grep for. It is the difference between a
checkpoint that works and one that produces NaN, and it does not stop the run.

### 6.4 Which statistics you actually get

Worth knowing, because it is not what most people assume. `make_policy` builds the policy with
your dataset's statistics, and then `from_pretrained` loads the base checkpoint's state dict over
it with `strict=False`. The normalize buffers are parameters, they are keys in that state dict,
so **the base's statistics win and your dataset's are discarded.**

This is self-consistent rather than broken: the same statistics are used to normalize during
training and to un-normalize at inference, so the adapter learns against them. It matters in two
ways. Your `normalize_buffers.pt` will not match your dataset's `meta/stats.json`, and that is
expected, not a bug. And an adapter is only valid on top of the base it was trained on — merging
it onto a different base pairs your adapter with different statistics. §7.2 is the same point
from the other direction.

### 6.5 Verify a saved adapter

Run this on the directory you intend to pass as `policy.path`, on the smoke-run checkpoint first
and on every checkpoint you plan to deploy. It needs `torch` and `safetensors` and no GPU.

```python
#!/usr/bin/env python
"""Check that an adapter checkpoint is complete and will load.

Usage: check_adapter.py <the directory you will pass as policy.path>
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

p = Path(sys.argv[1])
ok = True


def report(passed, msg):
    global ok
    ok = ok and bool(passed)
    print(f"{'PASS' if passed else 'FAIL'}  {msg}")


for f in ("config.json", "train_config.json", "normalize_buffers.pt",
          "lora_adapters/adapter_config.json",
          "lora_adapters/adapter_model.safetensors"):
    report((p / f).is_file(), f)

report(p.name == "pretrained_model",
       f"directory name is 'pretrained_model' (got '{p.name}')")

if (p / "lora_adapters/adapter_model.safetensors").is_file():
    sd = load_file(p / "lora_adapters/adapter_model.safetensors")
    n_lora = sum("lora_" in k for k in sd)
    n_full = len(sd) - n_lora
    finite = all(torch.isfinite(v.float()).all() for v in sd.values())
    report(n_lora > 0 and finite,
           f"adapter tensors: {n_lora} lora, {n_full} fully trained, all finite = {finite}")

if (p / "normalize_buffers.pt").is_file():
    stats = torch.load(p / "normalize_buffers.pt", weights_only=True)
    for key in ("observation.state", "action"):
        s = stats.get(key)
        good = (s is not None
                and torch.isfinite(s["mean"]).all()
                and torch.isfinite(s["std"]).all()
                and (s["std"] > 0).all())
        report(good, f"normalize_buffers.pt holds finite {key} mean/std")

sys.exit(0 if ok else 1)
```

A healthy checkpoint:

```
PASS  config.json
PASS  train_config.json
PASS  normalize_buffers.pt
PASS  lora_adapters/adapter_config.json
PASS  lora_adapters/adapter_model.safetensors
PASS  directory name is 'pretrained_model' (got 'pretrained_model')
PASS  adapter tensors: 1332 lora, 125 fully trained, all finite = True
PASS  normalize_buffers.pt holds finite observation.state mean/std
PASS  normalize_buffers.pt holds finite action mean/std
```

The tensor counts depend on your `target_modules` and `extra_trainable_modules`; what matters is
that both are non-zero and everything is finite. The directory-name check is §7.1 and is the one
that catches a checkpoint damaged in transit.

---

## 7. Inference with an adapter

06 covers robot wiring, camera backend, calibration, every runtime option, the dry-run procedure
and safety. All of it applies unchanged. Only three things differ for an adapter.

### 7.1 The layout is not the same as a full checkpoint

A LoRA checkpoint has one more directory level than the flat three-file layout in 06 §4.1:

```
<output_dir>/checkpoints/<step>/pretrained_model/
    config.json                 policy config saved at training time
    train_config.json           read at load time for the LoRA hyperparameters
    normalize_buffers.pt        §6
    lora_adapters/
        adapter_config.json
        adapter_model.safetensors     about 571 MB
```

**`policy.path` points at `pretrained_model`, not at `<step>`.** The loader resolves the adapter
as `policy.path.parent / "pretrained_model" / "lora_adapters"`, so the name of that directory is
load-bearing.

This has already cost a full debugging cycle once (06 §4.2). A sync tool flattened the
`pretrained_model` level out of an archived checkpoint. `policy.path` was still pointing one
level too high, and the failure is silent in the worst way: the loader finds a `lora_adapters`
directory so it takes the adapter branch, `load_lora_adapters` then looks one level further up,
finds nothing, logs a warning and returns `False`. The adapters that get merged are the
freshly-built ones, whose `lora_B` is zero-initialized, so the merge is the identity. The result
is the stock base running with your statistics, on a robot, with no exception raised. Different
distribution paths produce different depths — an extracted archive and a directory sync do not
agree — so check rather than assume:

```bash
ls -d "$CKPT/lora_adapters" && basename "$CKPT"
```

### 7.2 The base path is hardcoded — one line has to change

`training-docker/patched/run.py` line 452:

```python
        base_model_path = "lerobot/pi05_base"
```

The adapter branch merges onto whatever that names. It is the stock pi0.5 base, because that is
what the LoRA runs behind this code were trained on. **An adapter trained on this release's base
must be merged onto this release's base**, so change it to your local copy:

```python
        base_model_path = "/path/to/base/so101-pi05-base"
```

Getting this wrong does not raise. The adapter merges onto the wrong weights and the arm moves;
it just moves like something that was never trained on your task, which is the same symptom as
§7.1 and is diagnosed the same way. If you keep both bases around, keep the string in sync with
`policy.pretrained_path` from the config that trained the adapter — that field is recorded in
`train_config.json` inside the checkpoint.

### 7.3 Running it

Everything else is 06. The command is 06 §7's, with `policy.path` at the `pretrained_model`
directory. Dry run before commanding motors, always:

```bash
export VLASH_DRY_RUN=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python inference/v4l2_launch.py run inference/eval.yaml \
  --policy.path=<OUTPUT_DIR>/checkpoints/010000/pretrained_model \
  --policy.compile_model=false \
  --inference_overlap_steps=0 \
  --control_time_s=30
```

Three lines to look for, in order. The first is logged at info level by the adapter loader, the
other two are printed by the patched code:

```
[LoRA] Loaded adapter weights from ...: <N> LoRA params, <M> modules_to_save params
[patch] normalize_buffers.pt applied (state + action stats)
[step 0] action: {'shoulder_pan.pos': ..., ...}
```

A `No lora_adapters directory found` warning instead of the first line is §7.1. A
`normalize_buffers.pt missing` warning instead of the second is §6. `nan` in the third is the
consequence of having ignored the second.

`single_task` must be the instruction your demonstrations were recorded under. 06 §8.3 covers why
the trained phrasing and a probe phrasing have to be kept apart when interpreting results.

---

## 8. Choosing a checkpoint

**The lowest training loss is not the best policy on hardware.** In an earlier fine-tune in this
stack the lowest-loss checkpoint was step 18000 and the checkpoint that actually worked best on
the arm was step 12000 — a third of the way earlier, at a visibly worse loss. The 90-episode run
also had step 20000 above step 18000 on loss, which is the beginning of overfitting on a
90-episode dataset at 4 epochs.

Save every 2000 steps and sweep the mid-to-late band, roughly 30% to 100% of total steps. Below
that checkpoints are undertrained. Hold everything constant except `policy.path`:

```bash
for STEP in 006000 008000 010000 012000 014000 016000 018000 020000; do
  echo "=== step $STEP ==="
  python inference/v4l2_launch.py run inference/eval.yaml \
    --policy.path=<OUTPUT_DIR>/checkpoints/$STEP/pretrained_model \
    --single_task="$PROMPT" \
    --policy.compile_model=false \
    --inference_overlap_steps=0 \
    --control_time_s=120
done
```

Reset the scene to the same layout between checkpoints, run a fixed number of trials each, and
record successes rather than impressions. The sweep runs synchronously here on purpose:
`compile_model: true` pays two to three minutes of warmup per checkpoint, which is most of an
hour across eight of them. Re-run the winner under the operating configuration of 06 §7 before
concluding anything about it.

Adapters are small, so keeping all ten costs under 6 GB. Keep them until the sweep is done.

---

## 9. Cost

Estimates. One wall-clock figure below was measured and the rest are derived from it or from the
author's own pre-run estimates; prices are the §2.3 snapshot, retrieved 2026-08-03.

| Run | Wall clock | Source |
|---|---|---|
| 1 camera, 90 episodes, 20000 steps, effective batch 10 | 5.75 h | Measured, RTX 3090 |
| 2 cameras, 20000 steps, effective batch 10 | 12-20 h | Estimated from the camera-count cost, never measured |

Against the price table, ignoring transfer and storage:

| Run | at $0.35/h | at $0.80/h | at $1.86/h |
|---|---|---|---|
| 5.75 h | ~$2 | ~$5 | ~$11 |
| 12 h | ~$4 | ~$10 | ~$22 |
| 20 h | ~$7 | ~$16 | ~$37 |

The order of magnitude is what to take from this: a task LoRA on top of this base is a
tens-of-dollars job on a rented GPU, against roughly $570-890 for the base's own 40-hour
full fine-tune (01, Cost). That difference is the reason the base exists.

Not included, and not recorded: object storage, egress, and the time spent recording
demonstrations, which is the part that is actually expensive.

---

## 10. Troubleshooting

Only the LoRA-specific failures. Training problems that are not about LoRA are in 05; robot,
camera and calibration problems are in 06 §11.

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: LoRA is enabled but the peft library is not installed` | `peft` missing from the environment | `pip install peft`; the container has it |
| `LoRA is enabled but this policy does not expose a .model attribute` | LoRA enabled against a policy type that is not pi0/pi0.5 | Only pi0 and pi0.5 are supported |
| `FileExistsError: Output directory ... already exists` | `output_dir` from a previous run | Use a new directory. LeRobot refuses to overwrite when `resume` is false |
| `TypeError: 'NoneType' object is not subscriptable` at policy construction | `meta/stats.json` missing | `training-docker/scripts/inject-stats <dataset_root>` |
| Checkpoint directory holds `lora_adapters/` but no `model.safetensors` | Expected | The LoRA branch saves adapters only. §7.1 has the layout |
| W&B artifact upload fails at the first checkpoint | The artifact logger expects `model.safetensors` | `wandb.disable_artifact: true`, already set in §3 |
| `[LoRA] No lora_adapters directory found at ...` at load | `policy.path` is not the `pretrained_model` directory, or the level was flattened in transit | §7.1 |
| Runs, no error, behaves like an untrained policy | The same thing, one level up: the adapter branch was taken but nothing was loaded, so a zero-initialized adapter merged as the identity | §7.1, then §7.2 |
| `normalize_buffers.pt missing (...) - inference may produce NaN` in the log | The checkpoint was written by an unpatched `train.py`, or the file was lost in transit | §6. The adapter cannot be repaired after the fact except by re-deriving the statistics from the base it was trained on |
| Every action is `nan`, magnitude guard does not abort | Uninitialized normalize buffers | §6. `nan` fails every comparison, so the guard cannot see it |
| Actions are finite but the motion ignores the task | Adapter merged onto the wrong base | §7.2 |
| Out of memory with two cameras at batch 5 on 24 GB | Expected, measured | Batch 2 with accumulation 5, or a 48 GB card. `gradient_checkpointing` is not implemented and will not help |
| `Bus error` at startup | Docker's 64 MB `/dev/shm` | `--shm-size=8g` |
