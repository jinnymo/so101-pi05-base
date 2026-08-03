# so101-pi05-base

Physical Intelligence's **pi0.5** vision-language-action model, fully fine-tuned on a unified
corpus of public SO-101 / SO-100 single-arm manipulation data. This package holds the model
card, the dataset card, the scripts that built the dataset, the training container, and the
documentation needed to repeat the work.

The checkpoint is a domain-adaptation base, not a finished policy: it is the starting point of
a task-specific fine-tune.

## Why this exists

A pretrained VLA checkpoint is not usable on a real arm as it ships. Every backbone tried here —
GR00T N1.7, SmolVLA, pi0, pi0.5 — had to be fine-tuned before it produced motion worth running on
hardware. Untuned output is erratic, and erratic output on a real arm is a hazard, not a poor
score. Fine-tuning is the precondition, not an optimization.

That moves the problem to data. A task fine-tune needs demonstrations, and recording them one at a
time does not get far. This corpus is an attempt to pay that cost once: 181 SO-101 datasets merged
into one, 17,137 episodes of which 16,687 were fed to the run, used to adapt pi0.5 to the robot
rather than to any task, so a task fine-tune on top starts from something that already knows the
arm. The released dataset is the redistributable part of that corpus — 156 sources and 13,969
episodes, the ones whose upstream repository declares a license.

### Why pi0.5

Several backbones were fine-tuned on the same data with LoRA and run on the same SO-101 hardware.
The pi0 family behaved best. No controlled benchmark was run and none is claimed: the comparison
was LoRA fine-tunes judged by closed-loop behavior on real hardware, which was clear enough in
that setting to decide the direction but is not a measurement.

### Why VLASH

Chunked action prediction leaves a gap. While the current chunk executes, the next inference has
not started, and the arm stops for the length of the forward pass. Measured on this setup,
synchronous inference stalled 13.7% of wall clock locally and 44% over a cloud link.

Overlapping the next inference with the current chunk closes that gap and exposes a second
problem: the first action of a new chunk does not continue the last action of the old one. A
stationary arm oscillated 1-2 degrees at every chunk boundary, and widening the overlap window to
533 ms did not remove it, which ruled out latency as the cause.

VLASH contributes the two pieces that make overlapping viable: it samples a random inference delay
during training, so the policy is trained for the condition it will run under, and it runs the next
inference concurrently with the current chunk. What closed the boundary discontinuity is a patch
added here — a linear ramp blending the first actions of a new chunk into the last of the old one
(patch 3.3, `training-docker/patched/run.py`, listed in `training-docker/NOTICE`). That is what
removed the oscillation: 75%, 89% and 26% reduction across three joints in the hold phase.
LeRobot's asynchronous path averages the overlapping actions instead, and the average of two valid
trajectories is not necessarily a valid trajectory.

VLASH supports pi0 and pi0.5, and does not support GR00T. The backbone and the inference stack
were therefore one decision rather than two.

## Start here

[`docs/01-overview.md`](docs/01-overview.md) — what was built, the result, the pipeline in
order, hardware and account requirements, and cost. The other six documents go stage by stage:

| Document | Covers |
|---|---|
| [`docs/02-dataset-construction.md`](docs/02-dataset-construction.md) | Hub crawl, screening, download, conversion, camera slots, deduplication, merge, statistics, verification |
| [`docs/03-training-environment.md`](docs/03-training-environment.md) | container image, the four patched source files, `base.yaml`, build verification |
| [`docs/04-training-execution.md`](docs/04-training-execution.md) | instance setup, data staging, launch command, every hyperparameter, monitoring |
| [`docs/05-troubleshooting.md`](docs/05-troubleshooting.md) | the problems that actually occurred, with the diagnosis path for each |
| [`docs/06-inference.md`](docs/06-inference.md) | robot and camera setup, checkpoint layout, runtime options, evaluation, limitations |
| [`docs/07-lora-finetuning.md`](docs/07-lora-finetuning.md) | using the checkpoint: data requirements, GPU choice, the LoRA config, the normalization-statistics trap, adapter inference, checkpoint selection, cost |

02 through 05 build the base. 07 uses it.

## Verified environment

Two environments, because training and inference did not run on the same machine. The first is
the container of `docs/03-training-environment.md`, the second the workstation the hardware
evaluation in `docs/06-inference.md` ran on.

### Training

| Component | Version |
|---|---|
| GPU | 8 x NVIDIA A100 80GB SXM4, single node, NVLink |
| Host OS | Ubuntu 22.04, provider stock GPU image |
| Host driver / CUDA | not recorded |
| Base image | `nvidia/cuda:12.1.0-runtime-ubuntu22.04` |
| Python | 3.10, conda-forge |
| torch | 2.7.1+cu126 (`torch.version.cuda` 12.6) |
| torchvision | 0.22.1 |
| torchcodec | 0.5 |
| ffmpeg | 7.1.x, conda-forge |
| lerobot | 0.4.1 |
| transformers | 4.53.3 |
| accelerate | 1.13.0 |
| peft | 0.18.0 |
| bitsandbytes | 0.48.2 |
| numpy | 2.2.6 |
| safetensors | 0.7.0 |
| datasets | 4.1.1 |
| fsspec | 2025.9.0 |
| wandb | 0.21.4 |
| nvidia-nccl-cu12 | 2.26.2 |
| VLASH | commit `22cbabfee0f57874987c75a35a7dac129e695db0` |

`torch`, `torchvision` and `torchcodec` are pinned in `training-docker/environment.yml`. Without
those three pins `pip install -e /opt/vlash` resolves anything inside LeRobot 0.4.1's ranges —
`torch>=2.2.1,<2.8.0`, `torchvision>=0.21.0,<0.23.0`, `torchcodec>=0.2.1,<0.6.0` — and the build
stops being reproducible. Everything under them is resolved by that install from the pinned
VLASH commit. `peft` and `bitsandbytes` are unused in a full fine-tune; VLASH requires them for
its LoRA path.

The training host's driver and CUDA version were not recorded. The provider image shipped a
driver and `nvidia-container-toolkit`, and `docker run --gpus all` worked as delivered; the
container installs torch 2.7.1+cu126, so the driver has to be new enough for CUDA 12.6.

### Inference

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 Ti, 24 GB |
| Driver | 590.48.01 |
| Host OS | Ubuntu 24.04 LTS, kernel 6.17 |
| Python | 3.10.20 |
| torch | 2.7.1+cu126 (`torch.version.cuda` 12.6, cuDNN 9.5.1) |
| torchvision | 0.22.1 |
| torchcodec | 0.5 |
| ffmpeg | 7.1.1, conda-forge |
| lerobot | 0.4.1 |
| transformers | 4.53.3 |
| peft | 0.18.0 |
| numpy | 2.2.6 |
| pyarrow | 24.0.0 |
| safetensors | 0.7.0 |
| huggingface_hub | 0.35.3 |
| opencv-python | 4.13.0.92 |
| draccus | 0.10.0 |
| VLASH | commit `22cbabfee0f57874987c75a35a7dac129e695db0` |
| Robot | SO-101 follower, 6 DoF, Feetech STS3215, `/dev/ttyACM0` |
| Cameras | 2 x USB UVC, MJPG, 640x480 at 30 fps |

The kernel is the one on that machine now; the exact build at evaluation time was not recorded.

Other versions may work but have not been verified.

## Quick start

Download the checkpoint and confirm it is intact and is the policy documented here. No GPU, no
torch and no VLASH: `scripts/verify_checkpoint.py` reads the safetensors header and
`config.json` with the standard library alone.

```bash
pip install huggingface_hub
hf download dongyoonkim/so101-pi05-base --local-dir ckpt/so101-pi05-base
python ckpt/so101-pi05-base/scripts/verify_checkpoint.py ckpt/so101-pi05-base
```

`hf` is the current Hugging Face Hub CLI. `huggingface-cli` is deprecated and on
huggingface_hub 1.x prints a notice and exits non-zero.

```
checkpoint: ckpt/so101-pi05-base

[1/4] required files                         -- PASS
      config.json          2,292 bytes
      model.safetensors    7,481,485,688 bytes
[2/4] safetensors header                     -- PASS
      tensors              824
      parameters           3,618,890,548
      dtypes               BF16, F32
[3/4] policy config                          -- PASS
      type                 pi05
      camera slot          base_0_rgb
      camera slot          left_wrist_0_rgb
      camera slot          right_wrist_0_rgb
      state / action dim   6 / 6
      chunk size           50
      image resolution     [224, 224]
      normalization        {'VISUAL': 'IDENTITY', 'STATE': 'MEAN_STD', 'ACTION': 'MEAN_STD'}
      empty_cameras        0  (override per camera count at run time)
[4/4] header offsets vs file size            -- PASS
      declared             7,481,485,688 bytes
      on disk              7,481,485,688 bytes

4/4 checks passed
```

3,618,890,548 parameters is the 3.62B the model card states, and `empty_cameras: 0` is the field
that has to be overridden for a robot with fewer than three cameras. The script exits non-zero
when a check fails, so it can gate a download; a transfer that stopped early shows up in
check 4:

```
[4/4] header offsets vs file size            -- FAIL
      declared             7,481,485,688 bytes
      on disk              300,000,000 bytes
      truncated or partially downloaded
```

`docs/06-inference.md` covers the live run from here, and `model/README.md` covers what the
checkpoint is and is not for.

### Then what

The checkpoint on its own does one thing: it knows the SO-101. Running it as-is reproduces the
evaluation in `docs/06-inference.md` on the task that happened to be in its training corpus, and
that is the end of what it does unassisted. The point of a domain-adaptation base is the next
step — a LoRA adapter trained on demonstrations of your own task, on top of these weights, which
is a run measured in hours on one rented GPU rather than the 40 hours on eight A100s that
produced the base. [`docs/07-lora-finetuning.md`](docs/07-lora-finetuning.md) is that procedure
end to end: how much data, which GPU and why, the training config, the launch command, and the
adapter-checkpoint failure that turns every action into `nan` at inference without raising
anything.

## Contents

```
docs/                     the seven documents above
  01-overview.md            .. 06-inference.md   building and running the base
  07-lora-finetuning.md     training your own task adapter on top of it
model/                    model card, license, notice
dataset/                  dataset card, per-source attribution, license
pipeline/                 the dataset-construction scripts, one per stage
  artifacts/              the JSON instruction sheets those stages emitted
training-docker/          everything needed to build the training image
  Dockerfile
  patched/                the four patched source files
  configs/base.yaml       the training configuration
  scripts/                the wrapper scripts
  tests/                  host-side checks that need no GPU
inference/                the two files the evaluation is driven by
  v4l2_launch.py          launcher that forces the V4L2 camera backend
  eval.yaml               the robot, camera and policy configuration that was evaluated
scripts/                  release-side: repack_open_subset.py, gen_attribution.py
  verify_checkpoint.py    the checkpoint check used in Quick start
```

Most pipeline stages take their working directory as `--root` and print what they wrote, so the
stages can be run one at a time and inspected between runs. Five stages take other arguments
instead; `docs/02-dataset-construction.md` lists them under "Before you start". Inference has no separate code of its
own: `inference/` holds only the launcher and the configuration, and the policy itself runs from
the patched files in `training-docker/patched/`, as `docs/06-inference.md` describes.

## Distribution

| | |
|---|---|
| Model | [`dongyoonkim/so101-pi05-base`](https://huggingface.co/dongyoonkim/so101-pi05-base) |
| Dataset | [`dongyoonkim/so101-pi05-base-dataset`](https://huggingface.co/datasets/dongyoonkim/so101-pi05-base-dataset) |

`model/README.md` and `dataset/README.md` are the cards published with those two repositories.

## Related

Author's other released work. The converter is a direct dependency of this build; the rest is
prior work on the same robot.

| | |
|---|---|
| [jinnymo/lerobot-v3-v2-converter](https://github.com/jinnymo/lerobot-v3-v2-converter) | Two-way LeRobot v2.1 / v3.0 dataset converter. Pipeline stage 5 drives its `v3_to_v2/convert.py` to bring downloaded v3.0 sources into the v2.1 layout the training stack reads |
| [jinnymo/gr00t-n17-lora](https://github.com/jinnymo/gr00t-n17-lora) | Restores LoRA fine-tuning on NVIDIA GR00T N1.7, which upstream removed, by monkey-patching Isaac-GR00T rather than editing it |
| [`dongyoonkim/grootn17-lora-so101-eraser-tier1`](https://huggingface.co/dongyoonkim/grootn17-lora-so101-eraser-tier1) | The adapter that wrapper produced: GR00T N1.7 on one SO-101 task |
| [`dongyoonkim/so101-eraser-90ep-wrist`](https://huggingface.co/datasets/dongyoonkim/so101-eraser-90ep-wrist) | The 90-episode single-wrist-camera SO-101 dataset that adapter was trained on |

Upstream:

| | |
|---|---|
| [openpi](https://github.com/Physical-Intelligence/openpi) | Physical Intelligence's pi0.5 reference implementation. VLASH's pi0.5 is a copy of it |
| [VLASH](https://github.com/mit-han-lab/vlash) | MIT HAN Lab. The training and real-time inference stack used here |
| [LeRobot](https://github.com/huggingface/lerobot) | Hugging Face. Dataset format, SO-101 driver, camera backends, video decoding |
| [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) | The weights this fine-tune started from |

## Licenses

| Component | License |
|---|---|
| Code and documentation — `pipeline/`, `scripts/`, `training-docker/`, `inference/`, `docs/` | Apache-2.0, `LICENSE` in this directory |
| Model weights | Gemma Terms of Use, `model/LICENSE` |
| Dataset | apache-2.0 and mit, `dataset/LICENSE` and `dataset/ATTRIBUTION.md` |

`LICENSE` in this directory is the Apache-2.0 text. It covers everything in the package that
`model/LICENSE` and `dataset/LICENSE` do not cover separately.

The weights are a Gemma derivative, because pi0.5 is built on PaliGemma. They are governed by
the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), not by
Apache-2.0, and anyone who redistributes them or a derivative of them must pass those use
restrictions on. `model/NOTICE` states this in full, along with the Apache-2.0 projects the
weights were produced with and the modifications made to them.

The released dataset redistributes work by other people. `dataset/ATTRIBUTION.md` is the
attribution notice required by Section 4 of Apache-2.0 and lists every source, its license and
its episode count. Source datasets whose upstream repository declares no license were used for
training but are not redistributed.

`training-docker/patched/` contains modified copies of files from VLASH and LeRobot, both
Apache-2.0. The modifications are listed in `training-docker/NOTICE` and `model/NOTICE`, as
Section 4(b) requires.

Author: Dongyoon Kim.
