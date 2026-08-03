# 03 — Training environment

How the training image is built, what is inside it, and the four source patches that make a
multi-source, mixed-camera SO-101 dataset trainable with pi0.5.

Everything here targets one model: pi0.5 (Physical Intelligence), roughly 3.62B parameters,
fully fine-tuned on SO-101 / SO-100 single-arm data. VLM is PaliGemma (`gemma_2b` backbone +
SigLIP), the action expert is `gemma_300m`, actions are produced by flow matching with
`chunk_size` 50 and action dimension 6.

---

## 1. Why a container

The run that produced this checkpoint took about 40 hours on 8 x A100 80GB. Two properties
mattered enough to justify building an image rather than a `pip install` script.

**Reproducibility of a moving stack.** The training stack is VLASH on top of LeRobot 0.4.1,
which in turn pulls `torch`, `torchcodec`, `transformers` (pinned by VLASH to a git commit),
`peft` and `bitsandbytes`. VLASH itself is pinned to a specific commit. Four of the files in
that stack are patched. A `requirements.txt` does not capture "this exact commit of VLASH, with
these four files replaced, where one replacement must happen after the editable install
overwrites `site-packages`". An image does.

**Portability across GPU providers.** Single-node 8x A100 80GB SXM4 capacity is scarce and is
rented wherever it exists. The same image was prepared for and run on more than one provider,
with the differences confined to how the container is started (`docker run` on a bare host with
SSH versus a provider that injects an image, credentials and a start command in one call). The
container entrypoint script handles both: with an SSH public key in the environment it starts
`sshd` and sleeps, and with a command override it runs training directly in batch mode. Nothing
inside the image knows which provider it is on.

A third, smaller reason: the run needs no network access to a model hub. The base checkpoint
and the tokenizer are baked into the image and `HF_HUB_OFFLINE=1` is set, so a hub outage or a
gated-repo prompt cannot interrupt a 40-hour job.

---

## 2. The image, layer by layer

The complete Dockerfile, with the internal registry and account names replaced by
placeholders.

```dockerfile
# syntax=docker/dockerfile:1.4
# Full fine-tune image for pi0.5 on a unified 3-slot SO-101 dataset
# (lora.enable=false). NOTICE lists what the patched files change.

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Pinned vlash commit. The patched files were written against this revision.
ARG VLASH_COMMIT=22cbabfee0f57874987c75a35a7dac129e695db0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget ca-certificates \
        ffmpeg libsm6 libxext6 libgl1 \
        openssh-client openssh-server sshpass \
        unzip jq gettext-base \
        linux-libc-dev build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

# AWS CLI v2, for S3 datasets and checkpoints
RUN curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp/ \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscli.zip

# Miniforge, not the Anaconda installer: conda-forge only, no Anaconda ToS.
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/conda.sh \
    && bash /tmp/conda.sh -b -p $CONDA_DIR \
    && rm /tmp/conda.sh \
    && $CONDA_DIR/bin/conda init bash
ENV PATH="$CONDA_DIR/bin:$PATH"

RUN conda config --remove channels defaults 2>/dev/null || true \
    && conda config --set channel_priority strict

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy

RUN git clone https://github.com/mit-han-lab/vlash.git /opt/vlash \
    && cd /opt/vlash \
    && git checkout $VLASH_COMMIT

COPY patched/train.py /opt/vlash/vlash/train.py
COPY patched/run.py /opt/vlash/vlash/run.py
COPY patched/modeling_pi05.py /opt/vlash/vlash/policies/pi05/modeling_pi05.py

RUN conda run -n vlash pip install -e /opt/vlash

# lerobot patch: bounded video decoder cache and a decode fallback.
# lerobot arrives as a transitive dependency of vlash, so this overwrite has to
# happen after `pip install -e`. Cache size: VLASH_DECODER_CACHE_MAX (default 64).
COPY patched/video_utils.py /opt/conda/envs/vlash/lib/python3.10/site-packages/lerobot/datasets/video_utils.py

# Base weights are baked in, so a long run needs no hub access.
# local_dir_use_symlinks is not passed: it was deprecated in huggingface_hub 0.23
# and removed in 1.0. Since 0.23 a local_dir download already writes real files.
RUN mkdir -p /opt/models/pi05_base \
    && conda run -n vlash python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('lerobot/pi05_base', local_dir='/opt/models/pi05_base')"

# The PaliGemma tokenizer is not vendored in this package: google/paligemma-3b-pt-224
# is a gated repository. Run scripts/fetch-tokenizer.sh once, with a Hugging Face
# account that has accepted the Gemma terms and been granted access, before building.
COPY paligemma_tokenizer_flat /opt/models/paligemma_tokenizer

COPY configs/base.yaml /opt/configs/base.yaml
COPY scripts/train-base /opt/scripts/train-base
COPY scripts/prepare-dataset /opt/scripts/prepare-dataset
COPY scripts/inject-stats /opt/scripts/inject-stats
COPY scripts/container-init.sh /opt/scripts/container-init.sh
COPY scripts/s3_checkpoint.py /opt/scripts/s3_checkpoint.py

RUN chmod +x /opt/scripts/*
ENV PATH="/opt/scripts:/opt/conda/envs/vlash/bin:$PATH"

# Applies PATH and the cache environment to SSH login shells as well
RUN cat > /etc/profile.d/train-env.sh <<'PROFILE_EOF' \
 && chmod +x /etc/profile.d/train-env.sh
export PATH="/opt/scripts:/opt/conda/envs/vlash/bin:/opt/conda/bin:${PATH}"
export HOME=${HOME:-/workspace}
export XDG_CACHE_HOME=/workspace/.cache
export TRITON_CACHE_DIR=/workspace/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torch_inductor
export TORCH_HOME=/workspace/.cache/torch
export TRANSFORMERS_CACHE=/workspace/.cache/transformers
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export HF_HOME=/opt/models/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLASH_PALIGEMMA_PATH=/opt/models/paligemma_tokenizer
PROFILE_EOF

ENV HOME=/workspace
ENV XDG_CACHE_HOME=/workspace/.cache
ENV TRITON_CACHE_DIR=/workspace/.cache/triton
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torch_inductor
ENV TORCH_HOME=/workspace/.cache/torch
ENV TRANSFORMERS_CACHE=/workspace/.cache/transformers
ENV MPLCONFIGDIR=/workspace/.cache/matplotlib

ENV HF_HOME=/opt/models/hf_cache
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV VLASH_PALIGEMMA_PATH=/opt/models/paligemma_tokenizer

RUN mkdir -p /workspace/checkpoints /workspace/.cache /workspace/logs \
    && chmod -R 777 /workspace /tmp \
    && chmod -R 777 /root/.cache /root/.config 2>/dev/null || true

WORKDIR /workspace

# Default CMD is the interactive SSH init. For batch runs, override it:
#   docker run <image> /opt/scripts/train-base --dataset-url=... --steps=...
ENTRYPOINT []
CMD ["/opt/scripts/container-init.sh"]
```

### 2.1 Base image

`nvidia/cuda:12.1.0-runtime-ubuntu22.04`.

The `runtime` variant, not `devel`: nothing in this image compiles CUDA kernels at build time.
PyTorch arrives as a pip wheel with its kernels precompiled, and the CUDA libraries it needs
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-nccl-cu12` and the rest) come as separate
pip wheels pulled in by that wheel. This is also why the CUDA version of the base image does
not have to match the CUDA version PyTorch was built against: the installed torch is
`2.7.1+cu126` and reports `torch.version.cuda == 12.6`, while the base image is 12.1. The
bundled `nvidia-*-cu12` wheels (12.6.x) are what actually gets loaded; the host driver has to
be new enough for 12.6, which is the real constraint. `build-essential` and `cmake` are present
because a few LeRobot dependencies build native extensions from source at install time.

Why specifically 12.1 rather than a 12.6 base image is not recorded. If you are rebuilding
from scratch, a base image matching the torch wheel's CUDA version is the more obvious choice
and there is nothing in the recipe that depends on 12.1.

### 2.2 System packages

- `ffmpeg`, `libsm6`, `libxext6`, `libgl1` — video decoding and the shared libraries OpenCV
  expects. `torchcodec` needs an ffmpeg 7 ABI (`libavutil.so.59`), which is why `ffmpeg` is
  also pinned in the conda environment; see section 3.
- `openssh-server`, `openssh-client`, `sshpass` — interactive mode. Providers that hand you a
  container rather than a host expect the container to run an SSH daemon. Password
  authentication is disabled; only a public key injected through the environment is accepted.
- `gettext-base` — provides `envsubst`, which the training wrapper uses to render the config
  template. Easy to miss; without it the wrapper fails at config generation.
- `jq`, `unzip`, `curl`, `wget`, `git` — scripting and installers.
- AWS CLI v2 is installed from the official zip rather than pip, so it does not share a
  dependency graph with the training environment.

### 2.3 conda environment

The environment is created before the VLASH checkout so that Docker's layer cache keeps the
slowest step (solving and installing the environment) intact while patches are iterated on.

`conda config --remove channels defaults` plus `channel_priority strict` keeps everything on
conda-forge. Miniforge is used instead of Miniconda for the same reason.

### 2.4 VLASH checkout and commit pin

```dockerfile
ARG VLASH_COMMIT=22cbabfee0f57874987c75a35a7dac129e695db0
RUN git clone https://github.com/mit-han-lab/vlash.git /opt/vlash \
    && cd /opt/vlash && git checkout $VLASH_COMMIT
```

The pin is load-bearing. All four patches are whole-file replacements rather than `patch` hunks,
and three of them — `train.py`, `run.py` and `modeling_pi05.py` — are copied over this checkout
in full. A different upstream commit means those copies silently revert whatever upstream changed
in those files. If you move the pin forward, re-derive the patches as diffs against the new tree
instead of reusing the files. The fourth, `video_utils.py`, is pinned by the LeRobot version
instead (0.4.1, §2.3) and has the same exposure.

### 2.5 Patch copy order

This is the one ordering constraint in the Dockerfile, and getting it wrong produces a build
that looks fine and then leaks memory for 40 hours.

```dockerfile
COPY patched/train.py           /opt/vlash/vlash/train.py
COPY patched/run.py             /opt/vlash/vlash/run.py
COPY patched/modeling_pi05.py   /opt/vlash/vlash/policies/pi05/modeling_pi05.py

RUN conda run -n vlash pip install -e /opt/vlash          # <-- editable install

COPY patched/video_utils.py \
     /opt/conda/envs/vlash/lib/python3.10/site-packages/lerobot/datasets/video_utils.py
```

The first three patches target files inside `/opt/vlash`, which is installed **editable**
(`pip install -e`). An editable install leaves the source tree in place and imports from it, so
those three can be copied before or after the install; they are copied before so that a patch
change invalidates the install layer and the environment is re-linked.

The fourth patch targets `lerobot/datasets/video_utils.py`, which lives in **site-packages**.
LeRobot is a transitive dependency: VLASH's `pyproject.toml` requires
`lerobot[feetech,smolvla]==0.4.1`, so pip downloads and installs LeRobot as part of
`pip install -e /opt/vlash`. Before that step the target path does not exist, and `COPY` to a
non-existent parent directory creates the directory tree — producing a stray
`site-packages/lerobot/datasets/video_utils.py` under a directory that pip then overwrites
wholesale when it installs the real package. The patch disappears, the build succeeds, and the
decoder cache leak (section 4.3) comes back.

### 2.6 Baking in the model and the tokenizer

```dockerfile
RUN mkdir -p /opt/models/pi05_base \
    && conda run -n vlash python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('lerobot/pi05_base', local_dir='/opt/models/pi05_base', local_dir_use_symlinks=False)"

COPY paligemma_tokenizer_flat /opt/models/paligemma_tokenizer
```

Two separate problems, one goal: at run time the container needs no Hugging Face Hub access.

`lerobot/pi05_base` (about 14 GB) is snapshotted at build time into `/opt/models/pi05_base`,
which the training config points at via `policy.pretrained_path`.

The tokenizer is the more annoying half. Upstream `modeling_pi05.py` loads it by hub id:

```python
self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
```

`google/paligemma-3b-pt-224` is a gated repository — it requires accepting terms and an
authenticated token — and with `HF_HUB_OFFLINE=1` the call fails outright unless the files are
already in the hub cache in exactly the layout the cache expects. The patch replaces the hub id
with a filesystem path taken from an environment variable:

```python
_paligemma_path = os.environ.get(
    "VLASH_PALIGEMMA_PATH",
    "/opt/models/paligemma_tokenizer",
)
self.language_tokenizer = AutoTokenizer.from_pretrained(_paligemma_path)
```

`paligemma_tokenizer_flat/` in the build context is a flat directory holding only the tokenizer
side of that repository, no weights: `tokenizer.json` (17.5 MB), `tokenizer.model` (4.3 MB),
`tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`,
`preprocessor_config.json`, `config.json`, `generation_config.json`, and
`model.safetensors.index.json`. About 21 MB in total.

`scripts/fetch-tokenizer.sh` produces that directory, and is the reason the list above does not
have to be transcribed by hand — the nine file names are its `allow_patterns`:

```bash
pip install huggingface_hub          # if it is not already available
hf auth login                        # or: export HF_TOKEN=<read token>
cd training-docker
./scripts/fetch-tokenizer.sh         # writes ./paligemma_tokenizer_flat
```

It takes an optional destination argument (default `./paligemma_tokenizer_flat`) and reads
`PALIGEMMA_REPO_ID` if a mirror is being used. It runs on the build host, before
`docker build`, and it will fail until the account it authenticates as has been granted access
to `google/paligemma-3b-pt-224`: accepting the Gemma terms on the model page is a manual step
with a wait, so do it before the day of the build. The tokenizer files are redistributed by
Google under the Gemma terms rather than this project's license, which is why they are fetched
instead of vendored.

`HF_HOME`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set as `ENV` so that any
remaining hub lookup fails loudly at import rather than hanging on a network call at step
30,000.

### 2.7 Environment variables, twice

Every variable is declared both as a Docker `ENV` and inside `/etc/profile.d/train-env.sh`.
This is not redundancy. `ENV` applies to the process started by `docker run`; an SSH login shell
started later by `sshd` inside the container gets a fresh environment from `/etc/profile`, and
without the profile snippet an operator who SSHes in to run a command by hand gets a different
`PATH`, no conda environment, and no `VLASH_PALIGEMMA_PATH`.

All caches are redirected under `/workspace/.cache` (Triton, TorchInductor, torch hub,
transformers, matplotlib) so a single mounted volume covers them. `HOME=/workspace` for the
same reason.

`/workspace` is created and chmod-ed at build time **and again** at container start by
`container-init.sh`, because a volume mounted over `/workspace` hides whatever the image
created there.

### 2.8 What is excluded from the build context

`.dockerignore`:

```
.git
.gitignore
*.md
*.log
tests
__pycache__
.pytest_cache
```

The tests are deliberately not in the image; they run on the build host (section 8).
`__pycache__` and `.pytest_cache` are there so that running them locally does not invalidate
the build cache. `*.md` covers `README.md`; there is no separate `README.md` line.

---

## 3. Environment specification

`environment.yml`:

```yaml
name: vlash
channels:
  - conda-forge

dependencies:
  - python=3.10
  - pip
  - cmake             # native builds for some LeRobot dependencies
  - ffmpeg=7.1.*      # torchcodec needs the ffmpeg 7 ABI (libavutil.so.59)
  - pip:
      - huggingface_hub
      - gdown
```

It is short on purpose. Only three things are pinned here:

- **`python=3.10`** — the floor VLASH declares (`requires-python = ">=3.10"`). It also fixes the
  site-packages path used by the `video_utils.py` COPY, which is why changing the Python
  version means changing that path in the Dockerfile.
- **`ffmpeg=7.1.*` from conda-forge** — `torchcodec` links against `libavutil.so.59`, which is
  ffmpeg 7. The apt `ffmpeg` on Ubuntu 22.04 is older; the conda-forge build is what
  `torchcodec` actually loads because the conda environment's lib directory comes first.
- **`huggingface_hub` and `gdown`** — used by the wrapper scripts to fetch datasets and by the
  Dockerfile to snapshot the base model. They are not VLASH dependencies.

Everything else, including PyTorch, is resolved by `pip install -e /opt/vlash` from VLASH's
`pyproject.toml`, which requires `lerobot[feetech,smolvla]==0.4.1`, a pinned
`transformers` git commit, `peft==0.18.0` and `bitsandbytes==0.48.2`. LeRobot in turn requires
`torch`, `torchvision` and `torchcodec`, which is where those come from.

Resolved versions in the environment that produced this checkpoint:

| Package | Version |
|---|---|
| Python | 3.10 |
| torch | 2.7.1+cu126 (`torch.version.cuda` 12.6) |
| torchvision | 0.22.1 |
| torchcodec | 0.5 |
| lerobot | 0.4.1 |
| transformers | 4.53.3 (from VLASH's pinned git commit `dcddb970176382c0fcf4521b0c0e6fc15894dfe0`) |
| accelerate | 1.13.0 |
| peft | 0.18.0 |
| bitsandbytes | 0.48.2 |
| datasets | 4.1.1 |
| numpy | 2.2.6 |
| safetensors | 0.7.0 |
| fsspec | 2025.9.0 |
| wandb | 0.21.4 |
| ffmpeg | 7.1.x (conda-forge) |
| nvidia-nccl-cu12 | 2.26.2 |
| nvidia-cudnn-cu12 | 9.5.1.17 |
| nvidia-cublas-cu12 | 12.6.4.1 |
| VLASH | commit `22cbabfee0f57874987c75a35a7dac129e695db0` |
| Base image | `nvidia/cuda:12.1.0-runtime-ubuntu22.04` |

`peft` and `bitsandbytes` are unused in a full fine-tune (no LoRA, no 8-bit optimizer); they
are pulled in because VLASH requires them for its LoRA path.

---

## 4. The four patches

Four files are replaced. Three of them are VLASH; the fourth is LeRobot. They ship in
`training-docker/patched/` and the Dockerfile copies them over the installed tree.

| File | Package | What it is for |
|---|---|---|
| `train.py` | VLASH | 224 resize-with-pad before collate, so sources of different resolutions can be stacked |
| `modeling_pi05.py` | VLASH | per-sample camera masking, plus keeping the mask columns out of normalization |
| `video_utils.py` | LeRobot | a bounded decoder cache — a memory fix, and the reason a 40-hour run finishes |
| `run.py` | VLASH | inference-side guards. Not used in training at all; it is in the image only to keep one patched tree |

So the multi-source, variable-camera problem is solved by exactly two of them — `train.py` and
`modeling_pi05.py`. Those two are the ones that are no-ops on a conventional single-source
dataset: both key off the presence of `_mask` columns in the dataset features, and a dataset
without them takes the original code path unchanged. `video_utils.py` applies to any dataset,
and `run.py` never runs during training.

### 4.1 `train.py` — resize before collate, gated on mask columns

**What upstream does.** `make_vlash_dataset` builds the image transform from the config's
augmentation settings and passes it to `LeRobotDataset`:

```python
image_transforms = (
    ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
)
```

LeRobot applies that callable per sample, per camera key, in `__getitem__`, after video frames
are decoded and before the sample reaches the collate function.

**The problem.** The source datasets have different video resolutions — 640x480 and 1920x1080
are both common, and there are others. The default collate function stacks per-sample tensors
into a batch tensor, and tensors of different `H x W` cannot be stacked. Training on the merged
corpus fails at the first batch that mixes two resolutions.

The model does resize internally: `prepare_images` calls `resize_with_pad(img, 224, 224)`. But
that happens inside `forward`, on an already-collated batch, which is too late.

The alternatives were re-encoding 51,411 videos to a common resolution (hours of transcoding and
a second copy of a 198 GB dataset), or normalizing at load time. Load time won.

**The patch.**

```diff
+def _resize_with_pad(img, size=224):
+    """Letterbox a camera frame to size x size, preserving aspect ratio.
+
+    Defined at module level so it stays picklable for DataLoader workers
+    under both fork and spawn start methods.
+    """
+    import torch.nn.functional as F
+    h, w = img.shape[-2:]
+    r = size / max(h, w)
+    nh, nw = max(1, round(h * r)), max(1, round(w * r))
+    img = F.interpolate(img.unsqueeze(0).float(), (nh, nw),
+                        mode="bilinear", align_corners=False).squeeze(0)
+    return F.pad(img, (0, size - nw, 0, size - nh), value=0.0)
+
+
+def _resize_with_pad_then_aug(aug, img):
+    """Letterbox, then apply the existing augmentation (module level, picklable)."""
+    return aug(_resize_with_pad(img))
```

and in `make_vlash_dataset`:

```diff
-    image_transforms = (
+    _aug = (
         ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
     )
 
     ds_meta = LeRobotDatasetMetadata(
         cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
     )
 
+    # Only inject the resize for datasets that carry per-slot mask columns.
+    # Single-source datasets have uniform resolution and keep the original behaviour.
+    _is_unified = any(str(k).endswith("_mask") for k in ds_meta.features)
+    if _is_unified:
+        image_transforms = (
+            _resize_with_pad
+            if _aug is None
+            else functools.partial(_resize_with_pad_then_aug, _aug)
+        )
+    else:
+        image_transforms = _aug
```

**Why this shape.**

- *Gated on `_mask`.* The detection key is the same one used by the masking patch, so one
  dataset property switches both behaviours on together. A single-source dataset sees exactly
  upstream behaviour.
- *Module level, not a lambda or a closure.* The transform is called inside DataLoader worker
  processes. Under `spawn` the transform must pickle, and a lambda or a locally defined closure
  does not. The composed case uses `functools.partial` for the same reason. A crash at the first
  `next(dl_iter)` is the recorded symptom of getting this wrong, which is why the functions were
  lifted to module level and why there is a unit test that pickles them (section 8).
- *No re-encoding.* The video files are never touched.
- *Interaction with the model's own resize.* Because the transform already emits exactly
  224x224, `resize_with_pad` inside `prepare_images` computes a ratio of 1 and pads nothing, so
  the image is not resized twice. Note the two implementations pad on different sides: the
  training transform pads right and bottom, and the model's `resize_with_pad` pads left and top.
  For a non-square camera image fed directly to the model at inference time, the padding band
  therefore sits on the opposite side from where it sat during training. The hardware
  evaluation used 640x480 cameras and succeeded; this is recorded as an observation about the
  two code paths, not as a measured effect.

### 4.2 `modeling_pi05.py` — per-episode camera masking

This is the mechanism that makes a fixed-three-slot model train on a corpus where episodes have
one, two or three real cameras.

The file carries **three** modifications, of which two are the masking mechanism and are covered
here:

| # | Where | Change |
|---|---|---|
| 1 | `PI05Policy.__init__` | drop `_mask` keys from `config.input_features` before the normalization modules are built |
| 2 | `PI05Policy.__init__` | load the PaliGemma tokenizer from a local path instead of a hub id — unrelated to masking, covered in §2.6 |
| 3 | `prepare_images` | read the per-sample validity mask from the batch instead of assuming every sample is valid |

Modifications 1 and 3 are two halves of one mechanism and have to be read together: the first
stops the flag from being rescaled, the second is what consumes it. Modification 2 is only in
this file because the tokenizer happens to be constructed here.

**Background: how pi0.5 handles slots.** pi0.5 takes exactly three image slots, following the
openpi convention:

```
observation.images.base_0_rgb          external / overhead
observation.images.left_wrist_0_rgb    wrist
observation.images.right_wrist_0_rgb   second wrist
```

Upstream already has a notion of an absent camera. In `prepare_images`, any slot the batch does
not contain is filled with a constant `-1` image (black in SigLIP's `[-1, 1]` range) and a mask
of zeros, up to `config.empty_cameras`:

```python
for num_empty_cameras in range(len(missing_img_keys)):
    if num_empty_cameras >= self.config.empty_cameras:
        break
    img = torch.ones_like(img) * -1
    mask = torch.zeros_like(mask)
    images.append(img)
    img_masks.append(mask)
```

The mask reaches attention through the prefix embedder, where it becomes the padding mask for
that slot's image tokens:

```python
for img, img_mask in zip(images, img_masks, strict=True):
    img_emb = self.img_embedder(img)
    bsz, num_img_embs = img_emb.shape[:2]
    embs.append(img_emb)
    pad_masks.append(img_mask[:, None].expand(bsz, num_img_embs))
```

A slot with `mask=False` has all of its image token positions marked as padding, so they
contribute no attention keys. The black placeholder is embedded but cannot be attended to.

**The problem.** That mechanism is *per configuration*, not *per sample*. `empty_cameras` is a
config integer, and upstream builds the mask as `torch.ones(bsize, dtype=torch.bool)` for every
present key — every sample in the batch is assumed to have the same cameras. In the merged
corpus, camera presence varies episode by episode: 1,314 episodes have one real camera, 14,098
have two, 1,725 have three. All three slots are physically present in every parquet row, with
the unused ones holding a black video, and a `float32` column per slot says which is which:

```
observation.images.base_0_rgb_mask          1.0 or 0.0
observation.images.left_wrist_0_rgb_mask    1.0 or 0.0
observation.images.right_wrist_0_rgb_mask   1.0 or 0.0
```

Without a patch, the model treats those black frames as real observations and trains on them.

There is a second, less obvious failure. Those mask columns are `float32` non-image features, so
LeRobot's feature typing classifies them as **state** features and puts them in
`config.input_features`. pi0.5 normalizes state features (`STATE: MEAN_STD`), so the
normalization module would z-score the 0/1 flags into whatever the corpus statistics say,
destroying them before `prepare_images` ever sees them. It would also add three spurious
entries to the policy's input feature set.

**Patch part 1 — drop mask columns from `input_features`,** in `PI05Policy.__init__`, before
the normalization modules are constructed:

```diff
         config.validate_features()
         self.config = config
 
+        # observation.images.{slot}_mask is a 0/1 camera-validity flag. It is neither a
+        # normalization target nor a model input slot: prepare_images reads it straight
+        # from the batch. Excluding it here keeps normalization from rescaling the flag,
+        # while the column itself stays in the batch dict.
+        config.input_features = {
+            k: v for k, v in config.input_features.items() if not k.endswith("_mask")
+        }
+
         # Setup normalization modules
```

The placement matters: after `config.validate_features()` (which is what adds placeholder
cameras and ensures state/action features exist) and before the `Normalize` / `Unnormalize`
modules are built from `config.input_features`. Removing the keys from `input_features` does not
remove the columns from the batch — the dataset still loads them, they are still collated, and
`prepare_images` still finds them.

**Patch part 2 — read the mask per sample,** in `prepare_images`:

```diff
             bsize = img.shape[0]
             device = img.device
-            mask = torch.ones(bsize, dtype=torch.bool, device=device)
+            # If a {key}_mask column exists, use it as the per-sample validity mask, so a
+            # black placeholder camera contributes no attention keys. Otherwise fall back
+            # to the original behaviour: every sample valid.
+            mkey = f"{key}_mask"
+            if mkey in batch:
+                mask = batch[mkey].to(device=device, dtype=torch.bool).reshape(bsize)
+            else:
+                mask = torch.ones(bsize, dtype=torch.bool, device=device)
             images.append(img)
             img_masks.append(mask)
```

**Why this shape.**

- *It reuses the upstream masking path rather than adding one.* The only thing that changes is
  where the boolean comes from. Downstream (prefix embedder, attention, flow matching) is
  untouched, so there is no new interaction to reason about.
- *Per sample, not per batch.* `mask` becomes a `[B]` boolean instead of a `[B]` of ones, so a
  single batch can mix a three-camera episode and a one-camera episode. That is required: the
  sampler draws frames uniformly from the whole corpus and does not group by camera count.
- *`prepare_images` is the single choke point.* Both `forward` and the shared-observation
  variant go through it, so one change covers training and inference.
- *`.reshape(bsize)`* — the column arrives as `[B, 1]` from a `(1,)`-shaped feature; the mask
  must be `[B]` to broadcast correctly in `img_mask[:, None].expand(bsz, num_img_embs)`.
- *The `else` branch is the compatibility guarantee.* A dataset without mask columns behaves
  exactly as upstream.

An important consequence for how this model can be used: because masking is per sample and
camera count carries no weight anywhere in sampling, loss or normalization, a one-camera episode
and a three-camera episode contribute equally. The trained model runs unchanged on 1, 2 or 3
cameras at inference, using upstream's `empty_cameras` to fill the unused slots the same way.

**Verification.** A forward smoke test on a single 24 GB card, batch 2, bf16, `no_grad`, with
two episodes chosen to have different camera sets:

```
policy OK — 3.62B params, residual _mask entries in input_features = []
batch:
    base_0_rgb_mask:        [1.0, 0.0]
    left_wrist_0_rgb_mask:  [1.0, 1.0]
    right_wrist_0_rgb_mask: [1.0, 0.0]
forward OK — loss = 0.0190, finite = True
```

Both halves are visible in that output: no `_mask` keys survive in `input_features`, and two
samples in one batch carry different masks.

### 4.3 `video_utils.py` — bounded decoder cache

This one is in LeRobot, not VLASH, and it is the reason a 40-hour run finishes instead of being
killed.

**What upstream does.** `lerobot/datasets/video_utils.py` keeps a module-level decoder cache so
that repeated reads of the same video do not re-initialize a decoder:

```python
_default_decoder_cache = VideoDecoderCache()

class VideoDecoderCache:
    def __init__(self):
        self._cache: dict[str, tuple[Any, Any]] = {}
        self._lock = Lock()

    def get_decoder(self, video_path: str):
        with self._lock:
            if video_path not in self._cache:
                file_handle = fsspec.open(video_path).__enter__()
                decoder = VideoDecoder(file_handle, seek_mode="approximate")
                self._cache[video_path] = (decoder, file_handle)
            return self._cache[video_path][0]
```

`decode_video_frames_torchcodec` uses this cache by default.

**The problem.** The dict has no bound and nothing evicts from it. On a dataset of 51,411
distinct videos read in shuffled order, essentially every sample touches a video that is not in
the cache yet, so a `VideoDecoder` (with `seek_mode="approximate"`, holding an index and
buffers) plus an open file handle accumulates per video path, forever.

Observed on the full run: cgroup `anon` memory grew about **1.6 GB per step**, monotonically,
and the process was heading for an OOM kill around step 1000 (about 85 minutes in). Two things
made this hard to diagnose and are worth repeating:

- Total RSS is misleading. The signal is cgroup `memory.stat`: `anon` (unreclaimable) growing
  monotonically while `file` (page cache, reclaimable) stays flat. With no swap, `anon` cannot
  be reclaimed, so the growth is a real leak rather than caching.
- `num_workers=0` does not fix it. It stops worker copy-on-write growth but the leak simply
  moves to the main process, and losing prefetch idles the GPU (measured `data_s` 1.05 s, about
  1.5x slower overall). It is a symptom shift, not a fix.

This is upstream LeRobot behaviour, tracked in issues #2371 and #3712.

**The patch.** Turn the dict into an LRU with a bound, and close the file handle on eviction:

```diff
     def __init__(self):
-        self._cache: dict[str, tuple[Any, Any]] = {}
+        import collections, os
+        self._cache = collections.OrderedDict()
+        self._maxsize = int(os.environ.get("VLASH_DECODER_CACHE_MAX", "64"))
         self._lock = Lock()
 
     def get_decoder(self, video_path: str):
         ...
                 file_handle = fsspec.open(video_path).__enter__()
                 decoder = VideoDecoder(file_handle, seek_mode="approximate")
                 self._cache[video_path] = (decoder, file_handle)
-
+                self._cache.move_to_end(video_path)
+                while len(self._cache) > self._maxsize:
+                    _k, (_od, _ofh) = self._cache.popitem(last=False)
+                    try:
+                        _ofh.close()
+                    except Exception:
+                        pass
+            else:
+                self._cache.move_to_end(video_path)
             return self._cache[video_path][0]
```

**Why this shape.**

- *Low risk by construction.* The cache is a performance optimization. Evicting a decoder and
  recreating it later returns the same frames, so training data and results are unaffected. That
  property is what made it acceptable to change mid-run.
- *Closing the file handle on eviction* is not optional. Dropping the decoder alone would leave
  the `fsspec` handle open and convert a memory leak into a file-descriptor leak — which this
  run also hit from a different direction (see below).
- *Bound is an environment variable* (`VLASH_DECODER_CACHE_MAX`, default 64) rather than a
  constant, so it can be tuned without a rebuild: raise it if `data_s` rises from eviction
  churn, lower it if memory is tight.
- *Both branches call `move_to_end`* so the ordering reflects recency on hits as well as misses.

**Result, measured on the same run:**

| Metric | Before | After |
|---|---|---|
| `anon` growth | ~1.6 GB/step, OOM around step 1000 | ~0.015 GB/step, plateau near 96 GB |
| `data_s` | 1.05 with `num_workers=0` | 0.002 with `num_workers=8` |
| `updt_s` | 3.6 s | 3.55 s |
| Outcome | could not finish | ~40 h, completed |

A second, unrelated change ships in the same file: a fallback for videos with frames that fail
to decode.

```diff
-    frames_batch = decoder.get_frames_at(indices=frame_indices)
+    try:
+        frames_batch = decoder.get_frames_at(indices=frame_indices)
+    except Exception as _vlash_e:
+        logging.warning(f"[corrupt-video] {video_path} idx={frame_indices}: {_vlash_e} -> fallback frame0")
+        import torch as _vt
+        _fb0 = decoder.get_frames_at(indices=[0])
+        return _vt.stack([_fb0.data[0] for _ in timestamps])
```

A handful of source videos in a 51,411-video corpus have undecodable frames. Substituting
frame 0 corrupts one sample; letting the exception propagate ends a 40-hour run. The warning
line makes the substitution visible in the log rather than silent.

**Related runtime setting, not a patch.** With 8 workers per process across 8 DDP ranks over
tens of thousands of videos, the default `nofile` limit of 1024 is exceeded; DataLoader workers
die and the run deadlocks in an NCCL all-reduce (observed at step 70, with GPUs showing 100%
utilization at idle power). Start the container with
`--ulimit nofile=1048576:1048576`. Also give it `--shm-size` of at least 16 GB (64 GB was used)
or the workers hit `Bus error` on the default 64 MB `/dev/shm`.

### 4.4 `run.py` — inference-side guards

`run.py` is the closed-loop inference entry point. It is patched and copied into the training
image only so that one patched tree covers both roles; a full fine-tune never executes it. It
is described here because it ships in the image.

Note that the copy in the training image is an earlier snapshot than the one used for the
hardware evaluation. In particular it still has upstream's exact-match camera check
(`validate_robot_cameras` requires `robot.cameras` to equal the checkpoint's image features by
name), which rejects a two-camera robot against a three-slot checkpoint. That check was later
relaxed to accept a subset. If you are reproducing inference rather than training, take the
inference-side patches from 06 §3, which carries their diffs, not from this image.

Three changes.

**Chunk-boundary blending.** VLASH runs asynchronous inference: the next action chunk is
computed while the current one is still executing. At the switch, upstream replaces the current
chunk outright, which produces a discontinuity between the last action of the old chunk and the
first action of the new one. The patch ramps between them over the overlap window.

**Dry run and a joint range guard.** A policy that diverges drives the arm into its own
hardware. The patch prints actions periodically and aborts before sending anything whose
magnitude exceeds `VLASH_SAFE_DEG`. `VLASH_DRY_RUN=1` runs the whole loop, including the policy,
without commanding the arm — the first thing to run against a new checkpoint.

**Adapter-only checkpoint loading.** If `policy.pretrained_path` contains a `lora_adapters/`
directory, the patch loads the base policy, applies the LoRA structure using hyperparameters
read back from the saved `train_config.json`, loads the adapter weights, merges them, and then
restores normalization statistics from a `normalize_buffers.pt` saved alongside. This exists
because an adapter-only checkpoint does not carry dataset statistics, and an uninitialized
normalization buffer produces NaN at the first inference. It does not apply to full fine-tune
checkpoints, which save the whole model and its statistics through the normal path.

### 4.5 What is not patched

- The **LoRA save branch** in `train.py` is modified (adapter-only saving, a checkpoint upload
  hook and a W&B run-id pin for resumable spot training), but with `lora.enable: false` none of
  it executes. The full fine-tune path calls upstream `save_checkpoint` unchanged.
- The **model itself** is untouched apart from the two `modeling_pi05.py` edits above. The
  pi0.5 implementation in VLASH is a copy of the openpi reference; no architecture, loss or
  flow-matching change was made.
- **Normalization, sampling and the loss** are upstream. Camera count does not appear in any of
  them.

---

## 5. `base.yaml`, field by field

The training config is a template. The wrapper substitutes `${...}` placeholders with
`envsubst` and passes the rendered file to `vlash train`.

```yaml
policy:
  type: pi05
  pretrained_path: /opt/models/pi05_base
  push_to_hub: false
  dtype: bfloat16
  device: cuda
  state_cond: true
  compile_model: false
  fuse_qkv: false
  fuse_gate_up: false
  gradient_checkpointing: false

dataset:
  repo_id: ${DATASET_NAME}
  root: ${DATASET_PATH}
  video_backend: torchcodec

job_name: ${DATASET_NAME}_base
output_dir: ${OUTPUT_DIR}
batch_size: ${BATCH_SIZE}
grad_accum_steps: ${GRAD_ACCUM}
steps: ${STEPS}
num_workers: 8
seed: 1000

use_policy_training_preset: false
optimizer:
  type: adamw
  lr: ${PEAK_LR}
  betas: [0.9, 0.95]
  weight_decay: 1.0e-10
  grad_clip_norm: 1.0

scheduler:
  type: cosine_decay_with_warmup
  num_warmup_steps: ${WARMUP_STEPS}
  peak_lr: ${PEAK_LR}
  decay_lr: 2.5e-6
  num_decay_steps: ${STEPS}

save_checkpoint: true
save_freq: ${SAVE_FREQ}
log_freq: 10

wandb:
  enable: ${WANDB_ENABLE}
  project: ${WANDB_PROJECT}
  disable_artifact: true

max_delay_steps: 8

lora:
  enable: false
```

**policy**

| Field | Meaning |
|---|---|
| `type: pi05` | Selects VLASH's `PI05Policy` / `PI05Config`, not LeRobot's pi0 implementation. |
| `pretrained_path` | Where the base checkpoint was baked into the image. Overridable on the command line for a warm restart. |
| `push_to_hub: false` | LeRobot's config validation rejects a run with `push_to_hub` enabled and no `repo_id`. Nothing is pushed. |
| `dtype: bfloat16` | Compute dtype. The alternative in this config is `float32`. |
| `state_cond: true` | Robot state enters through adaRMS conditioning on the action expert instead of being discretized into the text prompt. With `false`, `prepare_language` bucketizes the state into 256 bins and writes it into the prompt string. |
| `compile_model: false` | `torch.compile` is an inference optimization here; leaving it off avoids a long warmup and recompiles at the start of a 40-hour run. |
| `fuse_qkv`, `fuse_gate_up` | Inference-time projection fusion. Off for training. |
| `gradient_checkpointing: false` | Declared in `PI05Config` but never referenced anywhere in the VLASH model code, so it is a no-op in this version regardless of value. Memory was verified to fit at per-GPU batch 8 on 80 GB without it. |

**dataset**

| Field | Meaning |
|---|---|
| `repo_id` | Identifier only; the data is read from `root`. The wrapper derives it from the dataset directory name. |
| `root` | Local path to the LeRobot dataset, produced by `prepare-dataset`. |
| `video_backend: torchcodec` | Frame decoding backend. This is the path that goes through the patched `video_utils.py`. |

Note what is *not* here: no image resize and no camera list. Both are derived at run time from
the dataset's own features by the `train.py` patch.

**run and schedule**

| Field | Value used | Meaning |
|---|---|---|
| `batch_size` | 8 | **Per GPU**, not global. |
| `grad_accum_steps` | 4 | Effective batch = 8 x 8 GPUs x 4 = **256**. |
| `steps` | 40000 | Optimizer steps. 40000 x 256 = 10.24M samples, which is 1.19 epochs over the 8,595,621 frames the run was actually fed. Against the full 8,690,531-frame merge it is 1.178; the run log shows 1.19 because the 450 excluded episodes were already gone. |
| `num_workers` | 8 | DataLoader workers per process. With 8 ranks this is 64 worker processes, which is what makes the `nofile` limit matter. |
| `seed` | 1000 | Seeded through `set_seed` with the accelerator. |
| `use_policy_training_preset` | false | Use the explicit `optimizer` and `scheduler` blocks instead of the defaults carried in `PI05Config`. This is why the `config.json` written into the checkpoint still shows the policy-level defaults (`optimizer_lr: 2.5e-5`, `optimizer_weight_decay: 0.01`, `scheduler_decay_steps: 30000`) rather than the values actually used. |
| `optimizer` | AdamW, lr 5e-5, betas (0.9, 0.95), weight decay 1e-10, grad clip norm 1.0 | Weight decay is effectively disabled; regularization comes from the short schedule. Gradient clipping at 1.0 is active. |
| `scheduler` | cosine decay with warmup, 1000 warmup steps, peak 5e-5, floor 2.5e-6, decay over 40000 | The decay horizon equals `steps`, so the schedule is baked into the step count: a run cannot be extended past 40000 and stay on the intended curve. |
| `save_checkpoint` / `save_freq` | true / **2000** | 20 checkpoints over the run. The wrapper's default is `steps/10`, which would be 4000 and ten checkpoints; the run passed `--save-freq=2000` to get twice the resolution for a checkpoint sweep. 04 §9 verifies against 20 archived directories. |
| `log_freq` | 10 | Log every 10 steps. The per-step `data_s` and `updt_s` in that log are the feeding-health signal. |
| `wandb.disable_artifact` | true | Do not upload checkpoints as W&B artifacts; a full pi0.5 checkpoint is about 7 GB. |
| `max_delay_steps` | 8 | VLASH temporal delay augmentation. Each sample's action chunk is offset by a random delay drawn from [0, 8], training the policy to predict where the robot will be by the time the chunk arrives rather than where it was when the observation was taken. This is what makes asynchronous inference work at run time. |
| `lora.enable` | false | Full fine-tune: all 3.62B parameters are trainable and `apply_lora` is a no-op. |

**Learning rate.** 5e-5 peak with cosine decay to 2.5e-6 over 40000 steps at effective batch
256. Note that `PI05Config`'s own default is 2.5e-5 over 30000 steps; the explicit block
overrides both.

---

## 6. Wrapper scripts

Five scripts in `/opt/scripts`. The training wrapper is the entry point; the others are called
by it or by the container runtime.

### 6.1 `train-base` — the training entry point

Roughly: parse arguments, fetch the dataset, ensure statistics exist, render the config, launch.

```
train-base --dataset-url=<URL> [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--dataset-url` | required | Local path, `s3://`, `hf://`, `sftp://`, or a Google Drive URL |
| `--batch-size` | 8 | Per GPU |
| `--grad-accum-steps` | 4 | |
| `--lr` | 5e-5 | Peak learning rate |
| `--steps` | 40000 | |
| `--save-freq` | `steps/10` | The released run overrode this with `2000`, giving 20 checkpoints instead of 10 |
| `--warmup-steps` | `max(steps * 2.5%, 1000)` | 1000 at the default step count |
| `--output-dir` | `/workspace/checkpoints/<name>` | |
| `--wandb-project` | unset = W&B disabled | |
| `--run-id` | dataset name | Checkpoint archive slot and W&B run name |
| `--init-checkpoint` | unset | Warm restart from a saved model directory |

What it does, in order:

1. **Cache directories.** Exports `HOME`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`,
   `TORCHINDUCTOR_CACHE_DIR`, `TORCH_HOME`, `HF_HOME`, `TRANSFORMERS_CACHE`, `MPLCONFIGDIR`
   under `/workspace/.cache` and creates them. Redundant with the image `ENV` when started by
   `docker run`, necessary when started from a bare shell.

2. **Defaults that depend on other arguments.** Warmup is `max(steps * 2.5%, 1000)`; save
   frequency is `steps/10`, floored at 1. Both are defaults only, applied when the flag is
   absent — the released run passed `--save-freq=2000` explicitly.

3. **GPU count, for the banner only.** From `CUDA_VISIBLE_DEVICES` if set, otherwise
   `nvidia-smi -L`. It prints the effective batch (`batch x gpus x accum`) so a mistake such as
   passing 256 to `--batch-size` is visible before 40 hours are spent. The actual multi-GPU
   launch is VLASH's: `vlash train` detects the GPU count and launches
   `accelerate launch --multi_gpu --num_processes=N` with plain DDP.

4. **Dataset.** `DATASET_PATH=$(prepare-dataset "$DATASET_URL")`.

5. **Statistics.** Calls `inject-stats`, which is a no-op if `meta/stats.json` already exists.

6. **Config rendering.**

   ```bash
   export SLOT_ID="${RUN_ID:-$DATASET_NAME}"
   export RESUME_FREQ="$STEPS"
   envsubst < /opt/configs/base.yaml > "$YAML_RESOLVED"
   ```

   `SLOT_ID` names the checkpoint archive slot. `RESUME_FREQ` is set to the total step count so
   that the resume-upload path (which belongs to the LoRA branch) never triggers mid-run.

7. **Launch.**

   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       vlash train "$YAML_RESOLVED" $PRETRAINED_OVERRIDE 2>&1 | tee "$OUTPUT_DIR/train.log"
   TRAIN_RC=${PIPESTATUS[0]}
   ```

   `expandable_segments` reduces allocator fragmentation. `PIPESTATUS` preserves the training
   exit code through the `tee`, so a failed run does not report success.

Two flags are worth calling out. `--init-checkpoint` runs the URL through `prepare-dataset` and
appends `--policy.pretrained_path=<downloaded>` to the launch, which restarts from a saved model
with a **fresh optimizer and a fresh schedule** — a recovery path after a host failure, not a
true resume. And omitting `--wandb-project` sets `enable: false` while still substituting a
placeholder project name, because `envsubst` leaves unset variables as empty strings and an
empty YAML value would be a parse error.

### 6.2 `prepare-dataset` — fetch to a local path

Takes a URL, prints a local path on stdout, logs to stderr. The stdout/stderr split is what lets
the caller do `DATASET_PATH=$(prepare-dataset "$URL")`.

Content-addressed by URL:

```bash
HASH=$(echo -n "$URL" | sha256sum | head -c 16)
TARGET="$CACHE_DIR/$HASH"

if [[ -d "$TARGET" && -f "$TARGET/.complete" ]]; then
    echo "$TARGET"; exit 0
fi
```

The `.complete` marker is written only after a successful transfer, so an interrupted download
is not mistaken for a cache hit on the next attempt.

Schemes:

| Scheme | Behaviour |
|---|---|
| local path (`/`, `./`, `~`) | Returned as is. **Not copied into the cache** — a 198 GB local dataset is used in place. |
| `s3://` | `aws s3 sync` |
| `hf://` | `huggingface_hub.snapshot_download(repo_type="dataset")` |
| `sftp://` | `scp -r`, with `sshpass` if `SFTP_PASSWORD` is set |
| Google Drive | `gdown`, folder or zipped file |

For SFTP and Google Drive folders the transfer usually lands one directory deeper than intended,
so the script flattens a single top-level subdirectory into the target.

A practical note on S3: the default AWS CLI concurrency of 10 is slow for a corpus of about
68,000 small objects. Run `aws configure set default.s3.max_concurrent_requests 256` in the
container before the wrapper. Transfer rate also depends on where the rented host sits relative
to the bucket, which 04 §1 measures.

### 6.3 `inject-stats` — generate `meta/stats.json` when it is missing

LeRobot requires normalization statistics in `meta/stats.json`. Some datasets recorded by
in-house tooling do not have them, and training without them produces NaN losses immediately.

The script exits early if the file exists, so on a properly built dataset it does nothing. When
it does run, it concatenates every parquet file under `data/` (both the `episode_*.parquet` and
the `file-*.parquet` layouts) and computes, per column, `min`, `max`, `mean`, `std`, `count`,
and the q01 / q10 / q50 / q90 / q99 quantiles for:

- scalar columns: `timestamp`, `frame_index`, `task_index`, `index`, `episode_index`
- vector columns: `observation.state`, `action`

Camera keys get placeholder statistics (mean 0.5, std 0.25, and so on). pi0.5 maps visual
features to `IDENTITY` normalization, so these numbers are not applied to images by this policy;
they exist so that the file is schema-complete for LeRobot.

For the run described here, the merged corpus already shipped statistics, so this was a no-op.
Reading whole parquet files into memory does not scale to a corpus this size, which is the other
reason not to rely on it there.

### 6.4 `s3_checkpoint.py` — checkpoint archive and local rotation

A small CLI with the pure logic separated from the `aws` calls so the logic is unit-testable.

```
s3_checkpoint.py download      <slot> <output_dir>
s3_checkpoint.py upload-resume <slot> <ckpt_dir> <step>
s3_checkpoint.py upload-archive <slot> <dir> <step>
s3_checkpoint.py prune-local   <checkpoints_dir> [keep]
s3_checkpoint.py cleanup       <slot>
```

The S3 base is taken from the `S3_CKPT_BASE` environment variable. **If it is unset, uploads are
skipped silently and checkpoints exist only on the container's local disk** — which is destroyed
with the instance. This is the single most expensive misconfiguration available here; confirm
the variable is literally present in the `docker run` command.

Layout under the base:

```
<S3_CKPT_BASE>/archive/<slot>/<step>/     model-only, permanent, one per save_freq
<S3_CKPT_BASE>/resume/<slot>/ckpt-<step>/ full state including optimizer, rolling
<S3_CKPT_BASE>/resume/<slot>/LATEST.json  pointer, written last
```

Relevant behaviour for a full fine-tune:

- After each checkpoint, `upload-archive` copies `<checkpoint>/pretrained_model/` (config plus
  `model.safetensors`) to `archive/<slot>/<step>/`, and `prune-local` deletes all but the two
  most recent local checkpoints. Without the pruning, ten 7 GB checkpoints fill the disk.
- `prune-local` never deletes the directory the `last` symlink points at, in addition to the two
  most recent.
- Integrity is checked with a `manifest.json` of file names and sizes written into each uploaded
  directory (excluding itself). `download` verifies the manifest and falls back to the previous
  rolling checkpoint if the newest one is incomplete. `LATEST.json` is uploaded last, so a
  partially uploaded checkpoint is never pointed at.
- All `aws` failures are caught and reported without a traceback, and `train.py` invokes the
  script with `check=False`: a transient S3 error must not end a 40-hour run.

The **full-state resume path is wired for the LoRA branch only**. In a full fine-tune, only the
model is archived; the optimizer state is not uploaded. Recovery from a lost host is a warm
restart via `--init-checkpoint`, which reinitializes the optimizer and restarts the cosine
schedule from step 0 — usable in an emergency, not equivalent to the original run. Use
on-demand rather than interruptible instances for a full fine-tune.

### 6.5 `container-init.sh` — the default command

Runs when no command is given to `docker run`.

1. Recreates `/workspace/{datasets,checkpoints,logs,.cache}` **at run time**, because a mounted
   volume hides whatever the image created at that path during build.
2. Looks for an SSH public key in `PUBLIC_KEY`, `SSH_PUBLIC_KEY` or `AUTHORIZED_KEYS` — three
   names because different providers use different ones — writes it to
   `/root/.ssh/authorized_keys` and starts `sshd`. With none of them set (a local run) it skips
   SSH entirely.
3. `exec sleep infinity`.

Batch mode is the same image with the command overridden:

```bash
docker run <image> /opt/scripts/train-base --dataset-url=... --steps=...
```

---

## 7. Building

The build context is `training-docker/`:

```bash
cd training-docker

DOCKER_BUILDKIT=1 docker build \
  --build-arg VLASH_COMMIT=22cbabfee0f57874987c75a35a7dac129e695db0 \
  -t <YOUR_REGISTRY>/pi05-so101-train:latest \
  .
```

**BuildKit is required, and the legacy builder fails quietly rather than loudly.** The
Dockerfile declares `# syntax=docker/dockerfile:1.4` on its first line and writes
`/etc/profile.d/train-env.sh` with a heredoc (§2.7). The legacy builder does not reject the
heredoc; it drops the body and carries on, producing an image that builds cleanly and whose SSH
login shells have no `PATH` and no conda environment. Because the default command is an SSH
server (§6.5), that defect surfaces as a `command not found` on a rented GPU host, not at build
time. Install the `buildx` component, confirm with `docker buildx version`, and pass
`DOCKER_BUILDKIT=1` unless the daemon already defaults to BuildKit. §8.2 is where a broken login
environment would be caught.

`pi05-so101-train` is the image name used throughout these documents, including the launch
command in 04 §3. It is not the same thing as `train-base`, which is the wrapper script inside
the image; keeping the two names distinct avoids a confusing command line.

No secrets are needed at `docker build` time, but the build context must already contain the
tokenizer, which is fetched with your own gated-repository access (§2.6). The context must
contain `environment.yml`, `configs/`, `scripts/`, `patched/` and `paligemma_tokenizer_flat/`.
The first four ship in this repository; `paligemma_tokenizer_flat/` you produce yourself, with
the script described in §2.6.

**Time.** About 30 to 60 minutes for a cold build on a reasonable connection. This is a recorded
estimate rather than a stopwatch measurement. Two steps dominate: creating the conda environment
and its pip dependency tree, and downloading the roughly 14 GB base checkpoint.

**Size.** The resulting image measured **53.2 GB on disk, 19.7 GB of compressed content**. Most
of it is the baked-in base checkpoint plus the conda environment with PyTorch and its bundled
CUDA libraries. Budget disk accordingly on the training host: the image, the dataset, and
rolling checkpoints all land on the same volume. For the run described here that meant at least
400 GB (image, 198 GB dataset, about 45 GB of rolling checkpoints, caches).

**Pull time** is not negligible either — about 13 minutes was observed on a well-connected
rented host.

**Layer cache.** The order is chosen so that iterating on patches does not rebuild the
environment: system packages, conda environment, then the VLASH checkout and patches. Editing a
file under `patched/` invalidates only from the `COPY` onward.

---

## 8. Verifying the build

Run these in order. The first three are cheap and catch most build mistakes; the last two cost
GPU time and catch the rest. Every one of them exists because something got past the previous
one at some point.

### 8.1 Host-side unit tests

Two things are tested outside the image, on the build context:

```bash
pytest tests/
```

- `test_transform_picklable.py` — asserts that `_resize_with_pad` and `_resize_with_pad_then_aug`
  survive `pickle.dumps`/`loads` and that they turn both a `(3, 480, 640)` and a
  `(3, 1080, 1920)` tensor into exactly `(3, 224, 224)`. This is the regression test for the
  DataLoader-worker crash described in section 4.1.
- `test_s3_checkpoint.py` — manifest verification and the rolling-deletion selection logic.

### 8.2 Image sanity

```bash
docker run --rm <IMAGE> bash -lc '
  python -c "import torch, lerobot, torchcodec, vlash; \
             print(torch.__version__, torch.version.cuda); \
             print(lerobot.__version__, torchcodec.__version__)"
  python -c "import multiprocessing as mp; print(mp.get_start_method())"
  ls /opt/models/pi05_base | head
  ls /opt/models/paligemma_tokenizer
  which train-base prepare-dataset envsubst aws
'
```

Expected: torch `2.7.1+cu126` / CUDA `12.6`, lerobot `0.4.1`, torchcodec `0.5`, start method
`fork`, both model directories populated, all four executables found. `envsubst` missing means
`gettext-base` was dropped from the apt list and the wrapper will fail at config rendering.

### 8.3 The patches are actually in the image

The one that matters most, because a wrong `COPY` order produces a working image with the patch
missing:

```bash
docker run --rm <IMAGE> bash -lc '
  python - <<PY
import inspect, lerobot.datasets.video_utils as v
src = inspect.getsource(v.VideoDecoderCache)
assert "VLASH_DECODER_CACHE_MAX" in src, "decoder cache patch MISSING from site-packages"
print("video_utils patch present")
PY
  grep -q "_resize_with_pad" /opt/vlash/vlash/train.py && echo "train.py patch present"
  grep -q "_mask" /opt/vlash/vlash/policies/pi05/modeling_pi05.py && echo "modeling patch present"
  git -C /opt/vlash rev-parse HEAD
'
```

The `git rev-parse` should print the pinned commit. Note that `git -C /opt/vlash status` will
show the three patched files as modified — that is expected and is how you confirm the patches
landed on the intended base tree.

### 8.4 GPU-side forward and one training step

On a single GPU, against a small dataset that has mask columns:

- **Forward.** Batch 2, bf16, `no_grad`, with two episodes that have different camera sets.
  Expected: `input_features` contains no `_mask` keys, the mask values differ between the two
  samples in the batch, and the loss is finite. Reference values from this build: loss 0.0190.
- **One training step.** Forward, backward, `optimizer.step`, and a check that weights changed.
  Reference values on a 24 GB card at batch 1 with an 8-bit optimizer and gradient checkpointing
  forced on: 3.62B trainable parameters, loss 0.0566, grad norm 4.1091 finite, peak 21.0 GB.
  (That configuration is only to make a full fine-tune fit on a single consumer card for the
  test; it is not the training recipe.)

### 8.5 End-to-end container smoke test

Before committing to the full run, run the real command with a tiny step count on the real
hardware:

```bash
docker run --gpus all --shm-size=64g --ulimit nofile=1048576:1048576 \
  -e AWS_ACCESS_KEY_ID=<KEY> -e AWS_SECRET_ACCESS_KEY=<SECRET> \
  -e AWS_DEFAULT_REGION=<REGION> \
  -e S3_CKPT_BASE=s3://<YOUR_BUCKET>/checkpoints \
  -e WANDB_API_KEY=<WANDB_KEY> \
  <IMAGE> \
  bash -lc 'aws configure set default.s3.max_concurrent_requests 256 && \
    train-base \
      --dataset-url=s3://<YOUR_BUCKET>/<DATASET_PREFIX> \
      --run-id=<RUN_ID>-smoke \
      --batch-size=8 --grad-accum-steps=4 --lr=5e-5 \
      --steps=20 --save-freq=10'
```

Takes about ten minutes and covers the whole path. Check:

| Gate | Expected |
|---|---|
| Image pull | Succeeds; no registry auth error |
| Dataset transfer rate | Enough to move the corpus in reasonable time. Under about 1 MiB/s means the host is in the wrong place; pick another. |
| GPUs | `nvidia-smi` shows all of them; NVLink present on an SXM node |
| Effective batch in the banner | `batch x gpus x accum` matches intent |
| First loss | Finite. Reference range from this recipe: roughly 0.06 to 0.09 |
| Peak GPU memory | About 52 GB per GPU at per-GPU batch 8 — a run observation rather than a controlled measurement (§9), against a pre-run estimate of 52-55 GB. 04 §1 records the real run's per-GPU peak as **not recorded**, so use this as a sanity band, not a threshold. If a card OOMs, halve the batch and double `grad_accum_steps`, which keeps effective batch at 256 and the learning rate schedule valid. |
| `data_s` versus `updt_s` in the log | `data_s` near zero, `updt_s` around 3.4 s. `data_s` comparable to `updt_s` means the GPUs are waiting on data. |
| DataLoader workers | No `Bus error` (shm) and no `Too many open files` (nofile) |
| **Checkpoint in S3** | `aws s3 ls s3://<YOUR_BUCKET>/checkpoints/archive/<RUN_ID>-smoke/10/` shows `model.safetensors` and a config. **If this is empty, do not start the full run** — `S3_CKPT_BASE` did not take effect and the full run's output will be lost when the instance is destroyed. |

Two of the four problems this run hit were invisible to a 20-step smoke test — the file
descriptor exhaustion appeared around step 70, and the timestamp violation appeared around step
140 because the sampler had not drawn one of the affected episodes yet. Watch the first
thousand steps of the real run with the same attention, not just the smoke test.

---

## 9. Gaps and things not recorded

- The choice of CUDA 12.1 for the base image is not recorded, and it does not match the torch
  wheel's CUDA 12.6. It works because the wheel bundles its own CUDA libraries.
- Build time is a recorded estimate (30 to 60 minutes), not a measurement.
- Peak GPU memory at per-GPU batch 8 on 80 GB is approximately 52 GB, from a run observation
  rather than a controlled measurement. The only rigorously measured memory figure is the
  single-card test in section 8.4, which used a different configuration.
- `gradient_checkpointing` appears in the config and in some notes as if it were active. It is
  declared in `PI05Config` but never read by the model code in this VLASH commit, so it has no
  effect either way.
- The training transform pads right and bottom while the model's own `resize_with_pad` pads left
  and top. The consequences of that asymmetry for non-square inference inputs were not measured.
