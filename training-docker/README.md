# Training image

Docker build for the pi0.5 full fine-tune described in `../docs/03-training-environment.md`
and `../docs/04-training-execution.md`. It trains a pi0.5 policy (about 3.62B parameters,
`lora.enable=false`) on a unified three-slot SO-101 dataset.

The image pins upstream VLASH to one commit, replaces four files in the stack with the
patched copies under `patched/`, and bakes in the pi0.5 base checkpoint so a long run needs
no model-hub access.

## Contents

```
Dockerfile                 the image
environment.yml            conda environment (Python 3.10, ffmpeg 7.1)
configs/base.yaml          training config template, resolved by envsubst
patched/                   four patched source files, copied over the installed stack
scripts/train-base         training entry point
scripts/prepare-dataset    dataset fetch (local / s3 / hf / sftp / Google Drive)
scripts/inject-stats       generates meta/stats.json when a dataset lacks one
scripts/container-init.sh  default CMD: interactive SSH mode
scripts/s3_checkpoint.py   S3 checkpoint slots and local rolling prune
scripts/fetch-tokenizer.sh downloads the gated PaliGemma tokenizer
tests/                     host-side tests, no GPU required
```

## Prerequisites

**BuildKit is required.** The Dockerfile writes `/etc/profile.d/train-env.sh` with a heredoc
and declares `# syntax=docker/dockerfile:1.4` on its first line. The legacy builder drops the
heredoc body instead of failing loudly, which yields an image whose SSH login shells have no
`PATH` and no conda environment. Install the `buildx` component and, if your daemon does not
already default to BuildKit, set `DOCKER_BUILDKIT=1`. Verify with `docker buildx version`
before building.

**Hugging Face access to the gated PaliGemma repository.**

`google/paligemma-3b-pt-224` is gated. Its tokenizer is required at training time and is
deliberately *not* vendored in this package: the terms are Google's to grant, not ours to
redistribute. Before the first build you must, with your own account:

1. accept the Gemma terms of use on
   <https://huggingface.co/google/paligemma-3b-pt-224>,
2. wait for access to be granted,
3. authenticate locally with `hf auth login`, or export a read token as `HF_TOKEN`.

Then fetch the tokenizer into the build context:

```bash
pip install huggingface_hub          # if it is not already available
./scripts/fetch-tokenizer.sh         # writes ./paligemma_tokenizer_flat (about 21 MB)
```

The pi0.5 base checkpoint `lerobot/pi05_base` (about 14 GB) is downloaded during the build,
with no token in the reference build.

## Verified environment

The resolved versions this image was built and trained with are in the training table of
[`../README.md`](../README.md#verified-environment), which is the canonical list. Only four
things are pinned by hand: `VLASH_COMMIT` in the Dockerfile, and `torch==2.7.1`,
`torchvision==0.22.1`, `torchcodec==0.5` plus `ffmpeg=7.1.*` in `environment.yml`. Everything
else falls out of `pip install -e /opt/vlash` at that commit.

## Build

```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg VLASH_COMMIT=22cbabfee0f57874987c75a35a7dac129e695db0 \
  -t <YOUR_REGISTRY>/pi05-so101-train:latest \
  .
```

Cold build is roughly 30 to 60 minutes, dominated by the conda environment and the 14 GB
base checkpoint. The resulting image is large: about 53 GB on disk.

The build context must contain `environment.yml`, `configs/`, `scripts/`, `patched/` and
`paligemma_tokenizer_flat/`. `tests/` is excluded by `.dockerignore`.

## Run

Reference hardware for the defaults below is a single node with 8 x A100 80GB.
`vlash train` detects the GPU count and launches `accelerate` with plain DDP, so an 8-GPU
box gets 8-way DDP without any extra flag.

```bash
docker run --gpus all --shm-size=64g \
    --ulimit nofile=1048576:1048576 \
    -v <LOCAL_OUTPUT>:/workspace/checkpoints \
    -v <LOCAL_CACHE>:/workspace/.cache \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    <YOUR_REGISTRY>/pi05-so101-train:latest \
    /opt/scripts/train-base \
    --dataset-url=s3://<YOUR_BUCKET>/<DATASET_PREFIX> \
    --batch-size=8 --grad-accum-steps=4 \
    --lr=5e-5 --steps=40000 --save-freq=2000 \
    --wandb-project=<WANDB_PROJECT>
```

Effective batch = 8 (per GPU) x 8 (GPUs) x 4 (accumulation) = 256.

Without a command override the image starts `container-init.sh`, which installs an SSH key
from `PUBLIC_KEY` / `SSH_PUBLIC_KEY` / `AUTHORIZED_KEYS`, starts `sshd` and idles.

**Run a smoke test first.** `--steps=20` exercises every path (dataset fetch, config
resolution, model load, checkpoint save, upload) in a few minutes.

**`--ulimit nofile` is required**, not an optimization. At `num_workers=8` across 8 ranks the
default limit of 1024 file descriptors is exhausted; a worker dies, the surviving ranks block
in all-reduce, and the job hangs instead of crashing. See `../docs/05-troubleshooting.md`.

## Options

| flag | default | meaning |
|---|---|---|
| `--dataset-url` | required | local path, `s3://`, `hf://`, `sftp://`, or a Google Drive URL |
| `--batch-size` | 8 | per GPU; effective = batch x GPUs x accumulation |
| `--grad-accum-steps` | 4 | 8 x 8 GPUs x 4 = effective 256 |
| `--lr` | 5e-5 | peak LR for the full fine-tune, at effective batch 256 |
| `--steps` | 40000 | total optimizer steps |
| `--save-freq` | steps/10 | checkpoint interval |
| `--warmup-steps` | max(steps x 2.5%, 1000) | LR warmup |
| `--output-dir` | `/workspace/checkpoints/<dataset_name>` | checkpoint root |
| `--wandb-project` | unset, logging disabled | W&B project name |
| `--run-id` | dataset name | S3 archive slot |
| `--init-checkpoint` | unset | warm restart from an archived model |

## Environment variables

| variable | used by | meaning |
|---|---|---|
| `WANDB_API_KEY` | training | required when `--wandb-project` is passed |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | dataset, checkpoints | scope them to the buckets in use |
| `S3_CKPT_BASE` | checkpoints | e.g. `s3://<YOUR_BUCKET>/checkpoints`. Unset means local checkpoints only |
| `SLOT_ID` | checkpoints | archive slot; `train-base` sets it from `--run-id` |
| `SFTP_PASSWORD` | dataset | only for `sftp://` sources |
| `HF_TOKEN` | build, dataset | gated repositories and `hf://` sources |
| `VLASH_PALIGEMMA_PATH` | training | tokenizer directory, preset in the image |
| `VLASH_DECODER_CACHE_MAX` | training | video decoder cache entries, default 64 |
| `VLASH_DRY_RUN` | inference | `1` predicts without sending actions to the robot |
| `VLASH_SAFE_DEG` | inference | abort threshold in degrees, default 180 |

## Patches

Four files replace their upstream copies. `NOTICE` lists every change, and each file carries
a `Modified from ...` header.

| file | replaces | why |
|---|---|---|
| `patched/train.py` | `vlash/train.py` | letterbox mixed-resolution frames to 224x224; adapter-only saves; S3 checkpoint slots |
| `patched/modeling_pi05.py` | `vlash/policies/pi05/modeling_pi05.py` | per-sample camera masking through `<camera>_mask`; local tokenizer path |
| `patched/run.py` | `vlash/run.py` | subset camera validation; chunk-boundary blending; adapter-only checkpoint loading |
| `patched/video_utils.py` | `lerobot/datasets/video_utils.py` | bounded decoder cache; decode-error fallback |

`video_utils.py` is copied *after* `pip install -e /opt/vlash`, because lerobot arrives as a
transitive dependency of vlash and the editable install would otherwise overwrite it.

## Tests

The tests need no GPU and are excluded from the build context.

```bash
pip install pytest
pytest tests/
```

`tests/test_transform_picklable.py` imports the patched `vlash.train`, so it needs a vlash
source tree; point `VLASH_SRC` at one, or run it inside the image where `/opt/vlash` already
exists. It is skipped when vlash cannot be imported.

## Known limitations

- **No spot resume for a full fine-tune.** The S3 upload of resume state in `train.py` sits
  on the LoRA branch, so full fine-tune checkpoints are written locally and archived
  model-only. Use on-demand instances with a persistent volume and collect the checkpoints
  afterwards, or warm-restart with `--init-checkpoint`.
- **80 GB per GPU.** At the defaults, plain DDP replicates the optimizer state on every GPU.
  40 GB cards go out of memory; that would need an 8-bit optimizer or an FSDP launcher,
  neither of which is wired up here.
- **`gradient_checkpointing` in `configs/base.yaml` is a no-op.** vlash does not read it. It
  is left in place because the key is part of the policy config schema.

## License

Apache License 2.0. See `LICENSE` for the full text and `NOTICE` for the third-party
copyright notices, the list of modified files, and the terms covering the model weights and
the tokenizer.

This stack contains derivatives of VLASH, LeRobot and openpi, all Apache-2.0. If you
redistribute an image built from this directory, ship `LICENSE` and `NOTICE` with it.
