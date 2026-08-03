# 04 — Training execution

How the released checkpoint was actually trained: instance requirements, the exact command,
every hyperparameter, what the run looked like while it was running, and how to tell that it
finished correctly.

The run being described: pi0.5 full fine-tune, 3.62B parameters, 8 x A100 80GB SXM4 on a
single node, 40000 steps, about 40 hours wall clock, clean exit.

Values marked "not recorded" were not measured during the run and are not reconstructed here.

---

## 1. Instance requirements

### GPUs

| Item | Requirement |
|---|---|
| Count | 8 GPUs in one node. Multi-node was not used and is not wired up. |
| Memory | 80 GB per GPU. The full 3.62B model in bf16 plus AdamW moments plus activations at per-GPU batch 8 does not fit in 40 GB. |
| Interconnect | NVLink (SXM). A PCIe-only node is expected to lose 10-15% to all-reduce; estimate, not measured. |
| Model used | A100 80GB SXM4. H100 SXM5 is equivalent for this recipe (same 80 GB, same NVLink). |

Per-GPU peak memory during the real run was **not recorded**. The pre-run estimate was 52-55 GB
at per-GPU batch 8; measure it during the smoke run and only fall back to per-GPU batch 4 with
gradient accumulation 8 if it does not fit (that keeps the effective batch at 256).

### Host

| Item | Value used | Note |
|---|---|---|
| OS | Ubuntu 22.04 | Provider's stock GPU image, unmodified |
| Driver / CUDA | not recorded | The provider image shipped an NVIDIA driver and `nvidia-container-toolkit`; `docker run --gpus all` worked as delivered. The container is built on `nvidia/cuda:12.1.0-runtime-ubuntu22.04` but installs PyTorch 2.7.1+cu126, so the host driver must be new enough for CUDA 12.6. |
| vCPU | 240 | Per instance-type spec. 8 ranks x 8 dataloader workers = 64 worker processes. |
| RAM | ~1.7 TiB | Anonymous memory plateaued near 96 GB once the decoder cache was bounded. Budget at least 128 GB of usable RAM; more if you raise `num_workers`. |
| Disk | 20 TiB local NVMe | Actual requirement is about 450 GB: container image 53.2 GB, dataset ~198 GB, rolling local checkpoints ~45 GB, caches. |
| Network | not recorded (>= 1 Gbps class) | See below; raw bandwidth was never the binding constraint. |

### Choosing where the instance runs

The single most expensive mistake available at this stage is renting a GPU node on the wrong
side of an ocean from the object storage holding the dataset.

Measured on an earlier attempt: a host in Japan pulling from a bucket in US East sustained
**0.87 MiB/s**. At that rate the 198 GB / ~68,000 object dataset never finishes, and you pay
GPU rent the entire time. A host on the US East Coast, close to the same bucket, moved the
large video files at 56 to several hundred MiB/s.

Rules that came out of that:

- Pick the region by proximity to the data, not by GPU price. An hour of 8 x A100 costs more
  than any plausible egress difference.
- Confirm throughput in the first minutes of the transfer and kill the instance if it is slow.
  Nothing later in the run recovers from a bad host location.
- Small files dominate. 17,137 parquet files transfer far slower per byte than 51,411 videos
  because of per-object overhead, so the average rate looks worse early on.

### Container image

Built from `training-docker/` — Dockerfile, the four patched source files under `patched/`, the
wrapper scripts under `scripts/`, and `configs/base.yaml`. 03 is the build recipe; this section
only lists what ends up inside.

The image bakes everything, so the training node needs no internet access beyond the object
store:

| Component | Version |
|---|---|
| Base image | `nvidia/cuda:12.1.0-runtime-ubuntu22.04` |
| Python | 3.10 (conda-forge) |
| PyTorch | 2.7.1+cu126 |
| torchcodec | 0.5 |
| ffmpeg | 7.1.x (conda-forge; torchcodec needs libavutil from ffmpeg 7) |
| LeRobot | 0.4.1 (transitive dependency of the training stack) |
| Training stack | VLASH, pinned to commit `22cbabfee0f57874987c75a35a7dac129e695db0` |
| Base weights | `lerobot/pi05_base`, downloaded at build time to `/opt/models/pi05_base` |
| Tokenizer | PaliGemma tokenizer copied into the image; `HF_HUB_OFFLINE=1` is set |

The image is 53.2 GB on disk, 19.7 GB compressed. A cold pull on a fresh host took about 13
minutes.

---

## 2. Data preparation

The wrapper's `prepare-dataset` step accepts several source forms and decides what to do by URL
scheme:

| Form | Behavior |
|---|---|
| `/absolute/path` | Used in place. No copy, no cache directory. |
| `s3://bucket/prefix` | `aws s3 sync` into `/workspace/.cache/datasets/<sha256 of url>`, then a `.complete` marker so a restart reuses it. |
| `hf://repo_id` | `snapshot_download` into the same cache layout. |
| `sftp://` / Google Drive URL | Downloaded and flattened into the cache. |

### Object storage path

Raise the AWS CLI concurrency **before** starting the wrapper. The default of 10 concurrent
requests is the wrong setting for ~68,000 small objects, and the wrapper does not set it:

```bash
aws configure set default.s3.max_concurrent_requests 256
```

Total transfer wall-clock time is **not recorded**. It is entirely host-location dependent; see
section 1.

### Local path

The actual run used a pre-staged local directory rather than the bucket, because the dataset
had to be repaired on the instance mid-run: the 450 episodes recorded at 10 fps were filtered
out and the result re-indexed, which is what the `-v /data/so101_base_30fps` mount in the launch
command below refers to. Section 10 has the failure that forced it; 02, under "Producing the
corpus that was actually trained", has the procedure. Passing an absolute path skips the sync
completely, which is also the fastest option when the node has a persistent volume or the data
was staged in a previous run.

### Pre-flight data checks

Both of the checks below cost nothing and each of them corresponds to a failure that killed a
paid run:

**1. Timestamps start where the video starts.** LeRobot asserts that a frame's parquet timestamp
matches the decoded video timestamp within 1e-4 s. Sources recorded at 10 fps can have a first
video PTS of 0.1 s while their parquet timestamps start at 0.

```bash
aws s3 cp s3://<YOUR_BUCKET>/datasets/<dataset>/data/chunk-000/episode_000000.parquet - \
  | python3 -c "import sys,io,pyarrow.parquet as pq; \
t=pq.read_table(io.BytesIO(sys.stdin.buffer.read()))['timestamp'].to_numpy(); \
print('ts0=',t[0],'monotonic=',bool((t[1:]>=t[:-1]).all()))"
```

Check every chunk, not a sample. The episodes that break this were 2.6% of the corpus and a
random sample missed them.

**2. `info.json` frame rate matches the videos.** A merged corpus can carry `fps: 30` in metadata
while some videos are 10 fps. That mismatch produces the same tolerance failure.

Verify the object count as a last sanity check:

```bash
aws s3 ls s3://<YOUR_BUCKET>/datasets/<dataset>/ --recursive | wc -l
```

---

## 3. Running the training

### Smoke run first

Same command as the real run with `--steps=20 --save-freq=10`. About 10 minutes. It verifies
image pull, dataset access, 8-way NCCL init, the first optimizer step, and — most importantly —
that a checkpoint actually lands in remote storage.

```bash
aws s3 ls s3://<YOUR_BUCKET>/checkpoints/archive/<RUN_ID>-smoke/10/
```

If `model.safetensors` is not there, stop. Everything else can be fixed later; a 40-hour run
whose checkpoints only exist on an ephemeral instance cannot.

Know what the smoke run does **not** catch: file-descriptor exhaustion needs a few hundred steps
to build up, rare malformed episodes need enough sampling to be hit, and the memory growth
pattern is not visible in 20 steps. All three of those bit this run after the smoke test passed.

### Full command

```bash
docker run \
  --gpus all \
  --shm-size=64g \
  --ulimit nofile=1048576:1048576 \
  -v /data/so101_base_30fps:/data/so101_base_30fps:ro \
  -e AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> \
  -e AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> \
  -e AWS_DEFAULT_REGION=<AWS_REGION> \
  -e S3_CKPT_BASE=s3://<YOUR_BUCKET>/checkpoints \
  -e WANDB_API_KEY=<WANDB_API_KEY> \
  -e VLASH_DECODER_CACHE_MAX=64 \
  <YOUR_REGISTRY>/pi05-so101-train:latest \
  bash -lc 'aws configure set default.s3.max_concurrent_requests 256 && \
    /opt/scripts/train-base \
      --dataset-url=/data/so101_base_30fps \
      --run-id=<RUN_ID> \
      --batch-size=8 \
      --grad-accum-steps=4 \
      --lr=5e-5 \
      --steps=40000 \
      --save-freq=2000 \
      --wandb-project=<WANDB_PROJECT>'
```

Every option earns its place:

| Option | Why |
|---|---|
| `--gpus all` | Exposes all 8 GPUs. The launcher counts GPUs and sizes the DDP world from what it sees. |
| `--shm-size=64g` | Docker defaults `/dev/shm` to 64 MB. PyTorch DataLoader workers pass tensors through shared memory; with 8 workers per rank the default produces `Bus error` and the run dies. 16 GB is the floor, 64 GB is what was used. |
| `--ulimit nofile=1048576:1048576` | **Required.** 8 workers x 8 ranks across a dataset of 51,411 videos blows through the default 1024 file-descriptor limit. The symptom is not an error message: around step 70 a worker dies, the remaining ranks block forever in NCCL all-reduce, and `nvidia-smi` shows 100% utilization at idle power draw (~96 W). The job hangs, it does not crash. |
| `-v /data/...:ro` | Only needed when passing a local dataset path. Omit when using `s3://`. |
| `-e AWS_*` | Credentials for the dataset read and the checkpoint archive. Use a scoped key limited to those two prefixes, never an administrative key: this key sits on a rented third-party machine for 40 hours. |
| `-e S3_CKPT_BASE` | **Silent failure if omitted.** The archive uploader turns itself off when this is unset, training proceeds normally, and every checkpoint is lost when the instance is terminated. Grep for it literally in your command before pressing enter. |
| `-e WANDB_API_KEY` | Optional. Enables the metrics dashboard. |
| `-e VLASH_DECODER_CACHE_MAX=64` | Bounds LeRobot's video decoder cache (section 10, issue 3). Only meaningful with a patched `video_utils.py`; harmless otherwise. |

### Wrapper arguments

| Argument | Value | Meaning |
|---|---|---|
| `--dataset-url` | local path or `s3://...` | Dataset source; resolved by `prepare-dataset` |
| `--run-id` | `<RUN_ID>` | Archive slot name (`<S3_CKPT_BASE>/archive/<RUN_ID>/`) and dashboard run name |
| `--batch-size` | 8 | **Per GPU.** Not the effective batch. |
| `--grad-accum-steps` | 4 | 8 x 8 GPUs x 4 = effective batch 256 |
| `--lr` | 5e-5 | Peak learning rate |
| `--steps` | 40000 | Total optimizer steps |
| `--save-freq` | 2000 | Checkpoint every 2000 steps, 20 checkpoints total |
| `--warmup-steps` | (default) | Defaults to `max(steps * 2.5%, 1000)`, which is 1000 here |
| `--wandb-project` | name | Omit to disable metrics logging |
| `--init-checkpoint` | (unused) | Warm restart from an archived model, see section 6 |

What the wrapper does, in order: resolve the dataset, inject normalization statistics if the
dataset lacks them, render the YAML config from environment substitution, then exec the trainer
with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, teeing everything to
`<output_dir>/train.log`.

---

## 4. Hyperparameters

These tables are transcribed from the `train_config.json` written next to the released weights.
That file, not any runbook, is authoritative.

### Training loop

| Key | Value | Note |
|---|---|---|
| `steps` | 40000 | |
| `batch_size` | 8 | per GPU |
| `grad_accum_steps` | 4 | effective batch 8 x 8 x 4 = 256 |
| `num_workers` | 8 | dataloader workers per rank |
| `seed` | 1000 | |
| `log_freq` | 10 | |
| `save_freq` | 2000 | 20 checkpoints |
| `save_checkpoint` | true | |
| `eval_freq` | 20000 | inert: no simulation environment was configured, so no evaluation ran |
| `use_policy_training_preset` | false | the top-level `optimizer`/`scheduler` blocks below are used instead of the policy's built-in preset |
| `max_delay_steps` | 8 | inference-delay simulation during training, matching the asynchronous inference used at evaluation time |
| `resume` | false | |
| `output_dir` | `/workspace/checkpoints/<dataset name>` | |

### Optimizer and schedule

| Key | Value |
|---|---|
| `optimizer.type` | adamw |
| `optimizer.lr` | 5e-5 (peak) |
| `optimizer.betas` | [0.9, 0.95] |
| `optimizer.eps` | 1e-8 |
| `optimizer.weight_decay` | 1e-10 |
| `optimizer.grad_clip_norm` | 1.0 |
| `scheduler.type` | cosine_decay_with_warmup |
| `scheduler.num_warmup_steps` | 1000 |
| `scheduler.num_decay_steps` | 40000 |
| `scheduler.peak_lr` | 5e-5 |
| `scheduler.decay_lr` | 2.5e-6 |

### Policy

| Key | Value |
|---|---|
| `type` | pi05 |
| `pretrained_path` | local copy of `lerobot/pi05_base` |
| `paligemma_variant` | gemma_2b |
| `action_expert_variant` | gemma_300m |
| `dtype` | bfloat16 |
| `use_amp` | false |
| `n_obs_steps` | 1 |
| `chunk_size` | 50 |
| `n_action_steps` | 50 |
| `max_state_dim` / `max_action_dim` | 32 / 32 |
| `state_cond` | true |
| `num_inference_steps` | 10 |
| `time_sampling_beta_alpha` / `_beta` | 1.5 / 1.0 |
| `time_sampling_scale` / `_offset` | 0.999 / 0.001 |
| `min_period` / `max_period` | 0.004 / 4.0 |
| `image_resolution` | [224, 224] |
| `empty_cameras` | 0 |
| `tokenizer_max_length` | 200 |
| `normalization_mapping` | VISUAL: IDENTITY, STATE: MEAN_STD, ACTION: MEAN_STD |
| `gradient_checkpointing` | false (unused by this stack; no-op) |
| `compile_model` | false |
| `fuse_qkv` / `fuse_gate_up` | false / false |
| `lora.enable` | **false** — full fine-tune, all parameters trainable |

Features: input `observation.state` (6), three image slots declared at (3, 480, 640); output
`action` (6).

### Dataset block

| Key | Value |
|---|---|
| `episodes` | null (all) |
| `image_transforms.enable` | false |
| `use_imagenet_stats` | true |
| `video_backend` | torchcodec |
| `streaming` | false |
| `revision` | null |

### Why these values

**Effective batch 256.** This is the reference full fine-tune batch for pi0.5 in openpi. It is
reached as per-GPU 8 x 8 GPUs x accumulation 4 rather than a larger per-GPU batch because 80 GB
is the constraint: only the per-GPU batch affects memory, so accumulation buys the batch size
for free at the cost of wall clock. The same decomposition is what makes the recipe portable —
see section 5.

**Peak LR 5e-5, cosine to 2.5e-6.** The openpi full fine-tune value. Full fine-tuning updates the
pretrained VLM backbone, which tolerates far less learning rate than an adapter would. The floor
is 1/20 of the peak rather than 0, so the last steps still move.

**Warmup 1000 steps (2.5%).** At step 0 the action expert is being asked to model an action
distribution it has not seen, and the resulting gradients are large — the first step logged a
gradient norm around 4.1 against a post-warmup steady state near 0.058. Warmup keeps those early
updates from damaging pretrained features.

**40000 steps = 1.19 epochs.** 40000 x 256 = 10,240,000 samples. Which epoch number that is
depends on which frame count you divide by, and the two differ because 450 episodes were removed
after the merge:

| Denominator | Frames | Epochs |
|---|---|---|
| the corpus actually fed to this run, 16,687 episodes | 8,595,621 | **1.19** |
| the full merged corpus, 17,137 episodes | 8,690,531 | 1.178 |

The run log shows 1.19 because by the time the run started the 450 episodes were already gone;
that is the number this document uses. Quote 1.178 only when describing the merged dataset rather
than the training run. Either way it is just over one pass. This sits inside the openpi full
fine-tune range (roughly 30k for small task sets, 100k for large ones) and was chosen for a
corpus this size and a fixed compute budget. Two consequences:
the cosine schedule is baked into the number, so extending the run afterwards produces a
mismatched schedule; and the loss curve should be read as "one pass over a large corpus", not as
convergence on a task.

**Weight decay 1e-10.** Effectively off. Decay is not useful over a single epoch of fine-tuning
and would pull the pretrained weights toward zero for no benefit.

**bf16 without AMP.** Weights are held in bfloat16 directly, so there is no gradient scaler and no
fp16 overflow handling to tune.

**Image transforms disabled.** The corpus already spans 181 source datasets with different rooms,
lighting, cameras and resolutions. Synthetic color jitter adds little on top of that. The one
image operation that does run is a resize-with-pad to 224, applied as a transform by the training
patch so that mixed source resolutions can be collated without re-encoding any video.

**`num_workers` 8.** With 8 workers, data loading time per step was 0.002 s. With 0 workers it was
1.05 s and the GPUs alternated between idle and busy, roughly 1.5x slower overall.

---

## 5. Multi-GPU

`vlash train` counts visible GPUs and re-launches itself under
`accelerate launch --multi_gpu --num_processes=<N>`. There is nothing to configure: run the
container with `--gpus all` and an 8-GPU node produces an 8-rank job. Restrict it with
`CUDA_VISIBLE_DEVICES` if you want fewer.

The parallelism is **plain DDP**. Each rank holds a complete replica of the 3.62B parameters plus
its own optimizer state; gradients are all-reduced at the accumulation boundary. No FSDP, no
ZeRO sharding, no tensor or pipeline parallelism. That is why 80 GB per GPU is a hard requirement
rather than something sharding could work around.

Scaling to a different GPU count: hold the effective batch at 256 by adjusting accumulation.

| GPUs | per-GPU batch | grad accum | Effective batch | Expected wall clock |
|---|---|---|---|---|
| 8 | 8 | 4 | 256 | ~40 h (measured) |
| 4 | 8 | 8 | 256 | ~80 h (extrapolated, not measured) |
| 2 | 8 | 16 | 256 | ~160 h (extrapolated, not measured) |

Changing per-GPU batch changes memory; changing accumulation does not. If you hit OOM, halve the
per-GPU batch and double accumulation — memory drops, the effective batch and therefore the
learning-rate schedule stay valid.

One property of DDP worth internalizing before a 40-hour run: **a single dead rank hangs the
whole job silently.** The survivors block in all-reduce, so utilization reads 100% while power
draw sits at idle. Watch power, not utilization.

---

## 6. Checkpointing

Two separate mechanisms, deliberately:

| | Local rolling | Remote archive |
|---|---|---|
| Location | `<output_dir>/checkpoints/<step:06d>/` plus a `last` symlink | `<S3_CKPT_BASE>/archive/<RUN_ID>/<step>/` |
| Contents | full training state, including optimizer | model only |
| Retention | 2 most recent, older ones pruned | all of them |
| Purpose | in-place resume after a container restart | survives instance termination; the input to checkpoint selection |
| Written | every `save_freq` steps | every `save_freq` steps |

At `save_freq=2000` over 40000 steps this produced 20 archived checkpoints, about 7.0 GiB each,
roughly 140 GiB in object storage. The size of a full local checkpoint including optimizer state
was **not recorded**.

Archived checkpoint layout — flat, three files, no nested directory:

```
40000/
  config.json             2,292 bytes    policy config
  model.safetensors       7,481,485,688 bytes
  train_config.json       6,830 bytes    the full training configuration
```

Note that `config.json` carries the policy class's default optimizer and scheduler fields
(`optimizer_lr`, `optimizer_weight_decay`, `scheduler_decay_steps`). Those were **not** the values
used, because `use_policy_training_preset` was false. `train_config.json` is the file to read.

### Resume

**Same instance, container restarted.** The trainer finds `<output_dir>/checkpoints/last` and
resumes with optimizer state and schedule position intact. Nothing to pass.

**Instance lost.** This configuration does not upload optimizer state, only weights, so a true
resume is not possible from remote storage. The fallback is a warm restart:

```bash
/opt/scripts/train-base --init-checkpoint=s3://<YOUR_BUCKET>/checkpoints/archive/<RUN_ID>/<step> ...
```

which downloads that model and uses it as the initialization instead of the stock base. Optimizer
moments and the LR schedule restart from zero, so warmup runs again and the cosine curve no longer
matches the original 40000-step schedule. It is a damage-control path, not a resume.

The trade-off was accepted knowingly: uploading full optimizer state every 2000 steps costs
bandwidth and time on every save, to insure against a host failure that is rare on non-preemptible
instances. If you run on preemptible or spot capacity, invert that decision.

---

## 7. Monitoring

```bash
docker logs -f <container>                 # step, loss, grad_norm, timings (log_freq 10)
nvidia-smi                                 # per-GPU utilization, memory, power
docker exec <container> grep '^anon ' /sys/fs/cgroup/memory.stat
aws s3 ls s3://<YOUR_BUCKET>/checkpoints/archive/<RUN_ID>/
```

Normal ranges, measured on this run:

| Signal | Normal | What a deviation means |
|---|---|---|
| `updt_s` (compute per step) | 3.4 - 3.6 s | Rising: thermal or interconnect problems |
| `data_s` (data wait per step) | ~0.002 s | Above ~0.5 s the GPUs are starving. Raise `num_workers`, or the storage is too slow |
| loss | 0.10 at start, 0.0065 at the end | See the curve in section 8 |
| grad norm | ~4.1 on step 1, ~0.058 after warmup | Spikes or NaN mean the learning rate or the data is wrong |
| GPU utilization | 90-100% on all 8 | Some ranks idle: NCCL or device visibility |
| GPU power | near board limit under load | **100% utilization at ~96 W means a hung rank**, not work |
| GPU memory | ~52-55 GB per GPU (estimate; not recorded for this run) | Near 80 GB, reduce per-GPU batch and raise accumulation |
| cgroup `anon` | plateau near 96 GB | **Monotonic growth of ~1.6 GB/step is the decoder cache leak** (section 10) |
| cgroup `file` | varies | Page cache, reclaimable, ignore it |
| archive objects | a new `<step>/` every 2000 steps | Nothing appearing means `S3_CKPT_BASE` is unset |

Separating `anon` from `file` in the cgroup accounting is the one diagnostic worth learning ahead
of time. Total RSS grows for benign reasons — page cache from reading videos — and looking at it
alone hides a real leak. `anon` is unreclaimable; if it rises monotonically, something is leaking.

The first 1000 steps decide whether the remaining 39 hours are worth paying for. Confirm step
time, `data_s`, all 8 GPUs busy, and that the first archived checkpoint appeared, then leave it
alone.

---

## 8. What the run looked like

| Phase | Duration |
|---|---|
| Image pull (cold host) | ~13 min |
| Dataset transfer | not recorded; host-location dependent |
| Smoke run | ~10 min |
| Training, 40000 steps | ~40 h |

3.4-3.55 s per step x 40000 steps is 38-40 hours. The first checkpoint lands at step 2000, about
two hours in.

Loss, as the mean over each tenth of the run:

```
0.0198  0.0113  0.0100  0.0093  0.0084  0.0077  0.0073  0.0069  0.0067  0.0065
```

Monotonically decreasing, flattening into a plateau at the end — the standard deviation over the
last 300 steps was 0.0006. That shape is the intended one for this budget: the model has stopped
extracting much from a single pass, and additional steps under the same schedule would not buy
much. Gradient norm settled near 0.058 after warmup and stayed there. No crashes, no NaN, no
corrupt checkpoints.

---

## 9. Verifying completion

Clean exit means all of these, not just the first:

1. **Process return code 0.** The wrapper propagates the trainer's exit code; the log ends with a
   completion line carrying `rc=0`.
2. **The last logged step is `40000/40000`**, with the epoch counter at 1.19. That figure is
   `steps x effective_batch / frames_in_the_dataset_you_passed`, so it will read differently if
   your corpus differs — see section 4.
3. **20 checkpoint directories in the archive**, at 2000-step intervals:
   ```bash
   aws s3 ls s3://<YOUR_BUCKET>/checkpoints/archive/<RUN_ID>/
   ```
4. **The final directory has all three files** and `model.safetensors` is 7,481,485,688 bytes.
5. **The weights load and are finite:**
   ```python
   import torch
   from safetensors import safe_open

   with safe_open("40000/model.safetensors", framework="pt") as f:
       for k in f.keys():
           t = f.get_tensor(k)
           assert torch.isfinite(t.float()).all(), k
   ```
6. **The loss curve is flat at the end**, not still descending steeply and not diverging.

Then terminate the instance. An idle 8 x A100 node bills exactly the same as a busy one.

### Selecting a checkpoint: the lowest loss is not the answer

**Do not ship the final checkpoint because it has the lowest training loss.** Training loss and
closed-loop performance on a physical robot are only loosely related. On an earlier fine-tune in
this same stack the lowest-loss checkpoint was at step 18000 while the checkpoint that actually
worked best on hardware was step 12000.

Save checkpoints across the whole mid-to-late range — which is why `save_freq` is 2000 rather than
something coarser — and choose by running each candidate on the real robot in closed loop on the
same task, same prompt, same run settings, counting successes. The mid-to-late band (roughly 30%
to 100% of total steps) is the range worth sweeping; earlier checkpoints are usually
undertrained.

For the released checkpoint this sweep has **not** been completed: only step 40000 was validated
on hardware. An earlier checkpoint may well be better.

---

## 10. Failures seen in this run

None of these were caught by the 20-step smoke test. Each appeared only in the real run.

| Symptom | Cause | Fix |
|---|---|---|
| Hang around step 70. GPUs at 100% utilization but idle power. `Too many open files` in the logs. | 8 workers x 8 ranks over 51,411 videos exceeds the default `nofile` limit of 1024; a worker dies and the surviving ranks block in all-reduce. | `--ulimit nofile=1048576:1048576` |
| `AssertionError: query timestamps violate tolerance (0.1 > 1e-4)` at step 140 in a dataloader worker | 10 fps sources whose first video PTS is 0.1 s while parquet timestamps start at 0; also `info.json fps: 30` against 10 fps video. | Align parquet timestamps to the video's real first PTS (`ts - ts[0] + pts0`) and trust the video's frame rate over metadata. In this run the 450 affected episodes were dropped instead, leaving 16,687 of 17,137. Note that fixing the timestamps alone can then push the last frame of an episode out of range under delay simulation — check both ends. |
| Anonymous memory grows ~1.6 GB per step, OOM predicted around step 1000 | LeRobot's global `VideoDecoderCache` keeps one decoder plus one open file handle per video path with no eviction. Random sampling across ~50,000 videos accumulates all of them. | Bound the cache as an LRU with a size limit, closing the file handle on eviction. Growth drops to ~0.015 GB/step and plateaus near 96 GB. `num_workers=0` does not fix it — the leak moves to the main process and the GPUs idle. Upstream LeRobot has since merged an equivalent bounded cache, but it was not in any release at the time of this run; `video_backend: pyav` also avoids the cache entirely at a decoding-speed cost. |
| A single frame fails to decode and ends the run | A few source videos have corrupt frames. | Catch the decode error and fall back to frame 0. One bad frame should not end a 40-hour job. |
| `Bus error` at startup with multiple workers | Docker's default 64 MB `/dev/shm`. | `--shm-size=64g` |
| Checkpoints never appear in object storage, training otherwise normal | `S3_CKPT_BASE` not set; the uploader disables itself silently. | Set it, and verify during the smoke run |
| Dataset transfer never finishes | Instance is geographically far from the bucket. | Re-launch in a region near the data (section 1) |

Budget for the first few hundred steps of the real run being part of the debugging loop, and watch
them. 05 §5 is on what a short test can and cannot prove.
