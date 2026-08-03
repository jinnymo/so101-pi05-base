# 05 — Troubleshooting

Problems that actually occurred while producing this checkpoint, with the diagnosis path that
led to each fix. Nothing here is hypothetical: every entry cost either GPU hours or a failed
run.

The training side covers a 40-hour full fine-tune on 8 x A100 80GB, run inside a container, over
8,595,621 frames — what remained of the 8,690,531-frame merged corpus after problem 2 below
removed 450 episodes. The inference side covers closed-loop evaluation on a physical
SO-101 with two USB cameras, LeRobot 0.4.1 and torch 2.7.1+cu126.

Software versions referenced throughout:

| Component | Version |
|---|---|
| Training / inference stack | VLASH (`https://github.com/mit-han-lab/vlash`) |
| Dataset and robot layer | LeRobot 0.4.1 |
| Video decode | torchcodec (`seek_mode="approximate"`), PyAV fallback |
| Dataset format | LeRobot v2.1 |

| # | Problem | Surfaced at | One-line symptom |
|---|---|---|---|
| 1 | DataLoader file-descriptor exhaustion | step ~70 | Run hangs, GPUs at 100% util but idle power |
| 2 | Timestamp tolerance violation | step ~140 | `query timestamps violate the tolerance (0.1 > 1e-4)` |
| 3 | Unbounded video decoder cache | step ~1000 | Host RAM exhausted, process killed |
| 4 | Corrupt source video | unpredictable | `get_frames_at` raises, run dies |
| 5 | Smoke tests miss all of the above | — | 20-step smoke passes, real run fails |
| 6 | UVC camera backend on Linux | connect time | `VIDIOC_QBUF: Bad file descriptor` |
| 7 | Camera pixel format | connect time | Format/rate negotiation fails or silently degrades |
| 8 | Calibration file format mismatch | connect time | LeRobot cannot parse the calibration JSON |
| 9 | Tokenizer path in a non-container environment | policy load | `HFValidationError` on a path that does not exist |

---

## Training

### 1. DataLoader deadlock: "Too many open files"

**Symptom**

The run stops making progress around step 70. No traceback appears on the main log for some
time. `nvidia-smi` shows six of eight GPUs at 100% utilization while drawing about 96 W, which
is idle-wait power rather than compute power. Ranks 1 and 7 are hung. Eventually a
`Too many open files` / `OSError: [Errno 24]` surfaces from a DataLoader worker.

The 100%-utilization-at-idle-power combination is the giveaway: a GPU spinning inside an NCCL
collective reports full utilization because the kernel is resident, but it is waiting, not
computing. When one rank's DataLoader worker dies, that rank never reaches the next
`all_reduce`, and every other rank blocks in the collective forever.

**When it appears**

Around step 70 of the full run, with `num_workers=8` on each of 8 DDP ranks. It does not appear
at step 0, and it does not appear in a short smoke test.

The reason for the delay is that the open-descriptor count grows as the run proceeds rather
than being fixed at startup. LeRobot's decoder cache (problem 3) holds one open file handle per
video path it has ever seen and never closes any of them. With `shuffle=True` over 51,411
videos, nearly every step opens files that were never opened before. Sixty-four worker
processes each accumulating handles cross the default limit of 1024 within a couple of minutes.

**Root cause**

`num_workers` x DDP world size exceeds the default `RLIMIT_NOFILE` of 1024. Two multipliers
compound: process count (8 workers x 8 ranks), and per-worker handle accumulation from the
unbounded decoder cache.

**Diagnosis**

Check the limit that the training process actually got, not the one on the host:

```bash
docker exec <container> bash -c 'ulimit -n'
docker exec <container> bash -c 'grep "open files" /proc/1/limits'
```

Count descriptors per process, worst first:

```bash
docker exec <container> bash -c '
for p in $(pgrep -f train); do
  echo "$(ls /proc/$p/fd 2>/dev/null | wc -l) $p"
done | sort -rn | head -20'
```

Repeat that a few minutes apart. A count that climbs monotonically is the confirmation. A count
that is high but flat means the limit is simply too low for the worker count, which is a
different (and easier) problem.

System-wide allocated / max descriptors:

```bash
cat /proc/sys/fs/file-nr
```

Confirm the idle-wait signature rather than assuming a slow step:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used --format=csv -l 5
```

If the hang needs to be pinned to a specific stack frame, dump it:

```bash
pip install py-spy
py-spy dump --pid <rank_pid>
```

A rank blocked in `torch.distributed` `all_reduce` while another is blocked in
`_MultiProcessingDataLoaderIter._try_get_data` confirms the shape of the failure.

**Fix**

Raise the limit for the training process. Under Docker:

```bash
docker run --gpus all \
  --ulimit nofile=1048576:1048576 \
  --shm-size=64g \
  ... <image> ...
```

Without Docker, raise it in the launching shell or unit file:

```bash
ulimit -n 1048576          # shell, requires a sufficient hard limit
# systemd unit:  LimitNOFILE=1048576
# /etc/security/limits.conf:  <user> soft nofile 1048576
```

**Prevention**

- Put `--ulimit nofile=1048576:1048576` in the launch template, not in a runbook step that can
  be skipped. It costs nothing when unnecessary.
- Bound the decoder cache (problem 3). With an LRU cap of 64 the open-handle count per worker
  becomes constant instead of growing, which removes the accumulation multiplier entirely.
- `--shm-size=64g` belongs in the same launch line. The default 64 MB `/dev/shm` in a container
  makes `num_workers=8` fail with `Bus error` when the workers pass tensors back. This is a
  different failure from the descriptor limit but hits the same launch command, and both are
  cheaper to set preemptively than to diagnose at hour 12.
- After the container starts, assert the limit before spending GPU time:
  `[ "$(ulimit -n)" -ge 65536 ] || { echo "nofile too low"; exit 1; }`

---

### 2. Timestamp tolerance violation on 10 fps sources

**Symptom**

A DataLoader worker raises, and the run dies:

```
AssertionError: One or several query timestamps unexpectedly violate the tolerance
(tensor([0.1000]) > tolerance_s=0.0001).
It means that the closest frame that can be loaded from the video is too far away in time.
...
video: .../videos/chunk-XXX/observation.images.base_0_rgb/episode_XXXXXX.mp4
```

**When it appears**

Around step 140 on one rank of the full run. Not during the pre-launch dataset check, and not
during the 20-step smoke test.

**Root cause**

LeRobot pairs each frame's parquet `timestamp` column with a frame decoded from the
corresponding video, and asserts that the decoded presentation timestamp is within
`tolerance_s` of the requested one. In `decode_video_frames_torchcodec` the request is
converted to a frame index first:

```python
frame_indices = [round(ts * average_fps) for ts in timestamps]
frames_batch = decoder.get_frames_at(indices=frame_indices)
```

and the returned `pts_seconds` are then compared against the query.

Two independent defects in the source data broke that pairing:

1. Videos from 10 fps sources have a **first PTS of 0.1 s, not 0**. The parquet timestamps for
   those episodes started at 0.0. Every query was therefore off by exactly one frame interval,
   0.1 s, which is three orders of magnitude above the 1e-4 tolerance.
2. `meta/info.json` declared `fps: 30` for episodes whose videos are actually 10 fps. Since the
   index conversion above multiplies by the decoder's `average_fps`, a metadata/stream
   disagreement pushes the computed index away from the intended frame independently of defect 1.

Two further findings from the investigation are worth recording because they redirect blame:

- PyAV's first-PTS read was correct. The video files were not lying; the parquet was.
- The immediate cause of the parquet being wrong was an **earlier fix**. The dataset merge step
  wrote `ts = ts - ts[0] + pts0`, which is correct: it anchors parquet timestamps to the video's
  real first PTS. A later repair pass over episodes 0-99 rewrote those timestamps as `i / 10`,
  starting at 0.0, and silently undid the anchoring for the episodes it touched. A repair
  applied to a subset became the defect discovered 140 steps into a run whose GPU rental was
  the largest single cost in the project. The invoiced amount is not recorded; 01 has what is
  known about the cost, which is a wall clock and a rate range rather than a figure.

**The first fix attempt failed, and the reason matters**

Realigning the affected parquet timestamps back to the video PTS (`ts - ts[0] + pts0`) passed
the tolerance check and then failed differently:

```
Invalid frame index=321 for streamIndex
```

The training loop simulates inference delay by shifting the queried action window forward
(`max_delay_steps=8`). The dataset-level index is clamped to `ep_end - 1`, but video access is
by timestamp, not by that clamped index. With every timestamp shifted by +0.1 s, the last
queried timestamp of an episode maps to a frame index one past the end of the video stream.

So for these episodes both options fail: `ts = 0.0` violates the tolerance, `ts = 0.1` runs off
the end of the video. There is no parquet-only fix. The video itself has to be re-muxed to
start at PTS 0, or the episodes have to go.

**Decision taken**

The 450 affected episodes (2.6% of 17,137, 94,910 frames) were excluded and the run proceeded on
the 16,687 episodes whose videos start at PTS 0.0. Reasons, in order: 30 fps episodes were
structurally safe and had already proven themselves for 140 steps; the excluded portion is small
and simulation-heavy, so the effect on the base distribution is limited; and re-muxing 1,350
videos is a video-level operation with its own failure modes and its own idle GPU bill.

All 450 come from five source datasets:

| Source repository | Episodes |
|---|---|
| `CoRL2026-CSI/SO101-teleop_stack_RGBblock_on_bluedish_150epi_10fps` | 150 |
| `CoRL2026-CSI/IsaacLab-SO101-PullCube-100epi-10fps-appendix` | 100 |
| `anvilbot-patrickhhh/SO101_relocate_cube_2cams_record_2` | 100 |
| `anvilbot-patrickhhh/SO101_PickAndPlace_front_wrist` | 50 |
| `anvilbot-patrickhhh/SO101_PickAndPlace_3cams` | 50 |

The filter that selected them ran on measured frame interval rather than on source name, since
the merge has already erased per-source identity by this point. 02, under "Producing the corpus
that was actually trained", has the full procedure including the re-indexing that has to follow.
Note there that normalization statistics were **not** recomputed after the drop.

**Diagnosis**

Inspect one episode's parquet timestamps:

```bash
python - "$PARQUET" <<'PY'
import sys
import numpy as np, pyarrow.parquet as pq
t = pq.read_table(sys.argv[1], columns=["timestamp"])["timestamp"].to_numpy()
d = np.diff(t)
print(f"n={len(t)} ts0={t[0]:.6f} monotonic={bool((d > 0).all())} "
      f"median_dt={np.median(d):.6f} implied_fps={1 / np.median(d):.2f}")
PY
```

Read the video's actual first frame PTS and its real frame rate:

```bash
ffprobe -v error -select_streams v:0 -read_intervals '%+#1' \
        -show_entries frame=pts_time -of csv=p=0 "$VIDEO"

ffprobe -v error -select_streams v:0 \
        -show_entries stream=r_frame_rate,avg_frame_rate,nb_frames,duration \
        -of default=noprint_wrappers=1 "$VIDEO"
```

Compare with what the dataset claims:

```bash
jq '{fps, total_episodes, total_frames}' "$DATASET/meta/info.json"
```

Scan the whole dataset before launching. This is the check that would have caught it:

```python
#!/usr/bin/env python
"""Pre-flight check on a LeRobot v2.1 dataset.

Flags: parquet timestamps not starting at 0 relative to the video's first PTS,
non-monotonic timestamps, parquet rate disagreeing with info.json, and videos
whose first frame cannot be read.

Usage: python check_dataset.py /path/to/dataset
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path(sys.argv[1])
declared_fps = json.loads((root / "meta" / "info.json").read_text())["fps"]
flagged = 0

for pfile in sorted(root.glob("data/*/*.parquet")):
    ts = pq.read_table(pfile, columns=["timestamp"])["timestamp"].to_numpy()
    problems = []

    if len(ts) > 1:
        d = np.diff(ts)
        if not (d > 0).all():
            problems.append("non-monotonic timestamps")
        implied = 1.0 / float(np.median(d))
        if abs(implied - declared_fps) > 0.5:
            problems.append(f"parquet rate {implied:.2f} != info.json {declared_fps}")

    for vid in sorted(root.glob(f"videos/*/*/{pfile.stem}.mp4")):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-read_intervals", "%+#1", "-show_entries", "frame=pts_time",
             "-of", "csv=p=0", str(vid)],
            capture_output=True, text=True).stdout.strip()
        if not out:
            problems.append(f"{vid.parent.name}: first frame not decodable")
            continue
        first_pts = float(out.splitlines()[0])
        if abs(first_pts - float(ts[0])) > 1e-4:
            problems.append(
                f"{vid.parent.name}: video first pts {first_pts:.6f} != ts0 {ts[0]:.6f}")

    if problems:
        flagged += 1
        print(pfile.name, "|", "; ".join(problems))

print(f"{flagged} episode(s) flagged out of {len(list(root.glob('data/*/*.parquet')))}")
```

On a dataset this size that is one `ffprobe` per video (51,411 of them), so run the video half
in parallel rather than serially:

```bash
find "$DATASET/videos" -name '*.mp4' -print0 \
  | xargs -0 -P 16 -I{} sh -c \
    'printf "%s %s\n" "{}" "$(ffprobe -v error -select_streams v:0 -read_intervals "%+#1" \
       -show_entries frame=pts_time -of csv=p=0 "{}" | head -1)"' \
  | awk '$2 != "0.000000" {print}'
```

**Fix**

Pick one, in order of preference:

1. **Re-mux the offending videos so the stream starts at PTS 0**, then set parquet
   `ts = ts - ts[0]`. This is the only option that keeps the data and works with delay
   simulation. `ffmpeg -i in.mp4 -c copy -reset_timestamps 1 -avoid_negative_ts make_zero out.mp4`
   is the starting point; verify the result with the ffprobe command above before trusting it.
2. **Drop the affected episodes** and re-index (`episode_index` and the global `index` column
   must stay contiguous). This is what was done here.
3. **Never trust `info.json` fps over the stream.** If a subset has a different real frame rate,
   either separate it into its own dataset or fix the metadata; do not let one declared rate
   cover both.

If you drop episodes, recompute normalization statistics as well, or accept that the stats
include a distribution slice the training run never sees. The stats were carried over unchanged
here, which is a known minor inconsistency.

**Prevention**

- Run the full-dataset check above before every launch, not once at pipeline-build time. It
  costs minutes of CPU against tens of hours of GPU.
- **Re-run the check after re-running any part of the data pipeline.** The merge step's correct
  output was overwritten by a later partial repair, and nothing detected that until step 140.
  Any script that rewrites timestamps invalidates a previous check.
- Assert timestamp-origin agreement inside the dataset builder itself, so a bad episode cannot
  be written in the first place.

---

### 3. RAM leak to OOM: unbounded video decoder cache

**Symptom**

Anonymous memory grows monotonically at roughly 1.6 GB per step. Page cache stays flat. With
swap disabled, the host's 1.7 TB of RAM is exhausted at approximately step 1000, about 85
minutes in, and the process is killed.

**When it appears**

From step 0. The growth is linear and steady; only the OOM is late. A 20-step smoke test
consumes about 30 GB, which on a large host is indistinguishable from normal startup.

**Root cause**

`lerobot/datasets/video_utils.py` holds a module-level cache with no eviction policy:

```python
_default_decoder_cache = VideoDecoderCache()   # one per process

class VideoDecoderCache:
    def __init__(self):
        self._cache: dict = {}                 # no cap

    def get_decoder(self, video_path):
        if video_path not in self._cache:
            file_handle = fsspec.open(video_path).__enter__()
            decoder = VideoDecoder(file_handle, seek_mode="approximate")
            self._cache[video_path] = (decoder, file_handle)   # never evicted
        return self._cache[video_path][0]
```

`decode_video_frames_torchcodec` uses this cache by default. A torchcodec `VideoDecoder` opened
with `seek_mode="approximate"` retains a frame index and internal buffers, so each cached entry
is substantial. With 16,687 episodes x 3 camera slots, roughly 50,000 distinct video paths, and
`shuffle=True`, nearly every step introduces paths that will never be requested again, and the
dictionary grows without bound. The open `file_handle` stored alongside each decoder is also
never closed, which is the accumulation behind problem 1.

This is a known upstream problem, unresolved at the time of the run:

- https://github.com/huggingface/lerobot/issues/2371
- https://github.com/huggingface/lerobot/issues/3712

**Diagnosis, including the wrong turns**

The wrong turns took longer than the fix, so they are worth reproducing.

1. *Extrapolating from a ten-minute window.* An early upward slope was treated as proof of a
   leak before enough samples existed to distinguish a leak from warm-up allocation. Premature,
   even though it happened to be right.
2. *Misreading a drop as a sawtooth.* A measurement showed anonymous memory falling from 775 GB
   to 334 GB, which was read as normal allocator sawtooth. It was not: a `docker rm -f` had
   killed the container between samples, and the "recovery" was a fresh process. **Verify that
   consecutive memory samples come from the same process instance** before interpreting any
   drop.
3. *Reasoning from object types.* `hf_dataset` and `episodes` are `datasets.Dataset` objects,
   which are Arrow-backed and memory-mapped, so the dataset "could not" be the leak. Wrong
   conclusion: an Arrow-backed store says nothing about the layers built on top of it, and the
   leak was in the video decode layer, not in the table.
4. **The decisive measurement: splitting cgroup `anon` from `file`.** Total RSS conflates page
   cache, which is reclaimable, with anonymous memory, which is not. With swap at 0, monotonic
   `anon` growth and flat `file` means an unavoidable OOM, not caching pressure. Everything
   before this step was speculation.
5. *Testing `num_workers=0`.* This eliminated copy-on-write growth across worker forks, and the
   main process still grew at 1.6 GB per step. That result excluded fork/COW as the mechanism
   and pointed at a genuine allocation. It is a diagnostic, not a fix; see below.
6. Following `__getitem__` -> `_query_videos` -> `decode_video_frames_torchcodec` ->
   `_default_decoder_cache` identified the unbounded dictionary.

**Diagnostic commands**

Split anonymous from file-backed memory (cgroup v2):

```bash
docker exec <container> grep -E '^(anon|file|slab) ' /sys/fs/cgroup/memory.stat
docker exec <container> cat /sys/fs/cgroup/memory.current
```

On cgroup v1 the equivalents are `rss` and `cache` in
`/sys/fs/cgroup/memory/memory.stat`.

Sample it over time, and record the container's PID with every sample so that a restart cannot
be mistaken for a recovery:

```bash
C=<container>
while sleep 60; do
  printf '%s pid=%s anon=%s file=%s\n' \
    "$(date +%H:%M:%S)" \
    "$(docker inspect -f '{{.State.Pid}}' "$C")" \
    "$(docker exec "$C" awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)" \
    "$(docker exec "$C" awk '/^file /{print $2}' /sys/fs/cgroup/memory.stat)"
done | tee anon.log
```

Read it as: `anon` rising and `file` flat is a leak. Both rising with `anon` flat is page cache
doing its job and can be ignored. `anon` rising and then falling is either a real sawtooth or a
process restart, which is what the `pid=` field settles.

Confirm swap really is zero, since with swap the same growth degrades performance instead of
killing the run:

```bash
free -g
cat /proc/sys/vm/swappiness
```

Confirm the cache is the thing growing, from inside the training process:

```python
from lerobot.datasets.video_utils import _default_decoder_cache
print(_default_decoder_cache.size())    # after the LRU patch below
```

**Fix**

Bound the cache as an LRU and close the file handle on eviction. Full patched method:

```python
class VideoDecoderCache:
    """Thread-safe cache for video decoders to avoid expensive re-initialization."""

    def __init__(self):
        import collections, os
        self._cache = collections.OrderedDict()
        self._maxsize = int(os.environ.get("VLASH_DECODER_CACHE_MAX", "64"))
        self._lock = Lock()

    def get_decoder(self, video_path: str):
        if importlib.util.find_spec("torchcodec"):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError("torchcodec is required but not available.")

        video_path = str(video_path)

        with self._lock:
            if video_path not in self._cache:
                file_handle = fsspec.open(video_path).__enter__()
                decoder = VideoDecoder(file_handle, seek_mode="approximate")
                self._cache[video_path] = (decoder, file_handle)
                self._cache.move_to_end(video_path)
                while len(self._cache) > self._maxsize:
                    _k, (_od, _ofh) = self._cache.popitem(last=False)
                    try:
                        _ofh.close()
                    except Exception:
                        pass
            else:
                self._cache.move_to_end(video_path)
            return self._cache[video_path][0]
```

This is low risk with respect to training results. A re-created decoder returns the same frames
for the same video and timestamps; the cache is purely a performance optimization, so evicting
from it cannot change the data the model sees.

Measured effect on the run:

| Metric | Before | After |
|---|---|---|
| `anon` growth | ~1.6 GB/step, OOM near step 1000 | ~0.015 GB/step, plateau near 96 GB |
| `data_s` | 1.05 with `num_workers=0` | 0.002 with `num_workers=8` |
| `updt_s` | 3.6 s | 3.55 s |
| Outcome | cannot finish | 40000/40000 steps, about 40 h |

96 GB of 1.7 TB is 5.6%, and it stops climbing. Tune `VLASH_DECODER_CACHE_MAX` if needed:
raise it if `data_s` climbs because eviction is too aggressive, lower it if memory is tight.
With shuffled sampling over 50,000 videos the hit rate is low regardless; the cap exists to stop
the leak, not to speed anything up.

**Why `num_workers=0` is not the fix**

It is the most commonly suggested response to a DataLoader memory problem and it is wrong here:

- The leak is in a module-level object, so it exists in whatever process does the decoding. With
  zero workers, the main process leaks at the same 1.6 GB per step.
- It removes prefetching. GPU utilization oscillates between 0% and 100%, `data_s` goes from
  0.002 to 1.05, and the run becomes roughly 1.5x slower. On a 40-hour run at 8-GPU rates that
  is a large amount of money spent to not fix the problem.

Keeping `num_workers=8` and bounding the cache gives both prefetching and zero growth.

**Prevention**

- Bake the patch into the training image (`COPY` the patched `video_utils.py` over the installed
  one during build) rather than bind-mounting it at run time. A `-v` mount is a temporary
  measure that will not survive being handed to someone else.
- Sample `anon` for the first 30 minutes of any long run and require a flat slope before walking
  away. Two data points an hour apart are enough to catch a 1.6 GB/step leak.
- Treat "reduce `num_workers`" as a hypothesis test, never as a resolution.

---

### 4. Corrupt source video

**Symptom**

`decoder.get_frames_at(indices=...)` raises for a small number of videos in the corpus. Without
handling, a single unreadable frame ends the entire run.

**When it appears**

Unpredictable. It depends on which episode the shuffled sampler reaches, so it can occur at any
point in a multi-day run, including near the end. In a corpus assembled from hundreds of
independently uploaded source datasets, some fraction of videos being truncated or partially
unreadable should be assumed rather than hoped against.

**Root cause**

Third-party source videos with truncated streams or undecodable frames. Not a bug in the
training stack.

**Diagnosis**

Scan for undecodable files offline, before training:

```bash
find "$DATASET/videos" -name '*.mp4' -print0 \
  | xargs -0 -P 16 -I{} sh -c \
    'ffmpeg -v error -i "{}" -f null - 2>&1 | grep -q . && echo "BAD {}"'
```

During training, count how often the fallback fires:

```bash
grep -c '\[corrupt-video\]' train.log
grep -o '\[corrupt-video\] [^ ]*' train.log | sort -u | head
```

A handful of lines across millions of samples is noise. Hundreds means a systematic problem in
the dataset build (a broken encode step, an interrupted download) and should be fixed at the
source rather than absorbed.

**Fix**

Catch the decode failure, log it with the path and requested indices, and fall back to frame 0
repeated to the requested length:

```python
try:
    frames_batch = decoder.get_frames_at(indices=frame_indices)
except Exception as e:
    logging.warning(f"[corrupt-video] {video_path} idx={frame_indices}: {e} -> fallback frame0")
    fb0 = decoder.get_frames_at(indices=[0])
    return torch.stack([fb0.data[0] for _ in timestamps])
```

The trade-off is explicit: that sample is wrong, since every frame in the returned stack is
identical to the video's first frame. Against 8.69M frames the contamination is negligible, and
against losing a 40-hour run it is obviously the right side of the trade. The logged warning is
what keeps it from becoming silent data corruption, so do not downgrade it to `debug`.

**Prevention**

- Pre-scan with `ffmpeg -v error -f null -` as part of dataset validation.
- Keep the warning in the log and check its count when the run finishes. A fallback that fires
  and is never read is a silent corruption path.

---

### 5. What a smoke test does not catch

All four training problems above passed a 20-step smoke test on the same dataset, the same
image and the same hardware, then broke the real run. That is not bad luck; it follows from what
each failure depends on.

**Three classes of failure a short run cannot expose**

| Class | Depends on | Examples above |
|---|---|---|
| Resource accumulation | elapsed steps | 1 (descriptors), 3 (memory) |
| Rare-subset data faults | samples drawn | 2 (timestamps), 4 (corrupt video) |
| Post-warm-up behavior | schedule position | learning-rate schedule, loss plateau shape |

A 20-step run accumulates 20 steps' worth of descriptors and about 30 GB of leaked memory,
neither of which is visible on a large host. It draws a small share of the corpus, so an episode
class occupying a few percent of the dataset is quite likely to be missed entirely — the first
timestamp violation in the real run came at step 140, and the first corrupt video was later
still.

**What the smoke test is still for**

It does verify everything that fails immediately, and those checks are worth keeping:

- image pull, credentials, dataset download throughput
- GPU count, NVLink topology, NCCL initialization
- first loss finite, gradient norm finite
- peak VRAM (determined at the first backward pass and optimizer-state allocation, so a short
  run does measure it correctly)
- checkpoint upload path actually receiving objects — the most valuable single check, since a
  missing output-path variable fails silently and is discovered only when the instance is
  terminated

**What to add**

1. **An offline dataset check.** Problems 2 and 4 are data defects and cost zero GPU time to
   find. The script in problem 2 catches both. Run it after every pipeline change.
2. **A soak run of several hundred steps** before committing to the full run. Both step-70 and
   step-140 failures are inside a 300-step window. At about 3.5 s/step this is roughly 20
   minutes of GPU time against a 40-hour commitment.
3. **Memory sampling with a pass/fail threshold** during the soak, using the loop from problem 3.
   Require the `anon` slope to be flat, and treat a positive slope as a stop condition rather
   than something to watch.
4. **`data_s` versus `updt_s` in the training log.** `data_s` near zero means prefetch is keeping
   up. `data_s` comparable to `updt_s` means the GPUs are waiting on the input pipeline, which
   is a cost problem long before it is a correctness problem.
5. **Assert environment invariants at container start**: `ulimit -n`, `/dev/shm` size, presence
   of the output path variable, dataset path resolvable. Each of these has a one-line check and
   each has a failure mode measured in hours.

The general rule: a smoke test proves the path works. It does not prove the run works. The gap
between them is filled by an offline data check plus a soak long enough for accumulation to
become visible.

---

## Real-robot inference

### 6. UVC camera backend on Linux

**Symptom**

The camera opens, but property configuration silently fails and reading raises:

```
cap.set(cv2.CAP_PROP_FOURCC, ...)        -> False
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   -> False
VIDIOC_QBUF: Bad file descriptor
```

**When it appears**

At robot/camera connect time, on Linux, with certain UVC cameras. Observed with an Innomaker UVC
module; the failure is a property of the backend/device combination rather than of one vendor.

**Root cause**

LeRobot's `get_cv2_backend()` returns `cv2.CAP_ANY` on Linux. `CAP_ANY` lets OpenCV pick, and
the backend it picks for these devices does not accept the `set()` calls LeRobot makes.
Requesting `cv2.CAP_V4L2` explicitly makes the same calls return `True` and reads succeed.

**Diagnosis**

Confirm the device exists and what it supports, independently of OpenCV:

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --get-fmt-video
```

Confirm the OpenCV build has V4L2 at all:

```bash
python -c "import cv2; print([l for l in cv2.getBuildInformation().splitlines() if 'V4L' in l])"
```

A/B the two backends directly. This is the test that isolates the problem in about ten seconds:

```python
import cv2

for name, backend in [("CAP_ANY", cv2.CAP_ANY), ("CAP_V4L2", cv2.CAP_V4L2)]:
    cap = cv2.VideoCapture("/dev/video0", backend)
    ok_fourcc = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    ok_w = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    ok_h = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok_read, frame = cap.read()
    print(f"{name:9s} opened={cap.isOpened()} set(fourcc,w,h)=({ok_fourcc},{ok_w},{ok_h}) "
          f"read={ok_read} shape={None if frame is None else frame.shape}")
    cap.release()
```

`CAP_ANY` returning `False` on the `set()` calls while `CAP_V4L2` returns `True` is the
confirmation.

**Fix**

Rebind `get_cv2_backend` to a function returning `cv2.CAP_V4L2` at import time from a launcher
script, rather than editing `site-packages`. Both `lerobot.cameras.utils` and the module-level
import inside `lerobot.cameras.opencv.camera_opencv` need rebinding, because the camera module
imported the name directly, and both have to happen before `vlash` is imported. That launcher
ships as `inference/v4l2_launch.py`; 06 §5.4 lists it.

Invoke it wherever the CLI would have been invoked:

```bash
python inference/v4l2_launch.py run inference/eval.yaml --policy.path=/path/to/checkpoint
```

**Prevention**

- Keep the launcher in version control next to the config. A `site-packages` edit disappears on
  the next `pip install`, does not travel to another machine, and is invisible to anyone reading
  the repository.
- Patch before the first import of anything that pulls the symbol in, and patch every module
  that imported the name directly. Patching only `lerobot.cameras.utils` is not sufficient.
- **Identify which physical camera is on which device node by capturing a frame, not by trusting
  enumeration order.** Node numbering changes across reboots and re-plugs. Swapping the overhead
  and wrist views produces a policy that runs and behaves badly, with no error anywhere:

```python
import cv2

for node in ("/dev/video0", "/dev/video2"):
    cap = cv2.VideoCapture(node, cv2.CAP_V4L2)
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(f"/tmp/cam{node[-1]}.png", frame)
        print(node, "->", f"/tmp/cam{node[-1]}.png")
    cap.release()
```

---

### 7. Camera pixel format must be pinned to MJPG

**Symptom**

Format or frame-rate negotiation fails at connect time, or succeeds and silently delivers fewer
frames per second than requested.

**Root cause**

USB cameras commonly default to an uncompressed format such as YUYV. At 640x480 and 30 fps the
uncompressed stream can exceed the available USB bandwidth, so the driver negotiates a lower
rate or refuses the combination. The camera used here lists MJPG as its primary format, and
requesting it explicitly makes the requested resolution and rate achievable.

The silent-degradation case is the more dangerous one. The policy was trained on data collected
at 30 fps, and `fps` in the run config sets the control loop rate. If the camera actually
delivers 15 fps while the loop runs at 30, observations are stale relative to the action
timeline the model learned, and the failure looks like poor policy behavior rather than a
configuration error.

**Diagnosis**

Ask the device what it actually supports, and at what rate per format:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

Ask what was actually negotiated after opening:

```bash
v4l2-ctl -d /dev/video0 --get-fmt-video
```

Or from OpenCV, after `set()`, read the properties back — `set()` returning `True` does not mean
the request was honored:

```python
import cv2
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print("fourcc:", "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)),
      "size:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
      "fps:", cap.get(cv2.CAP_PROP_FPS))
cap.release()
```

Measure the delivered rate rather than the reported one:

```python
import time, cv2
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
n, t0 = 0, time.time()
while time.time() - t0 < 5:
    if cap.read()[0]:
        n += 1
print(f"delivered {n / (time.time() - t0):.1f} fps")
cap.release()
```

**Fix**

Pin the format in the camera config alongside resolution and rate:

```yaml
robot:
  cameras:
    base_0_rgb:
      type: opencv
      index_or_path: /dev/video0
      fourcc: MJPG
      width: 640
      height: 480
      fps: 30
```

**Prevention**

- Always specify `fourcc`, `width`, `height` and `fps` explicitly. Defaults differ per camera,
  per driver and per kernel version.
- Verify the delivered rate once per hardware setup with the loop above, and keep `fps` in the
  run config equal to the rate the training data was collected at.

---

### 8. Calibration file format mismatch

**Symptom**

LeRobot cannot parse the robot's calibration JSON and fails while connecting to the arm.

**Root cause**

LeRobot expects a calibration file whose top-level keys are motor names, each mapping to that
motor's calibration record:

```json
{
  "shoulder_pan":  { "id": 1, "drive_mode": 0, "homing_offset": ..., "range_min": ..., "range_max": ... },
  "shoulder_lift": { ... },
  "elbow_flex":    { ... },
  "wrist_flex":    { ... },
  "wrist_roll":    { ... },
  "gripper":       { ... }
}
```

The file in use carried an additional top-level key holding toolchain metadata (creation
timestamp, hardware identifier, a configuration hash, a robot identifier) next to the motor
entries. LeRobot's parsing treats every top-level key as a motor and does not tolerate the extra
entry.

**Diagnosis**

Inspect the top-level keys and compare against the arm's motor names:

```bash
jq -r 'keys[]' calibration.json
```

Anything that is not a motor name is the problem. In practice the offending keys are prefixed
with an underscore by convention:

```bash
jq -r 'keys[] | select(startswith("_"))' calibration.json
```

Verify each remaining entry has the expected shape:

```bash
jq -r 'to_entries[] | "\(.key): \(.value | keys | join(","))"' calibration.json
```

**Fix**

Generate a cleaned copy in a separate calibration directory and point the config at it. Do not
edit the original in place; another tool owns it.

```bash
SRC=~/.robotcal/so101/my_follower.json
DST=~/.robotcal/so101_lerobot/my_follower.json
mkdir -p "$(dirname "$DST")"
jq 'with_entries(select(.key | startswith("_") | not))' "$SRC" > "$DST"
jq -r 'keys | join(" ")' "$DST"      # should list only motor names
```

The file's basename must equal `robot.id`, and `calibration_dir` must point at the directory
containing it:

```yaml
robot:
  type: so101_follower
  port: /dev/ttyACM0
  id: my_follower
  calibration_dir: /home/<user>/.robotcal/so101_lerobot
```

On the first connection LeRobot may report a mismatch between the file's values and what the
motors currently hold, and prompt for confirmation. Accepting the prompt uses the file and
writes those values to the motors, which is the intended behavior when the file is the source of
truth.

**Prevention**

- Treat the LeRobot-facing calibration as a generated artifact, produced by a script from the
  authoritative file. Regenerate it whenever the source changes; never hand-edit the copy, or
  the two will silently diverge.
- Keep the conversion in the repository next to the run config so that the derived file can
  always be rebuilt.
- Verify the arm's motor names against the file's keys before the first run rather than after a
  connection failure.

---

### 9. Tokenizer path outside the training container

**Symptom**

Loading the policy fails with an `HFValidationError` referring to a path that does not exist on
the machine.

**When it appears**

The first time the checkpoint is loaded outside the container it was trained in.

**Root cause**

The policy resolves its PaliGemma tokenizer from a path that the training image provided. On a
workstation that path is absent, and the resolver treats the missing local path as a Hub
repository identifier, which then fails validation.

**Fix**

Point the loader at a local tokenizer directory and run offline:

```bash
export VLASH_PALIGEMMA_PATH=/path/to/paligemma_tokenizer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Alternatively, let it resolve from the Hub. `google/paligemma-3b-pt-224` is a gated repository:
accept its terms on the Hub and run `hf auth login` before the first load, or
pre-download it and set the path above.

**Prevention**

Set these variables in the run script rather than expecting them from the environment, so the
same script works on the training host and on a workstation.

---

## Quick reference

```bash
# descriptor limit and per-process usage
ulimit -n
for p in $(pgrep -f train); do echo "$(ls /proc/$p/fd | wc -l) $p"; done | sort -rn | head

# anonymous vs page-cache memory (cgroup v2)
grep -E '^(anon|file) ' /sys/fs/cgroup/memory.stat

# video first PTS and real frame rate
ffprobe -v error -select_streams v:0 -read_intervals '%+#1' \
        -show_entries frame=pts_time -of csv=p=0 video.mp4
ffprobe -v error -select_streams v:0 \
        -show_entries stream=r_frame_rate,avg_frame_rate,nb_frames -of default=nw=1 video.mp4

# parquet timestamp origin
python -c "import sys,pyarrow.parquet as pq; t=pq.read_table(sys.argv[1])['timestamp'].to_numpy(); print(t[0], t[1]-t[0])" ep.parquet

# undecodable videos
ffmpeg -v error -i video.mp4 -f null -

# camera capabilities and negotiated format
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --get-fmt-video

# calibration keys
jq -r 'keys[]' calibration.json
```

Launch settings that prevent problems 1 and 3 rather than diagnosing them:

```bash
docker run --gpus all \
  --ulimit nofile=1048576:1048576 \
  --shm-size=64g \
  -e VLASH_DECODER_CACHE_MAX=64 \
  ... <image> ...
```
