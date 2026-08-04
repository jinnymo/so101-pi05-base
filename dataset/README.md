---
license:
- apache-2.0
- mit
task_categories:
- robotics
tags:
- lerobot
- so-101
- so-100
- imitation-learning
- manipulation
- robot-learning
size_categories:
- 1M<n<10M
---

# SO-101 / SO-100 unified manipulation dataset

A single LeRobot dataset built from 150 public SO-101 / SO-100 teleoperation
datasets collected from the Hugging Face Hub, plus 6 recorded by the author.
The sources were screened for action-space conformity, deduplicated, and
rewritten into one schema with a fixed three-camera layout.

> **[github.com/jinnymo/so101-pi05-base](https://github.com/jinnymo/so101-pi05-base)** — the
> scripts that built this, stage by stage: the Hub crawl, the screening criteria, the
> action-convention check, the camera-slot mapping, the deduplication, and the merge. The
> intermediate artifacts ship with them, so the funnel below can be checked rather than trusted.

| | |
|---|---|
| Source datasets | 156 |
| Episodes | 13,969 |
| Frames | 7,272,752 |
| Robot | SO-101 / SO-100 single arm, 6 DoF, follower frame |
| Format | LeRobot, codebase version v2.1 |
| Camera slots | 3 fixed slots with per-slot validity masks |
| Frame rate | `meta/info.json` declares 30 fps globally; 450 episodes from five sources are actually 10 fps and fail LeRobot's timestamp tolerance check as shipped (see Usage) |

Per-source provenance, licenses and episode counts are in
[ATTRIBUTION.md](./ATTRIBUTION.md).

## What this is

Public SO-101 datasets are individually small and mutually incompatible: they
use different camera key names, different numbers of cameras, different
resolutions, both LeRobot v2.1 and v3.0 layouts, and a mix of leader-frame and
follower-frame action recordings. This dataset is the result of normalizing
those differences so that the whole collection can be loaded as one repository
and used for pre-training a single policy.

The normalization is deliberately coarse. Camera viewpoints are reduced to two
semantic classes (wrist-mounted or not) rather than mapped precisely, and
per-robot calibration offsets are left in place. The goal is broad coverage of
SO-101 visuomotor data, not a curated benchmark.

## Provenance

```
Hugging Face Hub crawl                                    11,270 repositories
  SO-101 / SO-100 metadata confirmed                       1,724
  >= 50 episodes, camera blacklist, simulation split         408
  action-space match against a reference recording           307
  downloaded successfully                                    307  (305 automatic, 2 via git-lfs)
  precision analysis and quality exclusions                  296  (11 removed)
  camera classification and manual duplicate removal         205  (91 removed)
  datasets recorded by the author added                      211  (205 + 6)
  action-fingerprint deduplication                           181  (30 removed)
  sources with a declared upstream license                   156  <- this release
```

The action-space match compares each candidate against a reference SO-101
recording on action dimensionality, joint names, units, and the offset between
`action` and `observation.state`. That offset separates follower-frame
recordings (small offset, the target is close to the achieved position) from
leader-frame recordings (systematic offset). Leader-frame and bimanual
(12-dimensional) datasets were rejected.

### Deduplication

Public SO-101 datasets contain a substantial amount of duplication:
cross-author re-uploads, cumulative repositories that contain an earlier
repository plus more episodes, and train/validation splits of the same
recording published as separate repositories. Deduplication by repository name
does not catch any of these.

Each dataset is instead reduced to a fingerprint: the order-independent
multiset of `(episode length, per-episode mean action)` over its episodes.
Re-uploads and cumulative repositories keep the same trajectories byte for
byte, so their fingerprints are exactly equal or in a strict subset relation.
Comparison runs across all datasets, ignoring the author, so cross-author
re-uploads are caught as well.

- equal fingerprints: one dataset kept, the rest dropped
- strict subset: the subset is dropped, the largest superset is kept

This removed 30 of 211 datasets. Episode length alone produced false positives
(unrelated datasets with matching length profiles), which is why the mean
action is part of the key.

### The six sources recorded by the author

Six of the 156 sources (332 episodes) were recorded by the author rather than
crawled from the Hub. They have no upstream repository, so the Hub crawl above
cannot reach them and nobody else can obtain them from the Hub. They are
distributed here instead:

| Source key | Episodes |
|---|---|
| `self/stack_cube_normalized` | 149 |
| `self/skill_eraser_move_v2` | 90 |
| `self/skill_earser_move_v3_followercal` | 50 |
| `self/pick_place_blue_pen_v1` | 18 |
| `self/pickandplace_greencube_whitecup` | 14 |
| `self/pickandplace_bluecube_whitecup` | 11 |

To rerun the pipeline with the same 156 sources, take these six from this
release rather than looking for them on the Hub. `meta/sources.json` gives each
one its global episode range, and every entry in `meta/episodes.jsonl` carries
its `source_dataset`, so the episodes belonging to each source can be selected
directly; place each extracted source under `self/<name>` in the pipeline
workspace. The remaining 150 sources are Hub repositories and can be downloaded
from the `repo_id` recorded for each of them.

What comes back out of this release is the post-merge form of those recordings,
not the originals: three fixed camera slots, mask columns, global episode and
frame indices, and black placeholder video where a slot is empty. All six were
recorded with a top camera and a wrist camera except
`self/skill_eraser_move_v2`, which has the wrist camera only, so every one of
them has at least one placeholder slot. Re-running the camera-mapping stage on
them is close to a no-op for the slots that hold a real camera, but a
placeholder slot will be read as a real camera unless it and its mask column are
dropped first.

## Relation to the training corpus

This release is a subset of the corpus actually used for training. The full
corpus was 181 datasets, 17,137 episodes and 8,690,531 frames; 25 of those
datasets (3,168 episodes, 18.5% of all episodes) declare no license upstream
and are therefore not redistributed here.

The model trained on that corpus cannot be reproduced exactly from this
release. Two differences:

- 3,168 episodes that were trained on are missing from this release
- the 450 episodes from the five 10 fps sources are present here but were
  excluded from the training run, because their video streams start at a
  presentation timestamp of 0.1 s while their parquet timestamps start at 0,
  which makes the loader's timestamp tolerance check fail (see "Known defect"
  under Usage)

13,519 of the 13,969 episodes in this release were part of the training run.

## Schema

| Column | Type | Shape | Description |
|---|---|---|---|
| `action` | float32 | (6,) | commanded joint positions, follower frame, in the units used by the LeRobot SO-101 driver: `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos` |
| `observation.state` | float32 | (6,) | measured joint positions, same six names |
| `observation.images.base_0_rgb` | video | (H, W, 3) | external view (top, front, side, or unspecified) |
| `observation.images.base_0_rgb_mask` | float32 | (1,) | 1.0 if the slot holds a real camera, 0.0 if it holds a placeholder |
| `observation.images.left_wrist_0_rgb` | video | (H, W, 3) | first wrist slot; a wrist-mounted view in most sources, a spare external or unclassified stream in the rest |
| `observation.images.left_wrist_0_rgb_mask` | float32 | (1,) | slot validity |
| `observation.images.right_wrist_0_rgb` | video | (H, W, 3) | third slot; never a genuine second wrist view, since every source is single-arm, but it holds a real camera in 14 sources (1,151 episodes) where a spare external or unclassified stream was placed there. Placeholder in the rest |
| `observation.images.right_wrist_0_rgb_mask` | float32 | (1,) | slot validity |
| `timestamp` | float32 | (1,) | seconds from episode start |
| `frame_index` | int64 | (1,) | index within the episode |
| `episode_index` | int64 | (1,) | global episode index |
| `index` | int64 | (1,) | global frame index |
| `task_index` | int64 | (1,) | index into `meta/tasks.jsonl` |

### Camera slots and masks

The three slot names follow the pi0-family convention: one external slot and
two wrist slots. Source cameras are assigned by keyword. Keys containing
`wrist`, `handeye`, `gripper`, `endeff`, `tip`, `hand`, `arm`, `ego` or `robo`
go to a wrist slot, everything else goes to the external slot. Depth, infrared
and surplus streams are dropped.

A slot left empty by that pass is then filled from whatever cameras remain,
external streams first, then unclassified ones, before it is given up as a
placeholder. Nothing usable is discarded while a slot is free. The consequence
is that a wrist slot does not necessarily hold a wrist view: across the 156
sources, 42 sources (4,704 episodes) have a non-wrist camera in one of the two
wrist slots, and 14 sources (1,151 episodes) have a real camera in
`right_wrist_0_rgb` even though no source has two wrist cameras. Two of those 42
sources have a non-wrist camera in both wrist slots, so the count of occupied
wrist slots is 44 while the count of sources is 42.

**Slot names denote positions in the pi0-family layout, not a guaranteed
viewpoint semantics.** `base_0_rgb` is the cleanest of the three and is
external footage in nearly every source. The wrist slots are mostly, not
exclusively, wrist views.

Slots that nothing maps to are filled with a black video of the same length and
marked with `mask = 0.0`. The masks are the point of the layout: a policy that
reads them can treat a one-camera episode and a three-camera episode
identically, without a separate model per camera count. A policy that ignores
them will train on black frames.

Slot occupancy across the 156 sources: 17 with one real camera, 125 with two,
14 with three. Camera count carries no weight in the merge; episodes are
sampled uniformly by frame, so a three-camera source is not favoured over a
one-camera source.

The mask columns are not standard LeRobot features. Training code that derives
normalization statistics from all non-image columns will pick them up and
distort them; they need to be excluded explicitly.

## Usage

This dataset is in LeRobot codebase version v2.1. LeRobot 0.4.x targets v3.0
and raises `BackwardCompatibilityError` on v2.1 datasets. Either use a LeRobot
release that reads v2.1, or convert a local copy:

```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id=dongyoonkim/so101-pi05-base-dataset \
  --root=/path/to/local/copy \
  --push-to-hub=false
```

Loading:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("dongyoonkim/so101-pi05-base-dataset")

sample = dataset[0]
sample["action"]                                # (6,)
sample["observation.state"]                     # (6,)
sample["observation.images.base_0_rgb"]         # (3, H, W)
sample["observation.images.base_0_rgb_mask"]    # tensor([1.]) or tensor([0.])
sample["task"]                                  # language instruction
```

On older LeRobot releases the import path is
`lerobot.common.datasets.lerobot_dataset`.

Resolutions are not uniform across sources, so a batch cannot be stacked
without a resize. The training run this dataset was built for resized to
224x224 with aspect-preserving padding at load time, leaving the videos
untouched.

### Known defect: 450 episodes are 10 fps behind a global `fps: 30`

`meta/info.json` declares one frame rate for the whole repository, 30, but 450
episodes from five sources were recorded at 10 fps. For those episodes the
parquet `timestamp` column starts at 0.0 while the video's first presentation
timestamp is 0.1 s.

LeRobot pairs each parquet row with a decoded video frame and asserts that the
decoded timestamp is within `tolerance_s` (default 1e-4 s) of the requested
one. For these episodes that assertion **fails**; it raises, it does not warn:

```
AssertionError: One or several query timestamps unexpectedly violate the tolerance
(tensor([0.1000]) > tolerance_s=0.0001).
```

This is the same defect that aborted the training run this dataset was built
for, around step 140, after the pre-launch dataset check and a 20-step smoke
test had both passed. Reading `timestamp` instead of assuming a uniform rate is
correct but does not help on its own: the assertion fires inside the loader,
before your code sees a sample.

The affected sources:

| Source | Episodes |
|---|---|
| `CoRL2026-CSI/IsaacLab-SO101-PullCube-100epi-10fps-appendix` | 100 |
| `CoRL2026-CSI/SO101-teleop_stack_RGBblock_on_bluedish_150epi_10fps` | 150 |
| `anvilbot-patrickhhh/SO101_PickAndPlace_3cams` | 50 |
| `anvilbot-patrickhhh/SO101_PickAndPlace_front_wrist` | 50 |
| `anvilbot-patrickhhh/SO101_relocate_cube_2cams_record_2` | 100 |

Two ways around it:

- **Exclude those five sources.** `meta/sources.json` gives each source's global
  episode range and every entry in `meta/episodes.jsonl` carries its
  `source_dataset`, so the episode indices to skip are directly available. This
  is what the training run did, at a cost of 2.6% of the corpus.
- **Raise the tolerance.** `LeRobotDataset(..., tolerance_s=0.2)` clears the
  assertion, since the offset is one 10 fps frame period. The episodes then load,
  with each parquet row paired to a video frame up to 0.1 s away. That is a
  silent pairing error rather than a loud one, so prefer exclusion unless you
  need these episodes.

Re-anchoring the parquet timestamps to the video's real first PTS
(`ts - ts[0] + pts0`) clears the assertion too, but it moves the whole episode
0.1 s later. Any loader that queries past the last frame - a delay simulation
that shifts the action window forward, for instance - then runs off the end of
the video stream and fails there instead. That is how it failed in this build.

## Limitations

- The data comes from many operators, workspaces and robot units. Per-unit
  calibration zero-point differences are not corrected; they are visible as
  systematic offsets between datasets. Normalization statistics absorb scale
  but not frame or zero-point differences.
- Quality is not uniform. Sources were screened for action-space conformity,
  trajectory jumps, empty or static episodes, metadata integrity, and passed a
  visual check, but they were not screened for task success. Failed and
  partial demonstrations are present.
- Five sources (450 episodes) were recorded at 10 fps and their parquet
  timestamps do not match their video timestamps, so LeRobot's tolerance check
  fails on them as shipped. Exclude those five sources or raise `tolerance_s`;
  see "Known defect" under Usage.
- `meta/info.json` is wrong in three places, all of them metadata rather than
  data. The feature template was copied from whichever source came first in
  traversal order and never reconciled against the rest:

  | Field | What it says | What is true |
  |---|---|---|
  | top-level `fps` | 30 | 13,519 episodes are 30 fps, 450 from five sources are 10 fps. The per-episode `timestamp` columns are correct; this one field is not |
  | `video.fps` in each of the three video feature blocks | 10 | The opposite error, and it contradicts the top-level `fps: 30` in the same file. 10 fps is right for 450 episodes and wrong for the other 13,519 |
  | `shape` and `video.height` / `video.width` in those same three blocks | `[480, 640, 3]`, 480 x 640 | Resolutions vary by source — 640x480, 640x360, 1280x720 and 1920x1080 all occur. One resolution is declared for all three slots and every episode |

  Read the video stream rather than these declarations: the real frame rate and
  the real resolution are in the file. A loader that queries by timestamp gets
  the right frames regardless, which is why none of the three affected the
  training run.
- Four sources are simulation recordings, not real hardware.
- Camera viewpoint semantics are weak. Everything that is not a wrist camera
  goes to the external slot, whether it is a top-down, front or side view, and
  a spare camera fills a wrist slot rather than being dropped: 42 sources
  (4,704 episodes) have a non-wrist camera in a wrist slot. Slot names are
  positions in the layout, not viewpoint guarantees.
- Language instructions come from the upstream recorders. Phrasing, level of
  detail and vocabulary vary widely, and a small number were rewritten to be
  usable as prompts (see the modification list in ATTRIBUTION.md).
- `meta/episodes_stats.jsonl` carries statistics for `action` and
  `observation.state` only; the three image keys appear in `meta/stats.json`
  alone. LeRobot prefers per-episode statistics when that file is present, so a
  policy that normalizes images from dataset statistics should read
  `meta/stats.json` directly. This did not affect the training run, because
  pi0.5 maps visual features to `IDENTITY`.

## Attribution

Every source, its upstream repository, its license and its episode count are
listed in [ATTRIBUTION.md](./ATTRIBUTION.md), together with the modifications
applied to each source. Datasets whose upstream repositories declare no
license are not included in this release.

Provenance is also machine-readable: `meta/sources.json` maps each source to
its license and its global episode range, and each entry in
`meta/episodes.jsonl` carries the `source_dataset` it came from.

## License

The release as a whole is under the Apache License 2.0, full text in
[LICENSE](./LICENSE). One of the 156 component sources is MIT. Per-component:

| License | Origin | Sources | Episodes |
|---|---|---|---|
| apache-2.0 | Hugging Face Hub | 149 | 13,587 |
| mit | Hugging Face Hub | 1 | 50 |
| apache-2.0 | recorded by the author | 6 | 332 |

"Recorded by the author" is a provenance category, not a license: those six
sources are released under the Apache License 2.0 like the rest. The one MIT
source is [`yuk6ra/so101-pen-cleanup`](https://huggingface.co/datasets/yuk6ra/so101-pen-cleanup);
MIT permits redistribution under these terms provided its copyright notice and
permission notice are carried forward, which is done in
[ATTRIBUTION.md](./ATTRIBUTION.md).

The upstream license entry of each source repository is authoritative for that
source.

## Related

Released with this dataset:

- [`dongyoonkim/so101-pi05-base`](https://huggingface.co/dongyoonkim/so101-pi05-base)
  — the pi0.5 checkpoint this dataset was built to train.

Tools and upstream projects:

- [jinnymo/lerobot-v3-v2-converter](https://github.com/jinnymo/lerobot-v3-v2-converter)
  — two-way LeRobot v2.1 / v3.0 dataset converter. Every v3.0 source in this
  build was brought to v2.1 with it, and it also converts the other direction
  for loaders that want v3.0.
- [LeRobot](https://github.com/huggingface/lerobot) — the dataset format and
  the loader this repository targets.
- [VLASH](https://github.com/mit-han-lab/vlash) and
  [openpi](https://github.com/Physical-Intelligence/openpi) — the training
  stack the three-slot layout and the mask columns were shaped for, and the
  pi0.5 reference implementation behind it.
- [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) — the base
  model that training started from.

Author's other SO-101 data and models:

- [`dongyoonkim/so101-eraser-90ep-wrist`](https://huggingface.co/datasets/dongyoonkim/so101-eraser-90ep-wrist)
  — 90 episodes, single wrist camera, one task, LeRobot v3.0. Released on its
  own as a single-task training set rather than as part of a corpus.
- [jinnymo/gr00t-n17-lora](https://github.com/jinnymo/gr00t-n17-lora) and the
  adapter it produced,
  [`dongyoonkim/grootn17-lora-so101-eraser-tier1`](https://huggingface.co/dongyoonkim/grootn17-lora-so101-eraser-tier1)
  — LoRA fine-tuning restored on NVIDIA GR00T N1.7, trained on that dataset.
