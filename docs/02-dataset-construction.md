# 02 — Dataset construction

How the training corpus was built: from "SO-101 recordings exist somewhere on the
Hugging Face Hub" to a single LeRobot v2.1 repository that a pi0.5 full fine-tune
can read, and from there to the license-filtered subset that is published. The work
splits into twelve stages.

Two pitfalls in the source data are severe enough to have their own section at the
end: video presentation timestamps that do not start where the parquet timestamps
start, and mixed frame rates that survive a merge as a single wrong `fps` field.
Both of them aborted a running 40-hour training job rather than failing at build
time. Read that section before running the merge.

## Result

| | |
|---|---|
| Merged corpus | 17,137 episodes / 8,690,531 frames / 430 tasks / 51,411 videos |
| Format | LeRobot codebase version v2.1, single repository |
| Cameras | 3 fixed slots plus one validity mask column per slot |
| Action space | 6 DoF, follower frame, degrees |
| Fed to the released training run | 16,687 episodes / 8,595,621 frames (450 episodes from five 10 fps sources excluded, see pitfall 2) |
| Published subset | 156 source datasets / 13,969 episodes / 7,272,752 frames |

### Funnel

```
Hub name search over 9 query strings                 11,270 dataset repositories
  LeRobot metadata identifying an SO-101/SO-100 arm    1,724   catalogued
  screening (episode count, camera blacklist, sim)       408   (clean 390 + sim 18)
  action-space conformity == green                       307   (real 297 / sim 10)
  downloaded                                             307   305 automatic + 2 by git lfs, 303 GB
  precision analysis, quality exclusions                 296   11 removed
  camera-class selection and manual duplicate removal    205   external
  plus locally recorded datasets                         211   (205 + 6)
  action-fingerprint deduplication                       181   30 dropped
  merge                                     17,137 ep / 8,690,531 frames / 430 tasks
  license filter for publication            156 datasets / 13,969 ep / 7,272,752 frames
```

## Before you start

**Software.** Python 3.10 or newer, `huggingface_hub`, `pyarrow`, `numpy`, `av`
(PyAV), `openpyxl`, and `ffmpeg` on `PATH`. The v3.0 to v2.1 conversion additionally
needs a LeRobot install and a checkout of
[`lerobot-v3-v2-converter`](https://github.com/jinnymo/lerobot-v3-v2-converter). The
smoke tests at the end need the training stack
([VLASH](https://github.com/mit-han-lab/vlash), which pins LeRobot 0.4.1) and one
CUDA GPU. Exact patch versions of the Python packages used for this build are not
recorded; the LeRobot pin (0.4.1) and the two dataset format versions (v2.1 source
and target, v3.0 input for the converter) are.

**Disk.** Roughly 1 TB free if every intermediate is kept:

| Artifact | Size |
|---|---|
| Raw downloads, video included | 303 GB |
| Curated pool copied into a flat layout | 246 GB |
| Merged repository | 198 GB apparent, mostly hardlinks into the pool |
| Published subset | about 171 GB apparent, also hardlinked |

The merge hardlinks source video instead of copying it, so the merged repository
costs its parquet files plus the black placeholder videos on top of the pool, not
another 198 GB. Hardlinks require the merge output to sit on the same filesystem as
the pool; across filesystems the code falls back to `shutil.copy` and the cost
becomes real.

**Network.** A Hugging Face token in `HF_TOKEN` removes anonymous rate limits during
the crawl and the download. Revoke and reissue the token when the build is done; it
ends up in shell history and logs.

**Time.** Wall-clock durations for the build stages were not recorded. The download
of 303 GB and the merge of 17,137 episodes are the two long ones; everything else
operates on metadata.

**Paths.** This document uses `$ROOT` for the working directory and `$REF` for a
locally recorded reference dataset. The reference is one SO-101 recording made on
your own arm, in the follower frame, with a correct calibration; every candidate is
compared against it in stage 3. **This is the pipeline's one hardware prerequisite.**
Without a reference of your own you can substitute a public dataset you trust, but then
you are matching someone else's convention; stage 3 gives the substitute procedure and
three datasets that were measured during this build.

```
$ROOT/external_hf/          raw downloads, one directory per repository
$ROOT/base_train/           curated pool: external/<tier>/<name> and self/<name>
$ROOT/base_unified/         merged repository
$ROOT/base_open/            license-filtered subset for publication
$REF                        reference recording, follower frame
```

**Scripts.** The stage scripts ship in this repository under `pipeline/`, named as they
are referenced below (`01_hf_so101_crawler.py` onward). Paths are always command-line
arguments, never constants inside the files, but they are not all the same argument:

| Group | Scripts | Where paths come from |
|---|---|---|
| Dataset-tree stages | 04, 05, 06, 07, 08, 09, 10, 13, 14, 20, 22, 24, 28, 29, 30, 31, 32, 33 | `--root` (default `$ROOT`, then `.`), from which each derives `external_hf/`, `base_train/` or `base_unified/`. `--src`, `--dst` and `--dest` override the derived directory |
| Catalog stages | 01, 03 | no `--root`. They only touch spreadsheets, named by `--csv` and `--xlsx`, relative to the current directory |
| Report stages | 11, 12 | no `--root`. They edit the analysis workbook in place, named by `--report`, relative to the current directory |
| Manifest assembly | 26 | no `--root`. `--artifacts` is the directory it reads the three JSON inputs from, `--out` the manifest it writes |

Output locations differ in the same way. Stages 20, 22, 24 and 26 default to writing
their JSON instruction sheets into `pipeline/artifacts/`, which also holds the ones this
build produced, so those four can be inspected against the original before being re-run.
Stages 01, 03 and 06 write their spreadsheets into the current directory, and 07, 08, 10,
11 and 12 read and rewrite files already there. The remaining stages write into the
dataset tree under `--root`, or print a console report and write nothing. Every stage
prints what it wrote.

The two release-side scripts are separate, under `scripts/`: `repack_open_subset.py` and
`gen_attribution.py`. Gaps in the numbering are stages that were folded into a neighbour;
the sequence in the Minimum reproduction section at the end is the authoritative order.

---

## 1. Crawl the Hub

**Purpose.** There is no registry of SO-101 datasets. They are user uploads scattered
across the Hub under inconsistent names, and the only reliable evidence that a
repository holds SO-101 data is its LeRobot `meta/info.json`. So: search wide by
name, then confirm by metadata.

**Script.** `01_hf_so101_crawler.py --full`.

**Output.** `02_so101_catalog.csv` and a two-sheet `.xlsx` (`catalog`, `recommended`).

**How it searches.** Nine query strings, deduplicated by repository id:

```python
SEARCH_TERMS = ["so101", "so-101", "so100", "so-100",
                "soarm101", "soarm100", "so_arm101", "so_arm100", "soarm"]
```

`HfApi.list_datasets(search=q)` for each. SO-100 is included deliberately: it is the
predecessor arm with the same six joints and the same action convention, and its
recordings are usable as SO-101 prior.

**What it collects per repository.** Two raw fetches, `meta/info.json` and
`meta/tasks.jsonl`, over plain HTTP with a 404-aware retry (404 is final, 429/500/503
back off and retry). From those plus the index entry:

| Field | Source | Used for |
|---|---|---|
| `id`, `url`, `downloads`, `likes`, `last_modified`, `tags` | Hub index | provenance, license tag, ranking |
| `codebase_version` | info.json | v2.1 vs v3.0 routing (stage 5) |
| `robot_type` | info.json | recorded, not used as a filter |
| `action_dim`, action `names` | info.json `features.action` | single vs bimanual, stage 3 input |
| `n_cam`, `cam_keys`, `has_depth` | info.json keys under `observation.images` | stage 2 blacklist, stage 6 mapping |
| `episodes`, `frames`, `fps` | info.json totals | size screening |
| `tasks` | tasks.jsonl, first 8 unique | language instruction survey |
| `is_variant` | id matches `merged\|multiplied\|trimmed\|converted\|_copy\|backup\|eval_` | duplicate hint |
| `score` | `min(ep,300) + whitelist_cams*20 + min(downloads,400)/4`, minus penalties | sheet ordering only |

**Pass criterion.** A repository enters the catalog if `meta/info.json` parses and
identifies a LeRobot recording. Everything else is a `parse_fail` row.

**Result.** 11,270 repositories matched the name searches. 1,724 carried usable
LeRobot metadata and were catalogued. The per-repository reason each of the other
9,546 was discarded is not recorded; the recorded quantity is the 11,270 to 1,724
reduction.

**Failure modes.**

- `tasks.jsonl` does not exist in LeRobot v3.0, where the task table moved into a
  parquet file. Task extraction returns empty for every v3.0 repository. This is why
  a later stage re-extracts prompts after conversion (stage 8) instead of trusting
  the catalog column.
- Anonymous requests hit HTTP 429 at 12 workers. Set `HF_TOKEN`.
- 136 of the 1,724 catalogued rows are `parse_fail`: the repository name matched but
  `meta/info.json` was missing, private, or not JSON. They are kept as rows so the
  crawl is auditable, and they are filtered out at the next stage.
- Name search misses correctly-formatted SO-101 datasets that never mention the arm
  in their repository name. There is no cheap fix; the corpus is a large sample of
  public SO-101 data, not the whole of it.

---

## 2. Screen the catalog

**Purpose.** Reduce 1,724 catalog rows to a candidate set worth fetching statistics
for. The screen is deliberately permissive on viewpoint and strict on size, because
the goal is a base model that has seen many environments, not a curated benchmark.

**Script.** `01_hf_so101_crawler.py` `analyze()`, materialized by
`01_hf_so101_crawler.py --rebuild`.

**Input.** `02_so101_catalog.csv`. **Output.** the same CSV with a `category` column,
and the filtered `.xlsx`.

**Rules.** Each row gets exactly one category, evaluated in this order:

```python
row["category"] = ("sim"           if is_sim
                   else "blacklist_cam" if has_black
                   else "small"         if eps < 50
                   else "clean")
```

- `sim`: repository id or tags match `sim|isaac|mujoco|genesis|svla|synthetic`.
  Simulation is kept but labelled, so its share stays visible.
- `blacklist_cam`: any camera key contains `laptop`, `phone`, `webcam`, `screen`,
  `iphone`, or `android`. These are hand-held or screen-capture views, not a robot
  workspace camera. Everything else is allowed through, including depth,
  RealSense, and unidentifiable viewpoints. Only genuinely wrong cameras are cut.
- `small`: fewer than 50 episodes. Below that a dataset contributes little and costs
  the same per-repository handling as a large one.
- `clean`: everything else.

Selection for the next stage is `category in {clean, sim} and episodes >= 50`.

**What this stage does not gate on.** `robot_type` and `codebase_version` are
collected but do not gate anything here: `robot_type` is free text that many
uploaders leave empty or misspell, and both v2.1 and v3.0 are accepted (v3.0 is
converted in stage 5). `action_dim` is likewise recorded but enforced later, in
stage 3, where the whole action convention is checked at once. The only hard gates at
this stage are the camera blacklist and the 50-episode floor.

**Result.**

| Category | Rows |
|---|---|
| clean | 390 |
| sim | 49 |
| small | 822 |
| blacklist_cam | 327 |
| parse_fail | 136 |
| **Selected** (`clean`+`sim`, >= 50 episodes) | **408** (390 clean + 18 sim) |

31 of the 49 sim rows fell below the episode floor. 6 of the 408 selected datasets
carry a depth stream, which is dropped later at slot mapping, not here.

**Failure modes.**

- The sim test is a name and tag heuristic. A simulation dataset with a neutral name
  is labelled `clean` and stays in the pool. The visual check in the interlude below
  caught some of these; assume the label is a lower bound.
- Category order matters: a simulation dataset with a laptop camera is filed under
  `sim`, not `blacklist_cam`, so the camera blacklist is not applied to it. This did
  not change the outcome here but is worth knowing if you rerun the screen.
- The `score` column and the `recommended` sheet exist to make the spreadsheet
  readable. Nothing downstream reads them. Do not use them as a quality filter.

---

## 3. Check action-space conformity

**Purpose.** This is the gate that matters most, and it runs before anything is
downloaded. Two SO-101 datasets can look identical and be incompatible, because
`action` may be recorded in the leader arm's calibration frame rather than the
follower's. Normalization statistics absorb a scale difference; they do not absorb a
shifted zero point or a different semantic frame. A policy trained on mixed frames
learns a constant offset that shows up as drift in closed loop. Delta actions, raw
motor ticks, radians, reordered joints, and bimanual 12-dimensional actions are the
other ways a dataset can be silently wrong.

**Script.** `03_action_match.py`, then `--retry` for transient fetch failures, then
`--video` to attach a first-episode video link to each surviving row for eyeballing.

**This stage needs a reference recording, and that normally means hardware.** The
reference used for this build was a recording made on our own SO-101, deliberately saved
in the follower frame. Without an SO-101 you cannot produce that recording, and this is
the only stage in the pipeline that has a hardware prerequisite.

**Substitute: use a public dataset as the reference.** Every test above is relative, so
any dataset already in the follower frame with the standard joint layout serves. The
datasets this pipeline itself flags green are by construction exactly that. Three that
were measured during this build, all Apache-2.0, with an action-to-state offset well
inside the green threshold:

| Repository | Episodes | Cameras | Measured action-state offset |
|---|---|---|---|
| `zaringleb/pick_single_cube_so101` | 185 | top + wrist | 1.1 deg |
| `zacapa/SO101_chess_test2_6` | 274 | overhead + wrist | 0.6 deg |
| `lipsop/so101-block-in-bin-100ep` | 100 | front + wrist | 0.8 deg |

```bash
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('zaringleb/pick_single_cube_so101', repo_type='dataset', \
                      local_dir='$ROOT/reference_dataset')"
export REF=$ROOT/reference_dataset
```

Those numbers come from the crawl this document describes, not from a live check; a
repository can be re-uploaded or deleted. Re-verify before trusting one: look the row up
in `02_so101_catalog.csv` after stage 1, and after a first pass of stage 3 confirm the
candidate comes back green against itself. `03_action_match.py` accepts either
`meta/stats.json` or `meta/episodes_stats.jsonl` as the reference statistics, so a plain
downloaded v2.1 dataset works as-is.

Borrowing a reference costs two things. The range-overlap column becomes overlap with
*that* robot's workspace rather than yours, and any systematic calibration offset in the
reference is inherited by the whole screen. Prefer a reference with many episodes and
wide joint coverage, and read the `unit` distribution the stage prints as a sanity check.

**Input.** The 408 selected rows, plus `$REF/meta/info.json` and `$REF/meta/stats.json`
for the reference convention. **Output.** `03_action_match.csv` and a colour-coded
`.xlsx` sorted green, yellow, red.

**What it fetches per candidate.** `meta/info.json` for the action feature names, and
statistics. Statistics live in `meta/stats.json` in some datasets and only in
`meta/episodes_stats.jsonl` in others; the script falls back to the second and
aggregates it count-weighted:

```python
amin = elementwise min over episodes
amax = elementwise max over episodes
amean = sum(episode_mean * episode_count) / total_count
```

The same aggregation is applied to `observation.state`, because the difference
between the two means is the frame test.

**The four tests.**

1. **Dimension and joint names.** Names are normalized (`lower()`, strip `.pos`,
   `_pos`, `observation.`) and compared as a set against the SO-101 standard:

   ```python
   STD = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]
   ```

   Set equality is `names_ok`, exact sequence equality is `order_ok`.

2. **Unit.** Inferred from the magnitude of the action range, not from any declared
   field:

   ```python
   m = max(|min|, |max|) over all joints
   m < 1.5  -> "norm"      (already normalized to about [-1, 1])
   m < 8    -> "radian"
   m < 400  -> "degree"
   else     -> "raw"       (motor ticks)
   ```

   The reference is `degree`.

3. **Frame (leader versus follower).** The mean absolute difference between the
   action mean and the state mean, per joint, averaged:

   ```python
   offset = mean(|action.mean[i] - state.mean[i]| for i in range(dim))
   ```

   A follower-frame recording commands roughly what it achieves, so the offset is
   small. A leader-frame recording carries the leader's calibration zero, which
   appears as a systematic offset. Measured over the 307 green rows of
   `03_action_match.csv`, the offset runs from 0.3 to 5.4 degrees with a median of
   0.7; 302 of the 307 are at or below 2.0 degrees and 5 are above it, the largest
   being 5.4. The spread matters when reading the pass criteria below: the green
   band extends to 6 degrees, and the observed maximum sits just inside it rather
   than clustering safely at the bottom.

4. **Range overlap.** For datasets that pass names, order and unit, the per-joint
   intersection of the candidate range with the reference range, divided by the
   reference span, averaged. Informational; it does not change the flag.

**Pass criteria.**

| Flag | Condition |
|---|---|
| red | `dim != 6`, or joint names differ as a set, or offset > 15 degrees (strong leader suspicion), or metadata could not be fetched |
| yellow | unit differs from the reference (conversion needed), or joint order differs (reorder needed), or offset in 6 to 15 degrees (weak leader suspicion), or `observation.state` absent so the offset is unverifiable |
| green | dimension 6, names and order match, unit is degrees, offset <= 6 degrees |

Only green datasets proceed. Yellow is recoverable in principle and was not
recovered: with 307 clean candidates available, converting units or reordering joints
for three more datasets was not worth the risk of getting the frame wrong.

**Result over the 408 candidates.**

| Flag | Count | Breakdown |
|---|---|---|
| green | 307 | 297 real, 10 simulation |
| yellow | 3 | 2 in radians, 1 with a 13.6 degree offset |
| red | 98 | 97 dimension or name mismatch, 1 metadata fetch failure |

Unit distribution across all 408: degree 373, raw 23, radian 11, unknown 1. Format
split within the green set: 223 already v2.1, 84 v3.0. Camera count within the green
set: one camera 31, two 236, three 38, five 2 — two cameras (one wrist, one external)
is the SO-101 community default. Zero datasets landed in the strong-leader-suspicion
band, which matches the fact that SO-101 teleoperation recorders write follower
positions by default.

**Failure modes.**

- **This test cannot see your own arm's zero point.** It compares action against
  state *within* each dataset, so a consistent calibration offset in the uploader's
  own robot cancels out. It proves the dataset is internally follower-framed; it does
  not prove the dataset's zero agrees with yours. Correcting that requires replaying
  trajectories on your hardware, which was not done. The residual is left for the
  downstream task fine-tune to absorb.
- Missing `observation.state` makes the frame test impossible. Those rows are yellow,
  not green, and were dropped.
- The unit classifier is a magnitude heuristic. A dataset whose joints happen never
  to leave a small range could be classified `radian` while being degrees. The joint
  name and range-overlap columns exist so this is visible on inspection.
- Bimanual datasets (`action_dim` 12, or names containing both `left` and `right`)
  are red by construction. A bimanual base is a different action space and a separate
  project; do not mix the two.

---

## 4. Download

**Purpose.** Fetch the 307 green repositories with video, so that later stages can
inspect trajectories and frames rather than metadata.

**Script.** `04_download_green.py`.

**Input.** rows with `flag == green`. **Output.** `$ROOT/external_hf/{owner}__{name}/`.

**How.** `snapshot_download(repo_type="dataset", local_dir=...)` with 10 repositories
in parallel and 4 file workers each. Environment is set before the
`huggingface_hub` import:

```python
os.environ.setdefault("HF_HOME", f"{ROOT}/.hf_cache")   # keep the cache off the system disk
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
```

A `.download_done` marker file is written into each completed
repository directory, so an interrupted run resumes by skipping finished
repositories rather than by re-verifying 303 GB of files. One automatic retry per
repository.

**Pass criterion.** The marker file exists and the directory holds `meta/`, `data/`
and `videos/`.

**Result.** All 307 green repositories were obtained, 303 GB total.[^dl] 305 came down
through `snapshot_download`; the remaining 2 failed with a brotli decode error from the
transfer layer and were recovered with a plain `git lfs clone` of the repository.

[^dl]: Both counts appear in the record and mean different things, which is why every
    stage downstream of this one starts from 307. 305 is what the automated downloader
    reported; 307 is what was on disk when the stage finished, and is the number the
    quality-inspection stage read.

**Failure modes.**

- Brotli decode errors on specific repositories, unaffected by retrying. Fall back to
  `git lfs clone`.
- `local_dir` mode is used instead of the shared cache layout because the target
  filesystem here is NTFS, where symlinks into the cache do not behave. It costs disk
  (a full copy per repository) and buys portability.
- Leaving `HF_HOME` at its default fills the system disk with a second copy of
  everything. Set it explicitly.
- The token is passed through the environment and shows up in process listings and
  logs. Rotate it afterwards.

---

## 5. Convert LeRobot v3.0 to v2.1

**Purpose.** The corpus has to be a single repository in a single format, because
the training stack loads one LeRobot dataset. 84 of the 307 green datasets were
uploaded in v3.0, whose directory and metadata layout differs from v2.1: episode
metadata moves into parquet, chunking and file naming change. The target is v2.1,
because the training stack's compatibility shim reads v2.1 directly (LeRobot 0.4.1
otherwise raises `BackwardCompatibilityError` on it), and because v2.1 is what the
merge script emits.

**Script.** `pipeline/05_convert_v3_to_v2.py`. It selects the datasets to convert and
runs the converter once per dataset, four in parallel.

**What this build ran, and what reproduces it.** The original build imported
`convert_v3_to_v2.convert_dataset` straight out of a local Isaac-GR00T checkout. That
checkout no longer exists and its commit was not recorded at the time. The conversion
logic itself is recoverable: `lerobot-v3-v2-converter` vendors
`scripts/lerobot_conversion/convert_v3_to_v2.py` from Isaac-GR00T commit
`23ace64f17aa5015259b8609d371eb61a357c776` verbatim, with the original Apache-2.0
header and the source, path and commit recorded in its `NOTICE`. Reproducing stage 5
against that converter therefore runs the same conversion, not a reimplementation of
it. Nothing else from Isaac-GR00T is involved: GR00T is not part of this project, and
no GR00T model or weight is used.

```bash
git clone https://github.com/jinnymo/lerobot-v3-v2-converter
python 05_convert_v3_to_v2.py --root "$ROOT" \
    --converter /path/to/lerobot-v3-v2-converter
```

The wrapper invokes the converter's `v3_to_v2/convert.py` as a subprocess with
`--input <dataset dir>`, which converts in place and leaves the original beside it. The
converter is a plain checkout, not a package, and needs an environment with LeRobot; it
carries an import shim that makes the vendored code run on LeRobot 0.4.x and 0.5.x
alike, where the unpatched Isaac-GR00T script only imports on 0.4.x. Running
`v3_to_v2/convert.py` by hand on one directory is the same code path.

**Input.** Downloaded repositories where `codebase_version == "v3.0"`.
**Output.** The same path holding a v2.1 dataset; the v3.0 original is moved aside to
`{name}_v3.0`, and the backups are collected under `$ROOT/external_hf_v3_backup/`.

**Pass criterion.** `meta/info.json` of the converted directory reads
`codebase_version: v2.1`, and the episode and frame counts survive. The script is
idempotent: a directory already at v2.1 is skipped, so a failed run can be rerun.

**Result.** 84 datasets were selected for conversion (the green set's v3.0 share); 83
v3.0 backup directories are recorded. Video is stream-copied, not re-encoded, so
conversion does not touch pixel data or timing.

**Side effect that matters later: the converted directory does not carry the source
`README.md`.** Measured on the final pool: of 143 natively-v2.1 datasets, 126 have a
local `README.md`; of the 62 converted ones present in the pool, zero do. The YAML
front matter of that README is where a Hub dataset declares its license, so for
converted datasets the license is not resolvable from local files at all. Stage 12
therefore resolves licenses from the crawl catalog's tag string and, for anything
still unresolved, from the Hub API — not from the files on disk. If you skip that and
read licenses locally, every converted dataset looks unlicensed.

**Failure modes.**

- The converter shells out to `ffmpeg` per episode. It is I/O-bound; more than about
  four in parallel does not help and competes for the same disk.
- A partially converted directory is neither v3.0 nor valid v2.1. The `{name}_v3.0`
  backup is the recovery path; keep it until the merge has been verified.
- Episode-level statistics are regenerated during conversion. Do not compare
  `episodes_stats.jsonl` across the conversion boundary and expect byte equality;
  the deduplication fingerprint in stage 7 rounds to one decimal for exactly this
  reason.

---

## Interlude: precision analysis and pool selection

Between conversion and slot mapping, the 307 downloaded datasets are analyzed frame
by frame and reduced to the 205 external datasets that enter the merge, plus 6
locally recorded ones. This is bookkeeping rather than transformation, but it is
where the pool actually gets its shape.

**`06_analyze_datasets.py`** reads every episode of every dataset and writes one
spreadsheet row per dataset: size, camera keys, resolution, codec, action dimension
and unit, per-joint global range, and four quality counters.

| Counter | Rule | Meaning |
|---|---|---|
| `traj_max_jump`, `n_anomaly_ep` | max per-frame action delta > 50 degrees | discontinuity, teleoperation glitch, or bad splice |
| `n_static_ep` | every joint's range within the episode < 5 degrees | empty or frozen episode |
| `n_desync_ep` | mean per-episode `\|action - state\|` > 10 degrees | action and state out of sync |
| `range_warn` | joint span > 320 degrees, or < 1 degree on a non-gripper joint | implausible range, or a stuck joint |

Plus a video check that decodes the first ten frames of a representative camera and
flags an all-black opening. Ten frames rather than one, because many recordings open
on a black frame and a single-frame check produces false positives.

**`07_classify_camera.py`** sorts datasets into tiers by camera keyword and applies a
viewpoint budget. Tier 1 has both an overhead-type and a wrist camera, tier 2 has one
of them, tier 3 has neither but has a clear front or side view, and anything with
`up` or an unidentifiable key goes to a manual bucket. Tier 3 is capped at one
quarter of tier 1 and 2 episodes, taking the highest-scoring datasets first, so that
front views do not dominate the mix. `08_add_class.py` writes the resulting class
back into the spreadsheet, `09_verify_integrity.py` checks parquet count against
`total_episodes`, summed rows against `total_frames`, per-camera video count against
episode count, and first-episode video frames against first-episode rows.
`10_finalize_meta.py` repairs `info.json` totals from `episodes.jsonl` where they
disagree, `11_summary.py` writes an aggregate sheet, and `12_prompt_none.py` lists
datasets whose task string is missing so they can be filled in.

**Results.** 15 datasets carried inaccurate `total_frames` or `total_episodes` in
`info.json` while their data was intact (rows matched episode lengths); metadata was
corrected, data was untouched. 290 datasets had completely clean trajectories.
Quality exclusions removed 11 datasets, of which the record itemizes 7: one with a
nonstandard layout, three with 200-degree jumps, two where 92% of episodes contained
a jump, and one with no video files. The remaining 4 are not itemized in the record.
Camera-class selection and manual duplicate removal (11 datasets removed as obvious
re-uploads, cumulative uploads or evaluation splits) brought the pool to 205
external datasets: 46 tier 1, 115 tier 2, 39 tier 3, 5 held back for visual
confirmation. Adding 6 locally recorded datasets gives **211 datasets / 19,858
episodes / 246 GB** at `$ROOT/base_train/`, a copy in a flat layout, unconverted and
unnormalized.

A visual review of a sample (all 44 tier-3 and manual-bucket datasets, plus 51
sampled from tiers 1 and 2) found no unusable recordings. Five simulation datasets
were tagged and kept. A license survey at this stage found no non-commercial or
no-derivatives licenses in the pool.

**Two pre-merge checks run against the assembled pool, because after the merge an
episode can no longer be traced back to its source:**

- `13_precheck.py` reads every episode of all 211 datasets and reports NaN or
  infinite values in action or state, `frame_index` that is not `0..n-1`, non-monotonic
  timestamps, a measured frame rate more than 25% away from `info.json` `fps`,
  absolute values above 500, and episodes shorter than the policy's chunk plus delay
  (50 + 8 = 58 frames; short episodes are kept, since the trainer pads them).
- `14_video_sync.py` compares, per episode and per camera, the video frame count
  against the parquet row count.

Run both. The frame-rate consistency test in `13_precheck.py` is the one that would
have caught pitfall 2 at build time.

---

## 6. Map cameras onto three slots

**Purpose.** pi0.5 has exactly three image slots — `base_0_rgb`,
`left_wrist_0_rgb`, `right_wrist_0_rgb` — and the source datasets have between one
and five cameras under arbitrary key names: `front`, `top`, `wrist`, `main`,
`camera1`, `up`, `ego`, `handeye`, `laptop`, `realsense_depth`, and dozens more.
Something has to decide which stream goes into which slot, for 211 datasets, without
a human looking at 50,000 videos.

**The decision.** Map by keyword into two semantic classes only — wrist-mounted or
not — and accept that viewpoint semantics beyond that are lost. A base model's job
here is a broad SO-101 visuomotor prior, not a precise camera rig. Wrist cameras are
kept consistent because a downstream user's wrist camera should land in the same slot
it did during training; everything else goes to the external slot regardless of
whether it is overhead, front or side. Cost of the decision: `base_0_rgb` mixes
viewpoints. Benefit: zero manual work, and every dataset keeps all of its usable
cameras.

**Script.** `20_camera_slot_map.py`. **Input.** `meta/info.json` of each of the 211
datasets. **Output.** `21_camera_slot_map.json`, one entry per dataset.

**Rules, in order.**

```python
WRIST_KW    = ("wrist", "gripper", "endeff", "handeye", "ego", "tip", "hand", "arm", "robo")
DEPTH_IR_KW = ("depth", "_ir", "infrared")
AMBIG_KW    = ("obs_image", "camera1", "camera2", "camera_1", "camera_2")
EXT_PRIORITY = ("front", "top", "overhead", "side", "up", "head", "above",
                "context", "main", "fixed", "base", "horizon", "area")
```

1. Depth and infrared streams are excluded. The slots are RGB.
2. External cameras are sorted by `EXT_PRIORITY`; the best one takes `base_0_rgb`.
3. Wrist cameras take `left_wrist_0_rgb`, then `right_wrist_0_rgb`.
4. Slots still empty are filled from the remaining pool, external first, then
   ambiguous, then surplus wrist streams. Nothing usable is thrown away while a slot
   is free.
5. Cameras beyond three are dropped.
6. Slots that stayed empty are recorded in `masked_slots` and become a black
   placeholder plus `mask = 0.0` at merge time.

**Ambiguous keys** (`camera1`, `obs_image`, and friends) name no viewpoint at all.
They are not guessed at and not discarded: they are held back behind the named
external cameras and only used to fill a slot that would otherwise be masked, and
each is flagged in the output for review. In practice they land in `base_0_rgb` for
single-camera datasets, which is the correct default for an unidentified fixed view.

**Pass criterion.** Every dataset produces a slot assignment with at least one
present slot, and no depth stream reaches a slot.

**Result at dataset level** (211 datasets): 17 fill one slot, 165 fill two, 29 fill
all three. Audit of what actually landed where:

| Slot | Real external | Real wrist | Placeholder |
|---|---|---|---|
| `base_0_rgb` | 208 | 0 | 3 |
| `left_wrist_0_rgb` | 41 | 156 | 14 |
| `right_wrist_0_rgb` | 29 | 0 | 182 |

`base_0_rgb` is 99% clean. `left_wrist_0_rgb` is 74% real wrist views, with 20%
external views spilled into it by rule 4. That spill is the price of rule 4 and was
accepted: for a base model, more visual coverage is worth more than slot purity.
`right_wrist_0_rgb` never holds a real second wrist camera, because every source is
single-arm; it holds a spilled external view in 29 datasets and a placeholder in the
rest.

**Result at episode level**, after deduplication and merging (17,137 episodes): 1,314
episodes with one real camera, 14,098 with two, 1,725 with three.

**Camera count is not a weight.** Sampling is uniform over frames, the loss is on
actions only, and normalization treats images as identity. Nothing anywhere multiplies
by the number of cameras, so a three-camera episode does not outweigh a one-camera
episode. This is a deliberate property, not an accident: it is what lets a masked
one-camera dataset be merged with a three-camera dataset without reweighting.

**Failure modes.**

- Substring matching over-triggers. `arm` in `WRIST_KW` matches any key containing
  "arm", including some fixed cameras named after the arm they watch. This is part of
  why 20% of the wrist slot is external footage. Tighten the keywords, or accept it as
  this build did.
- A dataset with more than three cameras loses the surplus. Two datasets had five.
- Depth exclusion is by key name. A depth stream named `camera2` is not caught by the
  keyword and would end up in an RGB slot as an ambiguous fill. One dataset was
  excluded at the visual-review stage for exactly this reason.
- A dataset with only wrist cameras leaves `base_0_rgb` masked (3 of 211). That is
  legal but unusual; check that it is not a mapping bug.

---

## 7. Deduplicate by action fingerprint

**Purpose.** Public SO-101 datasets duplicate each other heavily, and the duplicates
do not share names. Three patterns:

- **Cross-author re-upload.** Someone downloads a dataset, re-uploads it under their
  own account, sometimes after a format conversion.
- **Cumulative upload.** An uploader publishes 100 episodes, records 100 more, and
  publishes all 200 as a new repository without deleting the first.
- **Train/validation leakage.** The same recording is split and published as two
  repositories, so the "validation" episodes are a subset of the training data.

All three inflate the corpus with exact copies of the same trajectories. Duplicated
episodes are effectively a per-source learning-rate multiplier for whichever
operator uploaded twice, which is not a property anyone chose.

**The method.** Reduce each dataset to an order-independent multiset over its
episodes:

```python
fingerprint = Counter( (episode_length, tuple(round(mean_action_j, 1) for j in joints)) )
```

Both components come from `meta/episodes.jsonl` and `meta/episodes_stats.jsonl`. No
parquet and no video is read, so 211 datasets fingerprint in seconds.

Why this works: a copy of a trajectory has the same length and the same per-joint mean
action, whatever the repository is called, whatever order the episodes are in, and
whichever chunk file they live in. Order independence matters because re-uploads and
merges routinely renumber episodes. Rounding to one decimal absorbs the statistics
regeneration that happens during v3.0 to v2.1 conversion while staying far tighter
than the spread between genuinely different episodes.

Why not length alone: unrelated datasets recorded by the same operator with the same
timer produce matching length profiles, and length-only matching flagged them as
duplicates. Adding the mean action removed those false positives. Why not hashing the
video or the parquet: re-encoding, re-chunking and format conversion all change bytes
without changing trajectories, so byte hashing finds far less than this does.

**Script.** `22_dedup_verify.py`. **Output.** `23_dedup_result.json` with the kept
set, the dropped set, and the relation and episode counts behind every drop.

**Decision rules.**

| Relation | Action |
|---|---|
| fingerprints exactly equal | keep one (lexicographically first name), drop the rest |
| `fp(x)` is a proper subset of `fp(y)` | drop `x`, keep the largest superset |

Comparison is all-pairs across every dataset, ignoring the author, which is what
catches cross-author re-uploads.

**Result.** 211 datasets in, 181 kept, 30 dropped:

| Reason | Count |
|---|---|
| proper subset of a larger dataset (cumulative uploads, splits) | 18 |
| identical fingerprint (re-uploads, evaluation copies) | 10 |
| manual: train/validation leakage the fingerprint missed | 2 |

The two manual drops are a train/final and validation/final pair from one uploader
whose metadata was inconsistent enough that the fingerprints did not line up, but
whose raw actions were plainly the same recording. They are hardcoded in the manifest
builder with the reason recorded next to them.

One correction went the other way: a workflow that compared datasets across authors
flagged a three-camera and a two-camera dataset from the same uploader as duplicates.
Their actions differ; the fingerprint kept both. Duplicate detection that works on
names, task strings or camera layouts will produce this class of false positive.

**Failure modes.**

- A dataset that re-records the same task from scratch is not a duplicate and must not
  be treated as one. This is the whole reason the key includes the mean action.
- Any dataset that is missing `episodes_stats.jsonl` cannot be fingerprinted and is
  reported rather than silently kept or dropped. Handle those by hand.
- Trimmed re-uploads (a copy with a few episodes removed and the rest re-encoded) can
  break exact-length matching and slip through as a partial overlap. The subset rule
  catches full containment, not near-containment.

---

## 8. Drop bad episodes

**Purpose.** Dataset-level screening keeps or discards whole repositories. A few
individual episodes inside otherwise good datasets will crash a training run or
teach nonsense: empty episodes, episodes whose video is shorter than the recorded
motion, and episodes whose task label is not an instruction.

**Script.** `24_ep_drop.py` produces the candidate list; `26_build_manifest.py`
applies the policy that decides which candidates actually get dropped.

**Detection rules** (`24_ep_drop.py`, per episode, per camera):

| Rule | Action |
|---|---|
| fewer than 2 rows | drop, empty episode |
| `timestamp[-1] - timestamp[0] <= 0` | drop |
| video file missing | drop; LeRobot will try to decode it and crash |
| `abs(video_duration - action_duration) / action_duration > 0.05` | flag as a duration mismatch |

**Compare durations in seconds, never frame counts.** A 10 fps video paired with a
30 fps parquet legitimately has one third of the rows as frames, and LeRobot indexes
video by timestamp, not by row. A frame-count comparison flags every mixed-rate
dataset as broken. Video duration is read from container metadata, which is fast, with
a decode-and-count fallback.

**Policy** (`26_build_manifest.py`): of the duration mismatches, only the ones where
the **video is shorter than the motion** are dropped. A video that runs *longer* than
the recorded motion is harmless, because LeRobot only decodes within the episode's
timestamp range and never touches the tail.

The detection pass records a human-readable reason per flagged episode, and the
manifest builder keys on that string and parses the two durations back out of it:

```python
if "duration mismatch" in r:
    m = re.search(r"video([\d.]+)s.*?action([\d.]+)s", r)
    if m and float(m.group(1)) >= float(m.group(2)) * 0.95:
        continue  # video is the longer side, harmless, keep the episode
eps.add(ep)
```

Everything else on the candidate list is dropped. The 0.95 factor is the same
tolerance the detection rule used, applied in one direction only. Passing structured
data between the two stages instead of re-parsing a message would be the better
design; this is what was built.

Without this exception 101 episodes of one dataset would have been dropped for a
non-problem.

Short episodes are also kept. Episodes below the 58-frame chunk-plus-delay length are
padded by the trainer with an action-padding mask, so dropping them would only shrink
the distribution.

**Result.** 7 episodes dropped across 181 datasets: 4 empty episodes in one dataset, 1
truncated episode in another, and 2 episodes whose task strings were junk labels
(`"Failure."` and `"."`) rather than instructions. The junk pair is listed explicitly
in the manifest builder, since no automatic rule identifies a label that is
syntactically fine and semantically empty.

**Prompt normalization.** pi0.5 conditions on the language instruction, so a task
string that is not an instruction is a training defect even when the trajectory is
good. Eight datasets were rewritten, by hand, in the manifest as a
`{task_index: new_string}` map applied at merge time:

| Problem | Example | Rewritten to |
|---|---|---|
| integer placeholder | `tasks.jsonl` task field is `0` | "Stack red, green, and blue blocks on the blue dish from bottom to top." |
| Python dict repr leaked into the label | `{'color': 'black', ...}` | "Place the cubes on the bench from left to right in the following order: black, white and brown" |
| non-English label | French verb phrase | "Pull out the weed." |
| snake_case identifier | `pick_and_place_random` | "Pick up the object and place it at a random target location." |

Source files are never edited; the mapping lives in the manifest and is applied while
writing the merged repository, so the rewrite is auditable and reversible.

**Failure modes.**

- Duration checks decode container metadata; a container with no duration field falls
  back to a full decode, which is slow. Expect this stage to be the slowest metadata
  pass.
- A missing video is only detected for cameras that `info.json` declares. A camera
  present on disk but absent from `info.json` is invisible to this check and to slot
  mapping.
- Junk task labels are only found by reading the task tables. Read them.

---

## 9. Merge into one repository

**Purpose.** Produce the single LeRobot v2.1 repository that the trainer loads:
global episode numbering, three camera slots for every episode whatever the source
had, mask columns, one task table, one set of statistics.

**Script.** `26_build_manifest.py` writes `27_manifest.json`, which combines the slot
map, the deduplication result, the episode drops and the prompt rewrites into one
per-dataset instruction sheet. `28_merge_unified.py` executes it.

**Input.** `$ROOT/base_train/` (211 datasets) plus the manifest.
**Output.** `$ROOT/base_unified/`, LeRobot v2.1.

**What the merge does per episode.**

1. **Video.** For each of the three slots: if the manifest assigns a real camera,
   hardlink the source mp4 into
   `videos/chunk-{chunk:03d}/observation.images.{slot}/episode_{global:06d}.mp4`.
   Hardlinking avoids duplicating about 200 GB and takes no time; it falls back to a
   copy across filesystems.

2. **Placeholders.** For each empty slot, hardlink a cached black video of the right
   frame rate, length and start offset:

   ```bash
   ffmpeg -y -f lavfi -i color=c=black:s=128x128:r=$FPS -frames:v $N \
          -output_ts_offset $TS0 -pix_fmt yuv420p black_${FPS}_${N}_${TS0}.mp4
   ```

   They are cached by `(fps, frame_count, start_offset)`, so a corpus of 17,137
   episodes needs a few hundred distinct placeholder files rather than 20,000. 128x128
   because the content is constant and it is resized at load time anyway.

3. **Timestamps.** The parquet timestamps are re-anchored to the first present
   video's first presentation timestamp:

   ```python
   v0 = first_pts(first_present_video)          # PyAV: first decoded frame pts * time_base
   ts = ts - ts[0] + v0
   ```

   This is what keeps a 10 fps source whose video starts at 0.1 s consistent with its
   own parquet. Read pitfall 1 below: this line is correct and a later well-meaning
   "fix" overwrote its output, which cost a training run.

4. **Indices.** `frame_index` restarts at 0 per episode; `episode_index` becomes the
   global episode number; `index` continues the global frame counter; `task_index`
   points into a rebuilt global task table, after applying any prompt rewrite. Action
   and state columns are copied through untouched.

5. **Mask columns.** Three float32 columns, `observation.images.{slot}_mask`, constant
   over the episode, 1.0 for a real camera and 0.0 for a placeholder.

6. **Metadata.** `info.json` (features including the three video slots and three mask
   columns, chunk size 1000, `codebase_version: v2.1`), `episodes.jsonl`,
   `episodes_stats.jsonl` (source per-episode action and state statistics carried
   over; global statistics are recomputed in stage 10), `tasks.jsonl`.

**Pass criterion.** Episode and frame totals match the sum of the sources minus drops,
and stage 11 reports zero integrity issues.

**Result.** 17,137 episodes / 8,690,531 frames / 430 tasks / 51,411 videos / 198 GB
apparent.

**Three difficulties worth knowing about before you hit them.**

- **The 10 fps timestamp offset**, handled by the re-anchoring above. See pitfall 1.
- **Mixed resolutions cannot be collated.** Sources range from 480x640 to 1920x1080,
  and `default_collate` cannot stack them. Fixed at load time with a
  resize-with-aspect-preserving-pad transform to 224, injected into the training
  script, so no video is re-encoded and the merge stays a linking operation. The mask
  columns are not camera keys, so the image transform does not touch them.
- **The mask columns pollute normalization.** They are float32 columns that are not
  images, so a trainer that derives normalization statistics from all non-image
  features classifies them as state and rescales them, turning 0/1 into something
  else. Fixed by excluding keys ending in `_mask` from `input_features` and reading
  them per-sample in `prepare_images`. Both patches are described in the model card.

**Known defects in the emitted `info.json`.** Three, all of them in metadata rather
than in the data, and all of them from the same cause: the feature template is copied
from whichever dataset comes first in traversal order and is never reconciled against
the rest.

| Field | What it says | What is true |
|---|---|---|
| top-level `fps` | 30 | 16,687 episodes are 30 fps and 450 from five sources are 10 fps. Per-episode timestamps are correct; this one field is not. See pitfall 2 |
| `video.fps` inside each of the three video feature blocks | 10 | The majority of the corpus is 30 fps, so this is wrong for 16,687 of 17,137 episodes — and it disagrees with the top-level `fps: 30` sitting a few lines above it in the same file |
| `shape` and `video.height` / `video.width` in the same three blocks | `[480, 640, 3]`, 480 x 640 | Source resolutions run from 480x640 to 1920x1080. One resolution is declared for all three slots and every episode |

None of the three affected the training run, because the loader reads frames by
timestamp and gets the real rate and the real resolution from the video stream, and
because the training patch resizes every image to 224 at load time regardless of what
the metadata claims. They matter to anything that trusts `info.json` instead: read the
stream, not the declaration. The two `video.*` errors are the reason the traversal-order
failure mode below is not hypothetical.

**Failure modes.**

- Hardlinking across filesystems silently becomes copying. Watch free space.
- Rerunning the merge into a non-empty output directory overwrites video links but
  appends nothing consistently; delete the output directory and start over.
- The first dataset in traversal order supplies the feature template for
  `info.json`. If it is unusual (odd action dtype, unusual video feature block), the
  template is wrong for everything else. Check the emitted `info.json` against a
  source you trust.

---

## 10. Normalization statistics

**Purpose.** The policy normalizes state and action with dataset statistics. The
merged corpus spans many operators and many robot units, so per-source statistics do
not apply; one global set does.

**Script.** `29_norm_stats_robust.py`. **Input.** every episode parquet in the merged
repository. **Output.** `meta/stats.json`.

**What it computes.** Per dimension, over all 8.69M frames of `action` and
`observation.state`: `min`, `max`, `mean`, `std` (with a 1e-8 floor), `q01`, `q99`, and
`count`. Video features get placeholder entries, because visual normalization is
identity and image statistics come from ImageNet constants inside the policy.

**Why both quantiles and mean/std.** The statistics file carries both because the
build wanted the option of robust quantiles — the corpus contains extreme calibration
outliers, and a q01/q99 mapping clips them instead of letting a handful of frames
stretch the scale for everyone.

That option was not exercised, and no configuration change was needed to avoid it.
`PI05Config.normalization_mapping` in the training stack already defaults to:

```python
"VISUAL": NormalizationMode.IDENTITY,   # images: no normalization
"STATE":  NormalizationMode.MEAN_STD,   # z-score
"ACTION": NormalizationMode.MEAN_STD,   # z-score
```

so `MEAN_STD` is what the released run used, by default rather than by override, and
it is what the checkpoint's `config.json` records. Upstream comments the choice in
place: mean/std training is more stable than quantile training for this policy.
Quantiles remain available by setting the mapping explicitly, and the q01/q99 values
are in `stats.json` if you want them, but nothing in this build turns them on. **If
you reproduce the checkpoint, change nothing here.**

**Pass criterion.** q01 and q99 are finite, ordered, and within the plausible joint
range in degrees; `count` equals the corpus frame count.

**Failure modes.**

- The straightforward implementation concatenates every episode's action and state
  into two arrays before computing quantiles — roughly 200 MB each at 8.69M frames by
  6 float32 dimensions, plus the per-episode list before concatenation. It fits, but
  it is the peak memory point of the whole build. Stream it if your corpus is larger.
- Mask columns must not appear in `stats.json` as state features. They do not here,
  because only `action` and `observation.state` are collected explicitly.
- Statistics computed before the episode drops are wrong. Run this after the merge,
  never before.

---

## 11. Verify

Five checks, cheapest first. The first two and the pre-merge pair need no GPU;
`32_forward_smoke.py` and `33_train_smoke.py` do.

**`31_verify_full.py` — full integrity of the merged repository.** Reads all 17,137
episode parquets and asserts: the standard columns and the three mask columns are
present; action and state are finite; `episode_index` is globally consecutive from 0;
`index` is globally consecutive; `frame_index` is episode-local `0..n-1`; timestamps
are strictly increasing; mask values are exactly 0.0 or 1.0; all three slot videos
exist on disk for every episode. Then metadata agreement: `info.total_episodes` and
`total_frames` against the measurement, line counts of `episodes.jsonl` and
`tasks.jsonl` against `info`.

*Result: 0 issues over 17,137 episodes. Present-slot distribution 1,314 / 14,098 /
1,725 for one, two and three real cameras.*

**`30_smoke.py` — loader and collation.** Loads the repository through the training
stack's v2.1 compatibility layer, applies the same 224 resize-with-pad transform the
trainer uses, samples 12 episodes spread across the corpus, and collates them into a
batch. Asserts that mask values arrive in the batch as clean 0/1 (this is the check
that catches normalization pollution) and that the number of masked slots varies
across the batch, which proves per-episode masking is really per-episode.

*Result: pass. Heterogeneous resolutions collate to 224; masking varies within a
batch; 10 fps episodes decode.*

**`32_forward_smoke.py` — model forward.** Builds the patched pi0.5 policy from the
merged repository's metadata, checks that no `_mask` key survived into
`config.input_features`, assembles a batch that deliberately mixes a three-camera
episode with a one-camera episode, and runs a forward pass under `no_grad` and bf16.

*Result: 3.62B parameters, no `_mask` keys in `input_features`, loss 0.019 and
finite, on a single 24 GB consumer GPU at batch size 2.*

**`33_train_smoke.py` — one training step.** Opens the merged repository through the same
letterbox transform the trainer uses, builds the policy in train mode at bf16 with gradient
checkpointing, and runs forward, backward and gradient clipping on a batch of 1, then one
`bitsandbytes` AdamW8bit step. A full-precision Adam over 3.6B parameters does not fit on a
24 GB card, which is why the 8-bit optimizer is there. The pass condition is that loss and
gradient norm come out finite after backward; the optimizer step is reported when it happens
and skipped without failing when bitsandbytes is unavailable or the step runs out of memory.
The script does not read the weights back, so a reported step means `optimizer.step()`
returned, not that a before/after comparison was made. 03 §8.4 runs the equivalent check
inside the built image, against the training environment rather than the dataset build.

*Result: loss 0.057, gradient norm 4.11, both finite, optimizer step included, 21 GB peak. A
full fine-tune of this model runs on a single consumer GPU, slowly.*

**Pre-merge checks** (`13_precheck.py`, `14_video_sync.py`, described in the
interlude) must be run before the merge, because they report per-source episode
identifiers that no longer exist afterwards.

**What none of these catch.** All five smoke tests passed, and the full training run
still died twice in its first 150 steps. A 20-step smoke test does not accumulate file
descriptors, and random sampling of 20 batches has a small chance of touching the 2.6%
of episodes that carry the timestamp defect. Add a targeted, exhaustive check for
anything that affects a small subset of episodes — for example, verify the first
timestamp against the first video PTS for *every* episode, not for a sample.

---

## 12. License filtering for publication

**Purpose.** The training corpus is not redistributable as a whole. A source dataset
whose repository declares no license grants no redistribution rights, so it can be
trained on but not republished. The published dataset therefore contains only sources
with a declared upstream license.

**Scripts.** A license join (catalog tags, then local README front matter), a Hub API
verification pass for anything unresolved, `scripts/repack_open_subset.py` to rebuild
the dataset from the surviving sources, `scripts/gen_attribution.py` to emit
`dataset/ATTRIBUTION.md`.

**Resolution order.** For each of the 181 kept datasets:

1. The `license:<id>` tag in the crawl catalog's tag string.
2. Failing that, the `license:` line in the local `README.md` front matter.
3. Failing that, `GET https://huggingface.co/api/datasets/{repo}`, reading
   `cardData.license` and then the `license:` tag. The Hub API is authoritative.

Step 3 is not optional. Two artifacts make local resolution unreliable: the catalog
truncates its tag string at 120 characters, so a dataset with many tags can lose its
license tag (12 datasets hit this), and datasets converted from v3.0 have no local
README at all (stage 5). 25 datasets reached step 3, and the Hub API confirmed that
all 25 genuinely declare no license — the truncation and README artifacts explained
none of them away.

**Pass criterion.** A dataset is published only if a license identifier resolves to a
non-empty value. The repack script refuses to run if any dataset in its include list
lacks one.

**Result.**

| License | Datasets | Episodes |
|---|---|---|
| apache-2.0 | 149 | 13,587 |
| mit | 1 | 50 |
| recorded by the author | 6 | 332 |
| **Published** | **156** | **13,969** |
| Withheld, no declared license | 25 | 3,168 |

The withheld portion is 18.5% of the corpus and is spread across tasks and camera
configurations rather than concentrated in one category.

**The repack** (`scripts/repack_open_subset.py`) rebuilds the dataset from the pool
rather than filtering the merged repository, following the same traversal order and
the same per-episode rules as the merge, so the two outputs are structurally
identical. Two additions:

- provenance is preserved: `meta/sources.json` maps each source repository to its
  license, its episode count and its global episode range, and every line of
  `meta/episodes.jsonl` carries `source_dataset`;
- global statistics in `meta/stats.json` are recomputed over the new, smaller episode
  set. Reusing the old statistics would describe a different corpus.

Run it with `--dry-run` first; it counts episodes, frames and tasks without writing,
and warns if the count differs from the plan.

**Consequence to state plainly.** The published checkpoint cannot be reproduced from
the published data. 3,168 episodes that the model trained on are not in the release,
and the 450 10 fps episodes that *are* in the release were excluded from the training
run. 13,519 of the 13,969 published episodes were part of the run.

**Failure modes.**

- Reading licenses only from local files marks every converted dataset as unlicensed.
- Reading them only from a truncated catalog field marks a dozen well-licensed
  datasets as unlicensed.
- Upstream repositories change. Licenses were resolved at collection time; the
  upstream entry is authoritative at any later time, which is why `ATTRIBUTION.md`
  says so and links every source.
- Attribution is a license obligation, not a courtesy: Apache-2.0 Section 4 requires
  carrying the notice forward. Generate the attribution file from the same data the
  repack uses, so the two cannot drift.

---

## Data pitfall 1: video PTS that does not start at zero

**The symptom.** A training run dies mid-flight, in a dataloader worker, with either
a timestamp tolerance assertion or an out-of-bounds frame request. It does not
happen at step 0. In this build it happened at step 140 of a 40,000-step run on 8
GPUs, and once before that at the very first step of a different attempt.

**The mechanism.** LeRobot pairs each parquet row with a video frame by timestamp and
checks that the decoded frame's presentation timestamp matches the requested one
within a tight tolerance (1e-4 s). Some recorders, particularly at 10 fps, write a
first video PTS of 0.1 s while the parquet timestamps start at 0.0. The two are then
offset by one frame interval for the whole episode. Two distinct crashes follow from
the same offset:

- **Tolerance violation.** Parquet says 0.0, video's first frame is at 0.1, the
  difference exceeds tolerance, `AssertionError` in a worker. Under DDP one dead
  worker takes the whole job down.
- **Out-of-bounds frame index.** Where the offset goes the other way (parquet
  timestamps carrying an extra +0.1 while `info.json` claims 30 fps), the decoder
  computes `round(timestamp * video_fps)` for the last frame and asks for one frame
  past the end of the video. There is no clamp in the loader, so the decoder raises
  immediately.

**How this build managed to hit it twice.** The merge already handles it: it reads
the first present video's PTS with PyAV and re-anchors the parquet timestamps to it
(`ts = ts - ts[0] + v0`, stage 9, step 3). That is correct. A later repair, made to
fix the out-of-bounds crash, rewrote the timestamps of 100 episodes to a clean
`i/10` starting at 0.0 — and thereby overwrote the merge's correct anchoring, which
converted an out-of-bounds crash into a tolerance violation 450 episodes wide. **Do
not "normalize" timestamps after a merge that anchored them deliberately.** If the
merge output looks wrong, rerun the merge; do not post-process its timestamps.

**Diagnosis.** For every episode, compare the first parquet timestamp against the
first video PTS. Exhaustively, not on a sample — the affected fraction here was 2.6%,
which random sampling misses.

```python
import av, pyarrow.parquet as pq

def first_pts(path):
    c = av.open(path)
    try:
        s = c.streams.video[0]
        for f in c.decode(video=0):
            return float(f.pts * s.time_base) if f.pts is not None else 0.0
        return 0.0
    finally:
        c.close()

ts0 = pq.read_table(ep_parquet, columns=["timestamp"]).column("timestamp")[0].as_py()
pts0 = first_pts(ep_video)
assert abs(ts0 - pts0) < 1e-4, f"{ep_parquet}: parquet {ts0} vs video {pts0}"
```

Also check the measured frame interval against the declared rate, which is the
companion defect:

```python
dt = numpy.median(numpy.diff(timestamps))
assert abs(1 / dt - info["fps"]) < info["fps"] * 0.25
```

**Repair, in order of preference.**

1. **Re-anchor.** Rewrite each affected episode's timestamps as
   `ts - ts[0] + first_pts(video)`. This is exactly what the merge does; running the
   merge in an environment where PyAV reports PTS correctly produces it for free.
2. **Exclude.** Drop the affected episodes. This is what the released training run
   did: 450 episodes from five 10 fps sources were excluded and 16,687 of 17,137
   episodes were used. It costs 2.6% of the corpus and takes minutes rather than a
   rebuild.
3. **Clamp in the loader.** Rejected. Clamping the frame index stops the crash but
   leaves the frame-to-action pairing off by one for the whole episode, which is a
   silent data defect instead of a loud one.

**Also fix the source of truth.** In this build the repair was applied to the training
machine's local cache and never propagated back to the archived copy of the dataset,
so the defect is still present upstream and will come back on the next run that syncs
from it. Fix the artifact you keep, not the copy you are about to delete.

---

## Data pitfall 2: mixed frame rates behind a single `fps` field

**The symptom.** Everything validates, the loader works, batches collate, and a small
subset of episodes produces frame-to-action pairs that are wrong by a constant
factor — or, if you are lucky, an assertion.

**The mechanism.** LeRobot's `info.json` carries one `fps` for the whole repository.
The merge writes `fps: 30` because that is the majority. Five of the 181 sources
recorded at 10 fps: 450 episodes, 2.6% of the corpus.

| Source rate | Datasets | Episodes |
|---|---|---|
| 30 fps | 176 | 16,687 |
| 10 fps | 5 | 450 |

The five are worth naming, because they are the ones a reproduction has to decide
about:

| Source repository | Episodes | Frames |
|---|---|---|
| `CoRL2026-CSI/SO101-teleop_stack_RGBblock_on_bluedish_150epi_10fps` | 150 | 32,702 |
| `CoRL2026-CSI/IsaacLab-SO101-PullCube-100epi-10fps-appendix` | 100 | 32,208 |
| `anvilbot-patrickhhh/SO101_relocate_cube_2cams_record_2` | 100 | 15,000 |
| `anvilbot-patrickhhh/SO101_PickAndPlace_front_wrist` | 50 | 7,500 |
| `anvilbot-patrickhhh/SO101_PickAndPlace_3cams` | 50 | 7,500 |
| **total** | **450** | **94,910** |

Two of them declare the rate in the repository name, which is the only reason the
group was identifiable after the merge erased per-source identity.

Per-episode timestamps are correct (the merge preserves each source's real timing).
The single `fps` field is not. Any code that converts a timestamp to a frame index
using `info["fps"]` rather than the video's own rate mis-indexes those 450 episodes
by a factor of three, and any tolerance check derived from the declared rate fires
spuriously. This is the second half of the failure in pitfall 1: the 0.1 s offset and
the wrong declared rate arrived together, from the same five sources.

**Why testing missed it.** Random sampling in a 20-step smoke test touches a 2.6%
subset with low probability, and the per-episode consistency check that would have
caught it deterministically (`13_precheck.py`, measured `dt` versus declared `fps`)
was run against the pool before the merge, where every source's `fps` was still
correct for its own episodes. The defect is created *by* the merge and only exists
afterwards.

**Detection.** After merging, verify per episode rather than per repository:

```python
declared = info["fps"]
for ep in episodes:
    dt = numpy.median(numpy.diff(timestamps(ep)))
    if abs(1 / dt - declared) > declared * 0.25:
        print(ep, "measured", round(1 / dt, 1), "declared", declared)
```

and cross-check against the video stream's own `average_rate`, which is the real
answer when the two disagree.

**Options.**

- **Trust `timestamp`, never the declared rate.** Correct, and costs nothing if the
  timestamps are anchored as in pitfall 1. This is the recommended posture for any
  code you write against the dataset.
- **Keep the rate per source.** More honest metadata, but the LeRobot v2.1 `info.json`
  has one field, so it means carrying a side table and teaching every consumer to read
  it.
- **Resample the minority to the majority rate.** Re-encodes video and interpolates
  actions; changes the data. Not done here.
- **Exclude the minority.** What the released run did.

Decide before the merge, not at step 140.

### Producing the corpus that was actually trained

The merged repository is 17,137 episodes. The released checkpoint saw 16,687. This is
the step that turns one into the other, and it happened mid-run on the training
instance rather than in the build, which is why it has no stage number.

The filter is on measured frame interval, not on source name: an episode whose median
`diff(timestamp)` is about 0.1 s is 10 fps and is dropped. That selects exactly the
450 episodes of the five sources above without needing per-source identity, which the
merge has already erased.

What the rewrite has to do, in order:

1. **Drop the matching episode parquets and their videos** from the copy being fed to
   the trainer.
2. **Re-index.** `episode_index` and the global `index` column must both come out
   contiguous from 0, because the integrity check in stage 11 and LeRobot's own
   episode lookup both assume it. `frame_index` is episode-local and needs no change.
3. **Rewrite `meta/`.** `episodes.jsonl` loses the dropped rows, `info.json`'s
   `total_episodes`, `total_frames` and `total_videos` come down to the new counts.
   The task table can stay as is; a task that now has no episodes is inert.
4. **Link, do not copy.** The surviving videos were hardlinked from the merged
   repository, so the reduced copy costs its parquet files and nothing else.

Result: 16,687 episodes / 8,595,621 frames, which is what the epoch count in 04 §4 is
computed against.

**Two things this run did not do, recorded as they happened rather than as they should
have been:**

- **Normalization statistics were not recomputed.** `meta/stats.json` was copied over
  unchanged, so the mean and standard deviation the policy normalizes with include the
  2.6% of frames the run never saw. The effect is small and in a known direction, but
  it is an inconsistency; `29_norm_stats_robust.py` against the reduced repository is
  the correct move and costs one CPU pass.
- **The archived copy of the merged dataset was never repaired.** The filter was
  applied to the instance's local cache and thrown away with the instance. Anything
  that syncs the merged corpus again gets all 17,137 episodes back, defect included.
  Fix the artifact you keep, not the copy you are about to delete.

---

## Minimum reproduction

Environment, once:

```bash
export ROOT=/path/to/workspace
export REF=/path/to/reference_dataset       # your own SO-101 recording, follower frame,
                                            # or a green public dataset (stage 3)
export CONVERTER=/path/to/lerobot-v3-v2-converter
                                            # git clone https://github.com/jinnymo/lerobot-v3-v2-converter
export HF_HOME=$ROOT/.hf_cache
export HF_TOKEN=<your token>                # removes anonymous rate limits
mkdir -p "$ROOT"/{external_hf,base_train,base_unified,base_open}
```

Run them from `pipeline/`. The spreadsheet paths are relative, so the working directory is
what keeps the catalog and the analysis workbook in one place; the `artifacts/` defaults
are anchored to the script's own directory and resolve from anywhere. The five stages that
take no `--root` are invoked accordingly below.

```bash
cd pipeline

# 1-2  crawl the Hub and screen the catalog          -> ./02_so101_catalog.{csv,xlsx}
python 01_hf_so101_crawler.py --full
python 01_hf_so101_crawler.py --rebuild             # xlsx from the csv, no network

# 3    action-space conformity vs the reference      -> ./03_action_match.{csv,xlsx}
python 03_action_match.py --ref "$REF"
python 03_action_match.py --ref "$REF" --retry      # transient fetch failures only
python 03_action_match.py --ref "$REF" --video      # attach preview links (optional)

# 4    download the green set                        -> $ROOT/external_hf/
python 04_download_green.py --root "$ROOT"
#      any repository that fails twice: git lfs clone it by hand

# 5    LeRobot v3.0 -> v2.1                          (in a LeRobot environment)
python 05_convert_v3_to_v2.py --root "$ROOT" --converter "$CONVERTER"

# 6    precision analysis, tiering, integrity, metadata repair
python 06_analyze_datasets.py --root "$ROOT" --ref "$REF"   # -> ./so101_external_analysis.xlsx
python 07_classify_camera.py  --root "$ROOT" --report so101_external_analysis.xlsx
#      MANUAL, and stage 8 depends on it. Stage 7 leaves confirmed/tier12,
#      confirmed/tier3, undecided and excluded. By hand: split tier12 into
#      confirmed/tier1 (an overhead-type AND a wrist camera) and confirmed/tier2
#      (one of the two), then review the undecided datasets visually and move the
#      usable ones into confirmed/tier2b. Stage 8 reads directory membership, so
#      whatever you decide here is what the report records.
python 08_add_class.py        --root "$ROOT" --report so101_external_analysis.xlsx
python 09_verify_integrity.py --root "$ROOT"
python 10_finalize_meta.py    --root "$ROOT" --report so101_external_analysis.xlsx
python 11_summary.py                        --report so101_external_analysis.xlsx
python 12_prompt_none.py                    --report so101_external_analysis.xlsx
#      MANUAL: 12 only produces the worklist of datasets with no task string. The
#      prompts written for this build are the TASK_NORM table inside
#      26_build_manifest.py; a different pool needs its own.
#      MANUAL: assemble the surviving datasets into $ROOT/base_train/{external,self}

# 7    pre-merge checks (must run before the merge; per-source ids vanish after it)
python 13_precheck.py   --root "$ROOT"
python 14_video_sync.py --root "$ROOT"

# 8    build the merge instruction sheet
python 20_camera_slot_map.py --root "$ROOT"         # -> artifacts/21_camera_slot_map.json
python 22_dedup_verify.py    --root "$ROOT"         # -> artifacts/23_dedup_result.json
python 24_ep_drop.py         --root "$ROOT"         # -> artifacts/25_ep_drop.json
python 26_build_manifest.py                         # -> artifacts/27_manifest.json
#      26 takes --artifacts and --out, not --root; both already default to
#      pipeline/artifacts, so it needs no argument here.

# 9    merge                                         -> $ROOT/base_unified/
python 28_merge_unified.py --root "$ROOT"           # --limit N for a partial trial run

# 10   global normalization statistics               -> meta/stats.json
python 29_norm_stats_robust.py --root "$ROOT"

# 11   verify
python 31_verify_full.py   --root "$ROOT"           # full integrity, no GPU
python 30_smoke.py         --root "$ROOT"           # loader, collation, masks
python 32_forward_smoke.py --root "$ROOT"           # model forward, needs a GPU
python 33_train_smoke.py   --root "$ROOT"           # one optimizer step, needs a GPU
#      30, 32 and 33 need the training package on the path: --vlash-src for a
#      checkout that is not installed. 32 and 33 also take --paligemma for the
#      tokenizer directory.

# 11b  NO SCRIPT SHIPS FOR THIS. Exhaustive per-episode timestamp check
#      (see pitfalls 1 and 2): first parquet timestamp == first video PTS, and
#      measured dt == declared fps. 31_verify_full.py does not cover it; the
#      detection snippets in the two pitfall sections are what was used.

# 11c  NO SCRIPT SHIPS FOR THIS EITHER. The reduced corpus the released run
#      actually trained on: drop episodes whose median dt is ~0.1 s, re-index
#      episode_index and index, rewrite meta/. It was done ad hoc on the training
#      instance after the run had already started and failed, against that
#      instance's local cache, and the code was discarded with the instance.
#      See "Producing the corpus that was actually trained" above for the four
#      things the rewrite has to do.

# 12   license-filtered release                      -> $ROOT/base_open/
cd ../scripts
python repack_open_subset.py --src "$ROOT/base_train" --dst "$ROOT/base_open" --dry-run
python repack_open_subset.py --src "$ROOT/base_train" --dst "$ROOT/base_open"
python gen_attribution.py --src "$ROOT/base_train"   # -> dataset/ATTRIBUTION.md
```

Checkpoints worth stopping at:

| After | Expect |
|---|---|
| stage 3 | green count in the low hundreds; zero strong-leader-suspicion rows |
| stage 4 | downloaded repository count within 1-2 of the green count |
| stage 7 | deduplication drops 10-20% of datasets; inspect the cluster report by hand |
| stage 9 | episode and frame totals equal the sum of sources minus drops |
| stage 11 | `31_verify_full.py` reports zero issues, and the exhaustive timestamp check is clean |
| stage 12 | dry-run episode count equals the plan |

## Related documents

- Model card, training recipe and the four issues that surfaced during the run:
  `model/README.md`
- Per-source provenance, licenses and the list of modifications applied to each
  source: `dataset/ATTRIBUTION.md`
- Dataset schema, camera slots and loading instructions: `dataset/README.md`
