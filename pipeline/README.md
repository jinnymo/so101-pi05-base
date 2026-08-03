# Dataset construction pipeline

The scripts that built the training set: crawl the Hugging Face Hub for SO-101 datasets,
screen them for action-space conformity, download and normalize them, deduplicate, and
merge everything into one LeRobot v2.1 repository with the three-slot camera layout that
pi0.5 expects.

The narrative version, with the reasoning behind each threshold and the results measured
at each stage, is `../docs/02-dataset-construction.md`. This file is the operating manual.

The number prefixes are the execution order and are load-bearing: several scripts read the
numbered output of an earlier one. Gaps in the numbering (02, 15-19, 21, 23, 25, 27) are
data files, not missing scripts.

## Before you start

```bash
export ROOT=/path/to/workspace          # every script defaults --root to this
export REF=/path/to/reference_dataset   # stages 3 and 6a default --ref to this
export HF_HOME=$ROOT/.hf_cache          # stage 4 downloads terabytes, keep it off /
export HF_TOKEN=<your token>            # removes anonymous rate limits
mkdir -p "$ROOT"/{external_hf,base_train,base_unified}
```

Paths are command-line arguments, never constants inside the files. `--root` defaults to
`$ROOT` and then to `.`, so the commands below work unchanged once the exports are set.
Every script derives its own directories from `--root`:

| Directory | Written by | Read by |
|---|---|---|
| `<root>/external_hf/` | stage 4 | stages 5-6 |
| `<root>/base_train/` | you, by hand, at the end of stage 6 | stages 7-9 |
| `<root>/base_unified/` | stage 9 | stages 10-11 |

`pip install -r requirements.txt` covers stages 1 to 10. Stage 5 additionally needs LeRobot
and a checkout of the v3.0 to v2.1 converter; stage 11 runs inside the training environment.

```bash
git clone https://github.com/jinnymo/lerobot-v3-v2-converter
export CONVERTER=$PWD/lerobot-v3-v2-converter   # stage 5 defaults --converter to this
```

The converter vendors NVIDIA's Isaac-GR00T conversion script verbatim, so no Isaac-GR00T
checkout is needed. Stage 5 runs its `v3_to_v2/convert.py` once per dataset.

## Stage 3 needs a reference recording, and that normally means hardware

`03_action_match.py` is the stage that decides which datasets are usable at all. It
compares every candidate against a reference dataset: the joint names and their order, the
unit the action column is written in, and the mean gap between `action` and
`observation.state`. That last number is the important one. A large gap means the action
column was recorded in the *leader* arm's calibration frame rather than the follower's,
which injects a constant per-joint offset into every training target. It trains cleanly and
diverges in closed loop.

The reference used for this build was a recording made on our own SO-101, deliberately
saved in the follower frame. **Without an SO-101 you cannot produce that recording.**

**Workaround: use a public dataset as the reference.** The comparison is relative, so any
dataset already in the follower frame with the standard joint layout works. Datasets this
pipeline itself flagged green are, by construction, exactly that. Three that were measured
during this build (Apache-2.0, `action`/`state` offset well inside the green threshold):

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

Those numbers are from the crawl this repository documents, not from a live check; a
repository can be re-uploaded or deleted. Re-verify before you trust one: after stage 1,
look up the row in `02_so101_catalog.csv`, and after a first pass of stage 3 confirm the
candidate you picked comes back green against itself. `03_action_match.py` accepts either
`meta/stats.json` or `meta/episodes_stats.jsonl` as the reference statistics, so a plain
downloaded v2.1 dataset works as-is.

Two consequences of borrowing a reference: the range-overlap column becomes overlap with
*that* robot's workspace rather than yours, and any systematic calibration offset in the
reference is inherited by the whole screen. Prefer a reference with many episodes and wide
joint coverage, and read the `unit` distribution that stage 3 prints as a sanity check.

## Order of execution

```bash
# 1-2  crawl the Hub, screen the catalog        -> 02_so101_catalog.{csv,xlsx}
python 01_hf_so101_crawler.py --full
python 01_hf_so101_crawler.py --rebuild

# 3    action-space conformity vs the reference -> 03_action_match.{csv,xlsx}
python 03_action_match.py
python 03_action_match.py --retry              # transient fetch failures only
python 03_action_match.py --video              # optional preview links

# 4    download the green set                   -> <root>/external_hf/
python 04_download_green.py

# 5    LeRobot v3.0 -> v2.1 (needs LeRobot + the converter checkout)
python 05_convert_v3_to_v2.py --converter "$CONVERTER"

# 6    analysis, tiering, integrity, metadata repair
python 06_analyze_datasets.py                  # -> so101_external_analysis.xlsx
python 07_classify_camera.py                   # moves datasets into tier directories
#      manual step, see below
python 08_add_class.py
python 09_verify_integrity.py
python 10_finalize_meta.py
python 11_summary.py
python 12_prompt_none.py
#      assemble the survivors into <root>/base_train/{external,self}

# 7    pre-merge checks (source identity is gone after the merge)
python 13_precheck.py
python 14_video_sync.py

# 8    build the merge instruction sheet
python 20_camera_slot_map.py                   # -> artifacts/21_camera_slot_map.json
python 22_dedup_verify.py                      # -> artifacts/23_dedup_result.json
python 24_ep_drop.py                           # -> artifacts/25_ep_drop.json
python 26_build_manifest.py                    # -> artifacts/27_manifest.json

# 9    merge                                    -> <root>/base_unified/
python 28_merge_unified.py                     # --limit N for a partial trial run

# 10   global normalization statistics          -> <root>/base_unified/meta/stats.json
python 29_norm_stats_robust.py

# 11   verify
python 31_verify_full.py                       # full integrity, no GPU
python 30_smoke.py                             # loader, collation, masks
python 32_forward_smoke.py                     # model forward, needs a GPU
python 33_train_smoke.py                       # one optimizer step, needs a GPU
```

## What each script does

| Script | Role |
|---|---|
| `01_hf_so101_crawler.py` | Searches the Hub by name, fetches each repository's metadata, writes the catalog with a coarse category per dataset. |
| `03_action_match.py` | Compares every candidate's action space against the reference and assigns green / yellow / red. The screen that matters. |
| `04_download_green.py` | Downloads the full snapshot of every green repository, resumable through a per-repository marker file. |
| `05_convert_v3_to_v2.py` | Converts LeRobot v3.0 trees to v2.1 in place, keeping the original as `<name>_v3.0`. |
| `06_analyze_datasets.py` | Per-dataset precision analysis: cameras, action ranges, trajectory jumps, idle episodes, desync, black videos. |
| `07_classify_camera.py` | Sorts datasets into tier directories by camera viewpoint, capping front-only views at a quarter of the episode budget. |
| `08_add_class.py` | Writes the tier a dataset ended up in back into the report as a `class` column. |
| `09_verify_integrity.py` | Checks parquet count, row totals, per-camera video count and first-episode frame count against the declared metadata. |
| `10_finalize_meta.py` | Repairs `info.json` totals from `episodes.jsonl` where they disagree. Metadata only, data untouched. |
| `11_summary.py` | Aggregates the kept pool into a `summary` sheet: per class, per viewpoint, fps/resolution/codec, totals. |
| `12_prompt_none.py` | Lists kept datasets with no task string, with a name-derived hint and a column to write the real prompt into. |
| `13_precheck.py` | Reads every episode: NaN/inf, frame_index continuity, timestamp monotonicity, fps consistency, extreme values. |
| `14_video_sync.py` | Compares video frame count against parquet row count for every episode and camera. |
| `20_camera_slot_map.py` | Assigns each dataset's cameras to `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`, marking empty slots as masked. |
| `22_dedup_verify.py` | Finds re-uploads, cumulative uploads and split copies by action fingerprint, ignoring owner names. |
| `24_ep_drop.py` | Lists episodes to drop: empty ones, missing videos, and video/action duration mismatch beyond 5%. |
| `26_build_manifest.py` | Combines the three files above, plus manual drops and prompt rewrites, into the merge instruction sheet. |
| `28_merge_unified.py` | Executes the manifest: hard-links videos, renumbers indices globally, writes mask columns, emits one v2.1 repository. |
| `29_norm_stats_robust.py` | Computes the global action and state statistics: min, max, mean, std, q01, q99, count. pi0.5 normalizes with mean and std. |
| `30_smoke.py` | Loads the merged repository, collates mixed resolutions, checks that the mask columns arrive intact. |
| `31_verify_full.py` | Exhaustive integrity check of the merged repository. No GPU. |
| `32_forward_smoke.py` | One forward pass of the patched policy on a masked batch. Needs a GPU. |
| `33_train_smoke.py` | Forward, backward and one optimizer step. Needs a GPU. |

## Manual steps

The pipeline is not one button. Three points need a person:

1. **Between `07_classify_camera.py` and `08_add_class.py`.** Stage 7 produces
   `confirmed/tier12`, `confirmed/tier3`, `undecided` and `excluded`. Split `tier12` into
   `confirmed/tier1` (both an overhead-type and a wrist camera) and `confirmed/tier2` (one
   of the two), then review the `undecided` datasets visually and move the usable ones to
   `confirmed/tier2b`. Stage 8 reads directory membership, so whatever you decide is what
   the report records.
2. **After stage 6.** Assemble the surviving datasets into `<root>/base_train/external/<tier>/`
   and `<root>/base_train/self/` before running stages 7 onward.
3. **Prompt rewrites.** `12_prompt_none.py` produces the worklist; the prompts that were
   written for this build are the `TASK_NORM` table inside `26_build_manifest.py`.

## artifacts/

The intermediate files produced by stage 8 for this build, kept so the merge can be
reproduced without re-running the analysis:

| File | Contents |
|---|---|
| `21_camera_slot_map.json` | Camera-to-slot assignment for all 211 datasets. |
| `23_dedup_result.json` | Keep / drop with the relation and episode counts. |
| `25_ep_drop.json` | Per-dataset episode drops with the reason text. |
| `27_manifest.json` | The merge instruction sheet, 211 datasets, 181 kept. |

`27_manifest.json` is required: `28_merge_unified.py` and the release scripts in
`../scripts/` read it. The other three are inputs to `26_build_manifest.py`, which
regenerates the manifest from them.

Dataset keys are `external/<tier>/<owner>__<name>` or `self/<name>`, matching the
`base_train` layout. The tiers are `tier1`, `tier2`, `tier2b` and `tier3`; `tier2b` is the
group that was held back after stage 7 and admitted after a visual review.

The reason strings in `25_ep_drop.json` are parsed by `26_build_manifest.py`: it matches
`duration mismatch` and reads the two durations back out of the text to tell a truncated
video (dropped) from a video that merely runs longer than the action stream (kept, since
the loader only decodes the range the actions cover). Changing the wording in
`24_ep_drop.py` means changing the matcher in `26_build_manifest.py` too.

## Costs and caveats

- Stage 4 downloaded 303 GB in the reference build and takes hours to days depending on the
  connection. `--limit` exists for a trial run.
- Stages 6, 13, 14 and 24 decode video and read every parquet file; they are hours-long on
  a full pool. All of them are thread-pooled, `--workers` is tunable.
- Stage 9 hard-links source videos into the merged tree, so `base_train` and `base_unified`
  must be on the same filesystem or the link silently degrades to a copy and the size
  doubles.
- Stages 30, 32 and 33 need the training package on the path. Pass `--vlash-src` if it is
  not installed, and `--paligemma` for the tokenizer directory.
- The episode counts quoted here (211 datasets, 181 kept) describe this build. A crawl run
  today will find a different pool.
