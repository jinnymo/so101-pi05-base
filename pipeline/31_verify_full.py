# SPDX-License-Identifier: Apache-2.0
"""Stage 11b: exhaustive integrity check of the merged repository (no GPU needed).

Every episode parquet:
  - the standard columns (action, observation.state, timestamp, frame_index,
    episode_index, index, task_index) plus the three mask columns are present
  - mask values are exactly 0.0 or 1.0, action and state contain no NaN or inf
  - episode_index is globally continuous 0..N-1, index is globally continuous,
    frame_index is episode-local 0..n-1
  - timestamp is strictly increasing
  - all three slot videos exist on disk

Metadata consistency: info.total_episodes and info.total_frames against the measured
totals, and the line counts of tasks.jsonl and episodes.jsonl.

Input:  <root>/base_unified/
Output: a console report.

Run:
  python 31_verify_full.py --root /path/to/workspace
"""

import argparse
import glob
import json
import os

SLOTS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
STD_COLS = {"action", "observation.state", "timestamp", "frame_index",
            "episode_index", "index", "task_index"}
MASK_COLS = {f"observation.images.{s}_mask" for s in SLOTS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get("ROOT", "."),
                    help="workspace root (default: $ROOT)")
    ap.add_argument("--dst", default=None,
                    help="merged repository (default: <root>/base_unified)")
    args = ap.parse_args()

    import numpy as np
    import pyarrow.parquet as pq

    dst = args.dst or os.path.join(args.root, "base_unified")
    info = json.load(open(f"{dst}/meta/info.json"))
    files = sorted(glob.glob(f"{dst}/data/chunk-*/episode_*.parquet"))
    print(f"full integrity check: {len(files)} episode parquet files "
          f"(info.total_episodes={info['total_episodes']})")

    issues = []
    exp_ep, exp_idx, total_rows = 0, 0, 0
    mask_dist = {1: 0, 2: 0, 3: 0}  # episodes by number of present slots
    for f in files:
        try:
            t = pq.read_table(f)
        except Exception as e:
            issues.append(f"{os.path.basename(f)} read failed:{str(e)[:30]}")
            continue
        cols = set(t.column_names)
        if not STD_COLS <= cols:
            issues.append(f"{os.path.basename(f)} missing standard columns:{STD_COLS-cols}")
        if not MASK_COLS <= cols:
            issues.append(f"{os.path.basename(f)} missing mask columns:{MASK_COLS-cols}")
            continue
        n = t.num_rows
        act = np.asarray(t.column("action").to_pylist(), dtype=float)
        st = np.asarray(t.column("observation.state").to_pylist(), dtype=float)
        ei = np.asarray(t.column("episode_index").to_pylist())
        idx = np.asarray(t.column("index").to_pylist())
        fi = np.asarray(t.column("frame_index").to_pylist())
        ts = np.asarray(t.column("timestamp").to_pylist(), dtype=float)
        if not np.isfinite(act).all() or not np.isfinite(st).all():
            issues.append(f"ep{exp_ep} NaN/inf")
        if not (ei == exp_ep).all():
            issues.append(f"ep{exp_ep} episode_index mismatch ({ei[0]})")
        if not np.array_equal(idx, np.arange(exp_idx, exp_idx + n)):
            issues.append(f"ep{exp_ep} index not continuous")
        if not np.array_equal(fi, np.arange(n)):
            issues.append(f"ep{exp_ep} frame_index not continuous")
        if not (np.diff(ts) > 0).all():
            issues.append(f"ep{exp_ep} timestamp not monotonic")
        present = 0
        for s in SLOTS:
            mv = set(np.unique(np.asarray(t.column(f"observation.images.{s}_mask").to_pylist())))
            if not mv <= {0.0, 1.0}:
                issues.append(f"ep{exp_ep} {s}_mask not 0/1:{mv}")
            if mv == {1.0}:
                present += 1
            chunk = exp_ep // info["chunks_size"]
            vp = (f"{dst}/videos/chunk-{chunk:03d}/observation.images.{s}/"
                  f"episode_{exp_ep:06d}.mp4")
            if not os.path.exists(vp):
                issues.append(f"ep{exp_ep} {s} video missing")
        mask_dist[present] = mask_dist.get(present, 0) + 1
        exp_ep += 1
        exp_idx += n
        total_rows += n
        if exp_ep % 3000 == 0:
            print(f"  {exp_ep}/{len(files)}  ({len(issues)} findings)", flush=True)

    print("\n=== metadata consistency ===")
    print(f"  episodes: measured {exp_ep} vs info {info['total_episodes']} "
          f"{'OK' if exp_ep==info['total_episodes'] else 'MISMATCH'}")
    print(f"  frames: measured {total_rows} vs info {info['total_frames']} "
          f"{'OK' if total_rows==info['total_frames'] else 'MISMATCH'}")
    eps_lines = sum(1 for _ in open(f"{dst}/meta/episodes.jsonl"))
    tasks_lines = sum(1 for _ in open(f"{dst}/meta/tasks.jsonl"))
    print(f"  episodes.jsonl: {eps_lines} {'OK' if eps_lines==exp_ep else 'MISMATCH'}")
    print(f"  tasks.jsonl: {tasks_lines} vs info {info['total_tasks']} "
          f"{'OK' if tasks_lines==info['total_tasks'] else 'MISMATCH'}")
    print(f"  present-slot distribution: {mask_dist}")

    print(f"\n=== integrity findings: {len(issues)} ===")
    for x in issues[:30]:
        print(f"  {x}")
    print(f"\n{'FULL INTEGRITY PASS' if not issues else f'FAIL: {len(issues)} findings'}")


if __name__ == "__main__":
    main()
