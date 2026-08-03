# SPDX-License-Identifier: Apache-2.0
"""Rebuild the unified LeRobot v2.1 dataset from the license-declared subset only.

Traversal order and per-episode rules match 28_merge_unified.py (sorted dataset keys,
sorted parquet glob, manifest drop_eps and task_normalize, 3-slot camera layout with
black dummy videos for missing slots). Two differences:

  - only the 156 datasets listed in repack_plan.json["include"] are merged
  - provenance is preserved: meta/episodes.jsonl carries source_dataset and
    meta/sources.json maps each source to its license and global episode range

Global normalization statistics in meta/stats.json are recomputed from the new
episode set; values from the previous merge are not reused.

Usage:
    python repack_open_subset.py [--dry-run]
"""

import argparse
import glob
import json
import os
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = f"{HERE}/repack_plan.json"
LICENSES = f"{HERE}/license_join.json"
DEFAULT_MANIFEST = os.path.normpath(f"{HERE}/../pipeline/artifacts/27_manifest.json")

SLOTS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
CHUNK = 1000
BLACK_W, BLACK_H = 128, 128

# Set from command-line arguments in main().
SRC = None
DST = None
MANIFEST = None
FFMPEG = "ffmpeg"

_black_cache = {}


def find_video(d, cam, epn):
    for pat in (f"{d}/videos/chunk-*/{cam}/episode_{epn:06d}.mp4",
                f"{d}/videos/{cam}/chunk-*/*{epn:06d}.mp4"):
        g = glob.glob(pat)
        if g:
            return g[0]
    return None


def link_or_copy(src, dst):
    """Hardlink to avoid duplicating ~171 GB; fall back to copy across filesystems."""
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def first_pts(path):
    """First video frame PTS in seconds, used to align parquet timestamps."""
    import av
    c = av.open(path)
    try:
        s = c.streams.video[0]
        for f in c.decode(video=0):
            return float(f.pts * s.time_base) if f.pts is not None else 0.0
        return 0.0
    finally:
        c.close()


def make_black(fps, n, ts0=0.0):
    key = (round(fps), n, round(ts0, 3))
    if key in _black_cache:
        return _black_cache[key]
    bdir = f"{DST}/.black"
    os.makedirs(bdir, exist_ok=True)
    path = f"{bdir}/black_{round(fps)}_{n}_{round(ts0 * 1000)}.mp4"
    if not os.path.exists(path):
        cmd = [FFMPEG, "-y", "-f", "lavfi",
               "-i", f"color=c=black:s={BLACK_W}x{BLACK_H}:r={fps}",
               "-frames:v", str(n)]
        if ts0:
            cmd += ["-output_ts_offset", f"{ts0:.4f}"]
        cmd += ["-pix_fmt", "yuv420p", path]
        subprocess.run(cmd, check=True, capture_output=True)
    _black_cache[key] = path
    return path


def stat_dim(X):
    return {
        "min": X.min(0).tolist(),
        "max": X.max(0).tolist(),
        "mean": X.mean(0).tolist(),
        "std": (X.std(0) + 1e-8).tolist(),
        "q01": np.quantile(X, 0.01, axis=0).tolist(),
        "q99": np.quantile(X, 0.99, axis=0).tolist(),
        "count": [int(len(X))],
    }


def load_plan():
    plan = json.load(open(PLAN))
    include = set(plan["include"])
    lic = {r["key"]: r for r in json.load(open(LICENSES))["keep"]}
    mani = json.load(open(MANIFEST))["datasets"]
    keep = {k: v for k, v in mani.items() if v["keep"] and k in include}
    missing = include - set(keep)
    if missing:
        raise SystemExit(f"in plan but not kept in manifest: {sorted(missing)}")
    unlicensed = [k for k in include if not lic.get(k, {}).get("lic")]
    if unlicensed:
        raise SystemExit(f"in plan but no license on record: {sorted(unlicensed)}")
    return plan, keep, lic


def main():
    global SRC, DST, MANIFEST, FFMPEG
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="directory holding the per-source datasets "
                         "(external/tier*/<owner>__<name> and self/<name>)")
    ap.add_argument("--dst", required=True, help="output directory for the merged dataset")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="27_manifest.json (default: ../pipeline/artifacts/27_manifest.json)")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary used for dummy videos")
    ap.add_argument("--dry-run", action="store_true",
                    help="count episodes/frames/tasks without writing anything")
    a = ap.parse_args()
    SRC, DST, MANIFEST, FFMPEG = a.src, a.dst, a.manifest, a.ffmpeg
    dry = a.dry_run

    plan, keep, lic = load_plan()
    print(f"plan: include {len(plan['include'])} / exclude {len(plan['exclude'])} datasets, "
          f"expected {plan['include_ep']} episodes")
    if dry:
        print("dry run: no files will be written")
    else:
        for sub in ("data", "videos", "meta"):
            os.makedirs(f"{DST}/{sub}", exist_ok=True)

    global_ep, global_idx = 0, 0
    task2idx = {}
    episodes = []
    ep_stats_lines = []
    sources = {}
    acts, states = [], []
    feat_template = None

    for i, (rel, e) in enumerate(sorted(keep.items()), 1):
        d = f"{SRC}/{rel}"
        info = json.load(open(f"{d}/meta/info.json"))
        fps = float(info.get("fps", 30))
        feats = info.get("features", {})
        if feat_template is None:
            feat_template = feats
        orig_tasks = {}
        for line in open(f"{d}/meta/tasks.jsonl"):
            if line.strip():
                j = json.loads(line)
                orig_tasks[j["task_index"]] = j["task"]
        ep_stat = {}
        sp = f"{d}/meta/episodes_stats.jsonl"
        if os.path.exists(sp):
            for line in open(sp):
                if line.strip():
                    j = json.loads(line)
                    ep_stat[j["episode_index"]] = j.get("stats", {})

        tnorm = e.get("task_normalize") or {}
        slots = e["camera_slots"]
        drop = set(e["drop_eps"])
        row = lic[rel]
        source = row["repo"] or rel
        ep_start = global_ep

        for ep_path in sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet")):
            local = int(os.path.basename(ep_path).split("_")[1].split(".")[0])
            if local in drop:
                continue
            if dry:
                pf = pq.ParquetFile(ep_path)
                n = pf.metadata.num_rows
                if n < 2:
                    continue
                cols = pf.schema_arrow.names
                t = None
            else:
                t = pq.read_table(ep_path)
                n = t.num_rows
                if n < 2:
                    continue
                cols = t.column_names
            chunk = global_ep // CHUNK

            if "task_index" in cols:
                col = t.column("task_index") if t is not None else \
                    pq.read_table(ep_path, columns=["task_index"]).column("task_index")
                ti = col[0].as_py()
            else:
                ti = 0
            task = tnorm.get(str(ti)) or orig_tasks.get(ti, "manipulation task")
            if task not in task2idx:
                task2idx[task] = len(task2idx)

            if not dry:
                # align parquet timestamps to the first present video's PTS
                v0 = 0.0
                for slot in SLOTS:
                    raw = slots.get(slot)
                    if raw:
                        vs = find_video(d, raw, local)
                        if vs:
                            v0 = first_pts(vs)
                            break
                ts = np.array(t.column("timestamp").to_pylist(), dtype=np.float64)
                ts = ts - ts[0] + v0

                new = {
                    "action": t.column("action"),
                    "observation.state": t.column("observation.state"),
                    "timestamp": pa.array(ts.astype(np.float32)),
                    "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                    "episode_index": pa.array(np.full(n, global_ep, dtype=np.int64)),
                    "index": pa.array(np.arange(global_idx, global_idx + n, dtype=np.int64)),
                    "task_index": pa.array(np.full(n, task2idx[task], dtype=np.int64)),
                }

                for slot in SLOTS:
                    raw = slots.get(slot)
                    present = raw is not None
                    vdir = f"{DST}/videos/chunk-{chunk:03d}/observation.images.{slot}"
                    os.makedirs(vdir, exist_ok=True)
                    vdst = f"{vdir}/episode_{global_ep:06d}.mp4"
                    if present:
                        vsrc = find_video(d, raw, local)
                        if vsrc:
                            link_or_copy(vsrc, vdst)
                        else:
                            present = False
                    if not present:
                        link_or_copy(make_black(fps, n, v0), vdst)
                    new[f"observation.images.{slot}_mask"] = pa.array(
                        np.full(n, 1.0 if present else 0.0, dtype=np.float32))

                ddir = f"{DST}/data/chunk-{chunk:03d}"
                os.makedirs(ddir, exist_ok=True)
                pq.write_table(pa.table(new), f"{ddir}/episode_{global_ep:06d}.parquet")

                acts.append(np.array(t.column("action").to_pylist(), dtype=np.float32))
                states.append(np.array(t.column("observation.state").to_pylist(), dtype=np.float32))

                st = ep_stat.get(local, {})
                ep_stats_lines.append({"episode_index": global_ep,
                                       "stats": {k: st[k] for k in ("action", "observation.state") if k in st}})

            episodes.append({"episode_index": global_ep, "tasks": [task], "length": n,
                             "source_dataset": source})
            global_ep += 1
            global_idx += n

        n_ep = global_ep - ep_start
        # license_join records author recordings as "self-owned", which is a provenance
        # marker rather than a license. They are released under Apache-2.0; keep the two
        # facts in separate fields so a consumer can filter on either.
        author_recorded = row["lic"] == "self-owned"
        sources[source] = {
            "repo_id": row["repo"],
            "license": "apache-2.0" if author_recorded else row["lic"],
            "provenance": "author-recorded" if author_recorded else "huggingface-hub",
            "episode_range": [ep_start, global_ep - 1] if n_ep else [],
            "n_episodes": n_ep,
        }
        print(f"[{i}/{len(keep)}] {rel}: +{n_ep} ep (total {global_ep})", flush=True)

    if not dry:
        def vfeat():
            for v in (feat_template or {}).values():
                if v.get("dtype") == "video":
                    return dict(v)
            return {"dtype": "video", "shape": [BLACK_H, BLACK_W, 3],
                    "names": ["height", "width", "channel"]}

        features = {
            "action": feat_template.get("action", {"dtype": "float32", "shape": [6]}),
            "observation.state": feat_template.get("observation.state", {"dtype": "float32", "shape": [6]}),
        }
        for slot in SLOTS:
            features[f"observation.images.{slot}"] = vfeat()
            features[f"observation.images.{slot}_mask"] = {"dtype": "float32", "shape": [1], "names": ["mask"]}
        features.update({
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        })

        info_out = {
            "codebase_version": "v2.1",
            "robot_type": "so101_follower",
            "total_episodes": global_ep,
            "total_frames": global_idx,
            "total_tasks": len(task2idx),
            "total_videos": global_ep * len(SLOTS),
            "total_chunks": (global_ep // CHUNK) + 1,
            "chunks_size": CHUNK,
            "fps": 30,
            "splits": {"train": f"0:{global_ep}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }
        json.dump(info_out, open(f"{DST}/meta/info.json", "w"), indent=2)
        with open(f"{DST}/meta/episodes.jsonl", "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
        with open(f"{DST}/meta/episodes_stats.jsonl", "w") as f:
            for s in ep_stats_lines:
                f.write(json.dumps(s) + "\n")
        with open(f"{DST}/meta/tasks.jsonl", "w") as f:
            for task, idx in sorted(task2idx.items(), key=lambda x: x[1]):
                f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")
        json.dump(sources, open(f"{DST}/meta/sources.json", "w"), ensure_ascii=False, indent=2)

        A = np.concatenate(acts)
        S = np.concatenate(states)
        del acts, states
        stats = {"action": stat_dim(A), "observation.state": stat_dim(S)}

        def dummy(v):
            return [[[float(v)]] for _ in range(3)]
        for k, f in features.items():
            if f.get("dtype") == "video":
                stats[k] = {"min": dummy(0.0), "max": dummy(1.0), "mean": dummy(0.5),
                            "std": dummy(0.25), "q01": dummy(0.02), "q99": dummy(0.98),
                            "count": [int(len(A))]}
        json.dump(stats, open(f"{DST}/meta/stats.json", "w"), indent=2)
        print(f"stats recomputed over {len(A)} frames")

    print(f"\ndatasets {len(keep)} / episodes {global_ep} / frames {global_idx} / "
          f"tasks {len(task2idx)} / videos {global_ep * len(SLOTS)}")
    if global_ep != plan["include_ep"]:
        print(f"WARNING: episode count differs from plan ({plan['include_ep']})")
    if not dry:
        print(f"output: {DST}")


if __name__ == "__main__":
    main()
