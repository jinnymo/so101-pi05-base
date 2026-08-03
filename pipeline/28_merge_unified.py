# SPDX-License-Identifier: Apache-2.0
"""Stage 9: merge the kept datasets into one LeRobot v2.1 repository, following the manifest.

For every kept dataset, every episode that is not dropped:
  - three video slots are built. A present slot is hard-linked from the source mp4 (no copy,
    which saves both disk and time); a masked slot is hard-linked from a cached black clip
    generated per (fps, frame count, start pts).
  - the parquet keeps action, observation.state and the original timestamps, and gets
    frame_index / episode_index / index / task_index renumbered globally, plus one
    float32 observation.images.{slot}_mask column per slot (0.0 or 1.0).
  - timestamps are shifted so the first sample lines up with the first video PTS. Datasets
    recorded at 10 fps carry a non-zero first PTS, and without this shift the first frames
    decode from the wrong place.
  - the task string is replaced where the manifest says so, and task indices are global.

Input:  <root>/base_train/ plus artifacts/27_manifest.json
Output: <root>/base_unified/, LeRobot v2.1, declared fps 30, original timestamps preserved.

Run:
  python 28_merge_unified.py --root /path/to/workspace [--limit N]
"""

import argparse
import glob
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SLOTS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
CHUNK = 1000
BLACK_W, BLACK_H = 128, 128

_black_cache = {}


def find_video(d, cam, epn):
    for pat in (f"{d}/videos/chunk-*/{cam}/episode_{epn:06d}.mp4",
                f"{d}/videos/{cam}/chunk-*/*{epn:06d}.mp4"):
        g = glob.glob(pat)
        if g:
            return g[0]
    return None


def link_or_copy(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def first_pts(path):
    """First video frame PTS in seconds, used to align the parquet timestamps."""
    import av
    c = av.open(path)
    try:
        s = c.streams.video[0]
        for f in c.decode(video=0):
            return float(f.pts * s.time_base) if f.pts is not None else 0.0
        return 0.0
    finally:
        c.close()


def make_black(dst_root, ffmpeg, fps, n, ts0=0.0):
    key = (round(fps), n, round(ts0, 3))
    if key in _black_cache:
        return _black_cache[key]
    bdir = f"{dst_root}/.black"
    os.makedirs(bdir, exist_ok=True)
    path = f"{bdir}/black_{round(fps)}_{n}_{round(ts0 * 1000)}.mp4"
    if not os.path.exists(path):
        cmd = [ffmpeg, "-y", "-f", "lavfi",
               "-i", f"color=c=black:s={BLACK_W}x{BLACK_H}:r={fps}",
               "-frames:v", str(n)]
        if ts0:
            cmd += ["-output_ts_offset", f"{ts0:.4f}"]
        cmd += ["-pix_fmt", "yuv420p", path]
        subprocess.run(cmd, check=True, capture_output=True)
    _black_cache[key] = path
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get("ROOT", "."),
                    help="workspace root (default: $ROOT)")
    ap.add_argument("--src", default=None, help="pool directory (default: <root>/base_train)")
    ap.add_argument("--dst", default=None,
                    help="merge output directory (default: <root>/base_unified)")
    ap.add_argument("--manifest", default=os.path.join(HERE, "artifacts", "27_manifest.json"))
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"),
                    help="ffmpeg binary used to render the black dummy clips")
    ap.add_argument("--limit", type=int, default=None,
                    help="merge only the first N kept datasets (trial run)")
    args = ap.parse_args()

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = args.src or os.path.join(args.root, "base_train")
    dst = args.dst or os.path.join(args.root, "base_unified")

    mani = json.load(open(args.manifest, encoding="utf-8"))["datasets"]
    keep = {r: e for r, e in mani.items() if e["keep"]}
    if args.limit:
        keep = dict(sorted(keep.items())[:args.limit])

    os.makedirs(f"{dst}/data", exist_ok=True)
    os.makedirs(f"{dst}/videos", exist_ok=True)
    os.makedirs(f"{dst}/meta", exist_ok=True)

    global_ep, global_idx = 0, 0
    task2idx = {}
    episodes = []
    ep_stats_lines = []
    feat_template = None  # action / state / video feature block of the first dataset

    for rel, e in sorted(keep.items()):
        d = f"{src}/{rel}"
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
        ep_meta = {}
        for line in open(f"{d}/meta/episodes.jsonl"):
            if line.strip():
                j = json.loads(line)
                ep_meta[j["episode_index"]] = j.get("tasks", [])
        # per-episode source stats (action / state only; global norm stats come from stage 10)
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

        for ep_path in sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet")):
            local = int(os.path.basename(ep_path).split("_")[1].split(".")[0])
            if local in drop:
                continue
            t = pq.read_table(ep_path)
            n = t.num_rows
            if n < 2:
                continue
            chunk = global_ep // CHUNK

            # Align the parquet timestamps to the first present video's PTS.
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
            }
            # task rewrite plus a global task index
            ti = t.column("task_index")[0].as_py() if "task_index" in t.column_names else 0
            task = tnorm.get(str(ti)) or orig_tasks.get(ti, "manipulation task")
            if task not in task2idx:
                task2idx[task] = len(task2idx)
            new["task_index"] = pa.array(np.full(n, task2idx[task], dtype=np.int64))

            # three video slots plus their mask columns
            for slot in SLOTS:
                raw = slots.get(slot)
                present = raw is not None
                vdir = f"{dst}/videos/chunk-{chunk:03d}/observation.images.{slot}"
                os.makedirs(vdir, exist_ok=True)
                vdst = f"{vdir}/episode_{global_ep:06d}.mp4"
                if present:
                    vsrc = find_video(d, raw, local)
                    if vsrc:
                        link_or_copy(vsrc, vdst)
                    else:
                        present = False
                if not present:
                    link_or_copy(make_black(dst, args.ffmpeg, fps, n, v0), vdst)
                new[f"observation.images.{slot}_mask"] = pa.array(
                    np.full(n, 1.0 if present else 0.0, dtype=np.float32))

            ddir = f"{dst}/data/chunk-{chunk:03d}"
            os.makedirs(ddir, exist_ok=True)
            pq.write_table(pa.table(new), f"{ddir}/episode_{global_ep:06d}.parquet")

            episodes.append({"episode_index": global_ep, "tasks": [task], "length": n})
            st = ep_stat.get(local, {})
            ep_stats_lines.append(
                {"episode_index": global_ep,
                 "stats": {k: st[k] for k in ("action", "observation.state") if k in st}})
            global_ep += 1
            global_idx += n
        print(f"  {rel}: cumulative episodes={global_ep}", flush=True)

    def vfeat():
        # reuse the first video feature block found, as a representative shape
        for k, v in (feat_template or {}).items():
            if v.get("dtype") == "video":
                return dict(v)
        return {"dtype": "video", "shape": [BLACK_H, BLACK_W, 3],
                "names": ["height", "width", "channel"]}

    features = {
        "action": feat_template.get("action", {"dtype": "float32", "shape": [6]}),
        "observation.state": feat_template.get("observation.state",
                                               {"dtype": "float32", "shape": [6]}),
    }
    for slot in SLOTS:
        features[f"observation.images.{slot}"] = vfeat()
        features[f"observation.images.{slot}_mask"] = {"dtype": "float32", "shape": [1],
                                                       "names": ["mask"]}
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
    json.dump(info_out, open(f"{dst}/meta/info.json", "w"), indent=2)
    with open(f"{dst}/meta/episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    with open(f"{dst}/meta/episodes_stats.jsonl", "w") as f:
        for s in ep_stats_lines:
            f.write(json.dumps(s) + "\n")
    with open(f"{dst}/meta/tasks.jsonl", "w") as f:
        for task, idx in sorted(task2idx.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    print(f"\n=== merge done: episodes={global_ep} frames={global_idx} "
          f"tasks={len(task2idx)} -> {dst} ===")


if __name__ == "__main__":
    main()
