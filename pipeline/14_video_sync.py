# SPDX-License-Identifier: Apache-2.0
"""Stage 7b: video-to-parquet synchronisation check, before the merge.

Per episode and per camera, compares the video frame count against the parquet row count.
A mismatch means dropped frames or a video that does not line up with the action stream,
which trains the model on wrong frame-action pairs. Frame counts come from container
metadata; if the container does not report them, the stream is decoded and counted.

A frame-count difference is only meaningful when the video and the data share the same
rate. `24_ep_drop.py` repeats the comparison in seconds, and that is the check that
decides what actually gets dropped.

Input:  <root>/base_train/{external/*/*,self/*}
Output: a console report.

Run:
  python 14_video_sync.py --root /path/to/workspace
"""

import argparse
import glob
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

_lk = threading.Lock()
_done = {"n": 0}


def frame_count(path):
    import av
    c = av.open(path)
    try:
        nf = c.streams.video[0].frames
        if nf and nf > 0:
            return nf
        return sum(1 for _ in c.decode(video=0))  # fallback, slow
    finally:
        c.close()


def check(d, total):
    import pyarrow.parquet as pq

    name = os.path.basename(d)
    iss = []
    try:
        info = json.load(open(f"{d}/meta/info.json"))
    except Exception:
        return name, ["info.json unreadable"]
    feats = info.get("features", {})
    cams = [k for k in feats if k.startswith("observation.images") and feats[k].get("dtype") == "video"]
    eps = sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet"))
    for ep in eps:
        epn = os.path.basename(ep).replace("episode_", "").replace(".parquet", "")
        try:
            rows = pq.read_metadata(ep).num_rows
        except Exception:
            continue
        for cam in cams:
            vp = (glob.glob(f"{d}/videos/chunk-*/{cam}/episode_{epn}.mp4") or
                  glob.glob(f"{d}/videos/{cam}/chunk-*/*{epn}.mp4"))
            if not vp:
                iss.append(f"ep{epn} {cam.split('.')[-1]} video missing")
                continue
            try:
                nf = frame_count(vp[0])
                if abs(nf - rows) > 2:
                    iss.append(f"ep{epn} {cam.split('.')[-1]} frames{nf}!=rows{rows}")
            except Exception as e:
                iss.append(f"ep{epn} {cam.split('.')[-1]} decode failed:{str(e)[:20]}")
    with _lk:
        _done["n"] += 1
        if _done["n"] % 20 == 0:
            print(f"  {_done['n']}/{total}", flush=True)
    return name, iss


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get("ROOT", "."),
                    help="workspace root (default: $ROOT)")
    ap.add_argument("--src", default=None, help="pool directory (default: <root>/base_train)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    base = args.src or os.path.join(args.root, "base_train")
    dirs = [d for d in glob.glob(f"{base}/external/*/*") + glob.glob(f"{base}/self/*")
            if os.path.isdir(d)]
    print(f"video/parquet sync check of {len(dirs)} datasets...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(lambda d: check(d, len(dirs)), dirs))

    bad = [(n, i) for n, i in res if i]
    print(f"\n=== sync findings: {len(bad)}/{len(dirs)} ===")
    for n, i in bad:
        print(f"  {n[:46]}: {len(i)} findings - {'; '.join(i[:3])}")
    if not bad:
        print("  every video frame count matches its parquet row count")


if __name__ == "__main__":
    main()
