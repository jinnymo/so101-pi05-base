# SPDX-License-Identifier: Apache-2.0
"""Stage 8c: list episodes to drop, from empty episodes and video/action duration mismatch.

Per episode of every dataset:
  - fewer than 2 rows                                   -> empty episode, drop
  - a camera's video file is missing                    -> drop (the loader would try to
                                                           decode it and crash)
  - |video duration - action duration| / action duration > THRESH -> drop

This compares seconds, not frame counts. A 10 fps video against a 30 fps action stream has
a legitimately different frame count; the loader decodes by timestamp. Only the duration
comparison is meaningful. Video duration comes from container metadata, which is fast.

Short episodes (2 to 57 rows) are not dropped: the trainer pads them with action_is_pad,
and keeping them preserves the widest distribution.

Input:  <root>/base_train/{external/*/*,self/*}
Output: artifacts/25_ep_drop.json, {dataset: {drop_eps: [...], reasons: {ep: text}}}.

Run:
  python 24_ep_drop.py --root /path/to/workspace
"""

import argparse
import glob
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
THRESH = 0.05  # more than 5% duration disagreement is a block
_lk = threading.Lock()
_done = {"n": 0}


def video_dur(path):
    import av
    c = av.open(path)
    try:
        s = c.streams.video[0]
        if s.duration and s.time_base:
            return float(s.duration * s.time_base)
        if c.duration:
            return float(c.duration) / 1_000_000.0  # AV_TIME_BASE
        n = sum(1 for _ in c.decode(video=0))
        fr = s.average_rate or s.guessed_rate
        return n / float(fr) if fr else None
    finally:
        c.close()


def check(d, base, total):
    import pyarrow.parquet as pq

    name = os.path.relpath(d, base)
    drop_eps, reasons = set(), {}
    try:
        info = json.load(open(f"{d}/meta/info.json"))
    except Exception:
        return name, [], {"_err": "info.json unreadable"}
    feats = info.get("features", {})
    cams = [k for k in feats if k.startswith("observation.images")
            and feats[k].get("dtype") == "video"]
    eps = sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet"))
    for ep in eps:
        epn = os.path.basename(ep).replace("episode_", "").replace(".parquet", "")
        ei = int(epn)
        try:
            t = pq.read_table(ep, columns=["timestamp"])
            ts = t.column("timestamp").to_pylist()
        except Exception as e:
            drop_eps.add(ei); reasons[ei] = f"parquet read failed:{str(e)[:20]}"; continue
        rows = len(ts)
        if rows < 2:
            drop_eps.add(ei); reasons[ei] = f"empty ep rows={rows}"; continue
        act_dur = float(ts[-1]) - float(ts[0])
        if act_dur <= 0:
            drop_eps.add(ei); reasons[ei] = "action duration<=0"; continue
        for cam in cams:
            vp = (glob.glob(f"{d}/videos/chunk-*/{cam}/episode_{epn}.mp4") or
                  glob.glob(f"{d}/videos/{cam}/chunk-*/*{epn}.mp4"))
            if not vp:
                drop_eps.add(ei); reasons[ei] = f"{cam.split('.')[-1]} video missing"; break
            try:
                vd = video_dur(vp[0])
            except Exception as e:
                drop_eps.add(ei)
                reasons[ei] = f"{cam.split('.')[-1]} decode failed:{str(e)[:20]}"
                break
            if vd is None:
                continue
            if abs(vd - act_dur) / act_dur > THRESH:
                drop_eps.add(ei)
                # The exact wording matters: 26_build_manifest.py matches
                # "duration mismatch" and parses the two durations back out of this string.
                reasons[ei] = (f"{cam.split('.')[-1]} duration mismatch "
                               f"video{vd:.1f}s!=action{act_dur:.1f}s "
                               f"({abs(vd-act_dur)/act_dur*100:.0f}%)")
                break
    with _lk:
        _done["n"] += 1
        if _done["n"] % 30 == 0:
            print(f"  {_done['n']}/{total}", flush=True)
    return name, sorted(drop_eps), reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get("ROOT", "."),
                    help="workspace root (default: $ROOT)")
    ap.add_argument("--src", default=None, help="pool directory (default: <root>/base_train)")
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts", "25_ep_drop.json"),
                    help="output JSON")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    base = args.src or os.path.join(args.root, "base_train")
    dirs = sorted(d for d in glob.glob(f"{base}/external/*/*") + glob.glob(f"{base}/self/*")
                  if os.path.isdir(d))
    print(f"episode drop check of {len(dirs)} datasets (duration based)...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(lambda d: check(d, base, len(dirs)), dirs))

    out = {}
    total_drop = 0
    for name, eps, reasons in res:
        if eps or reasons.get("_err"):
            out[name] = {"drop_eps": eps, "reasons": {str(k): v for k, v in reasons.items()}}
            total_drop += len(eps)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n=== {total_drop} episodes across {len(out)} datasets -> {args.out} ===")
    for name, info in sorted(out.items()):
        eps = info["drop_eps"]
        print(f"  {name}: {len(eps)} - {eps[:8]}{'...' if len(eps) > 8 else ''}")


if __name__ == "__main__":
    main()
