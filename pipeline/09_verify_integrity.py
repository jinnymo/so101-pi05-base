# SPDX-License-Identifier: Apache-2.0
"""Stage 6d: integrity check of the kept pool, before anything is merged.

Per dataset:
  - parquet file count == info.total_episodes
  - summed parquet rows == info.total_frames
  - per-camera video file count == info.total_episodes
  - first-episode video frame count == first-episode parquet rows (frame drops, desync)

Only datasets with findings are printed.

Input:  <root>/external_hf/confirmed/{tier1,tier2,tier2b,tier3}/*
Output: a console report.

Run:
  python 09_verify_integrity.py --root /path/to/workspace
"""

import argparse
import glob
import os
import json
from concurrent.futures import ThreadPoolExecutor

TIERS = ["tier1", "tier2", "tier3", "tier2b"]


def verify(d):
    import pyarrow.parquet as pq

    name = os.path.basename(d)
    issues = []
    try:
        info = json.load(open(f"{d}/meta/info.json"))
    except Exception as e:
        return name, [f"info.json unreadable: {str(e)[:40]}"]
    te = int(info.get("total_episodes") or 0)
    tf = int(info.get("total_frames") or 0)
    feats = info.get("features", {})
    cams = [k for k in feats if k.startswith("observation.images") and feats[k].get("dtype") == "video"]

    pqs = sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet")) or \
        sorted(glob.glob(f"{d}/data/chunk-*/*.parquet"))
    if len(pqs) != te:
        issues.append(f"parquet {len(pqs)} != episodes {te}")
    try:
        rows = sum(pq.read_metadata(p).num_rows for p in pqs)
        if rows != tf:
            issues.append(f"rows {rows} != frames {tf}")
    except Exception as e:
        issues.append(f"parquet read failed: {str(e)[:30]}")

    for cam in cams:
        vids = glob.glob(f"{d}/videos/chunk-*/{cam}/episode_*.mp4") or \
            glob.glob(f"{d}/videos/{cam}/chunk-*/*.mp4")
        if len(vids) != te:
            issues.append(f"{cam.split('.')[-1]} videos {len(vids)} != episodes {te}")

    if pqs and cams:
        ep0_rows = pq.read_metadata(pqs[0]).num_rows
        cam0 = cams[0]
        v0 = (glob.glob(f"{d}/videos/chunk-*/{cam0}/episode_000000.mp4") or
              glob.glob(f"{d}/videos/{cam0}/chunk-*/*.mp4"))
        if v0:
            try:
                import av
                c = av.open(v0[0])
                nf = c.streams.video[0].frames or sum(1 for _ in c.decode(video=0))
                c.close()
                if abs(nf - ep0_rows) > 2:
                    issues.append(f"ep0 video frames {nf} != rows {ep0_rows}")
            except Exception as e:
                issues.append(f"video decode failed: {str(e)[:30]}")
    return name, issues


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get("ROOT", "."),
                    help="workspace root (default: $ROOT)")
    ap.add_argument("--src", default=None,
                    help="confirmed directory (default: <root>/external_hf/confirmed)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src = args.src or os.path.join(args.root, "external_hf", "confirmed")
    dirs = []
    for t in TIERS:
        dirs += sorted(glob.glob(f"{src}/{t}/*"))
    dirs = [d for d in dirs if os.path.isdir(d)]
    print(f"verifying {len(dirs)} datasets...", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(verify, dirs))

    bad = [(n, iss) for n, iss in results if iss]
    print(f"\n=== datasets with findings: {len(bad)}/{len(dirs)} ===")
    for n, iss in bad:
        print(f"  {n[:48]}: {'; '.join(iss)}")
    if not bad:
        print("  all clean")


if __name__ == "__main__":
    main()
