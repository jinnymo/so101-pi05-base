# SPDX-License-Identifier: Apache-2.0
"""Stage 7a: exhaustive parquet check of the assembled pool, before the merge.

After the merge an episode can no longer be traced back to its source, so every defect
has to be found here. Per episode of every dataset:
  - NaN or inf in action / observation.state (fatal for training)
  - frame_index continuous 0..n-1 (duplicates, gaps)
  - timestamp strictly increasing, and the measured rate consistent with the declared fps
  - degenerate length (empty or 1-row episodes)
  - extreme values (> 500 degrees)
  - action-chunk compatibility: episodes shorter than chunk_size + max_delay are reported

Only datasets with findings are printed.

Input:  <root>/base_train/{external/*/*,self/*}
Output: a console report.

Run:
  python 13_precheck.py --root /path/to/workspace
"""

import argparse
import glob
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

CHUNK_MIN = 58  # action chunk size 50 + max delay 8

_lk = threading.Lock()
_done = {"n": 0}


def check(d, total):
    import numpy as np
    import pyarrow.parquet as pq

    name = os.path.basename(d)
    iss = []
    try:
        info = json.load(open(f"{d}/meta/info.json"))
    except Exception:
        return name, ["info.json unreadable"], 0, 0
    fps = float(info.get("fps") or 30)
    eps = sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet")) or \
        sorted(glob.glob(f"{d}/data/chunk-*/*.parquet"))
    n_short = 0
    for ep in eps:
        epn = os.path.basename(ep)
        try:
            t = pq.read_table(ep, columns=["action", "observation.state", "frame_index", "timestamp"])
        except Exception as e:
            iss.append(f"{epn} read failed:{str(e)[:25]}")
            continue
        act = np.array(t.column("action").to_pylist(), dtype=float)
        st = np.array(t.column("observation.state").to_pylist(), dtype=float)
        fi = np.array(t.column("frame_index").to_pylist())
        ts = np.array(t.column("timestamp").to_pylist(), dtype=float)
        n = len(act)
        if n < 2:
            iss.append(f"{epn} length={n}")
            continue
        if n < CHUNK_MIN:
            n_short += 1
        if not np.isfinite(act).all():
            iss.append(f"{epn} action NaN/inf")
        if not np.isfinite(st).all():
            iss.append(f"{epn} state NaN/inf")
        if not np.array_equal(fi, np.arange(n)):
            iss.append(f"{epn} frame_index not continuous")
        if not (np.diff(ts) > 0).all():
            iss.append(f"{epn} timestamp not monotonic")
        else:
            dt = np.median(np.diff(ts))
            if dt > 0 and abs(1 / dt - fps) > fps * 0.25:
                iss.append(f"{epn} fps mismatch (measured {1/dt:.0f} != declared {fps:.0f})")
        if float(np.abs(act).max()) > 500 or float(np.abs(st).max()) > 500:
            iss.append(f"{epn} extreme value {float(np.abs(act).max()):.0f}")
    with _lk:
        _done["n"] += 1
        if _done["n"] % 30 == 0:
            print(f"  {_done['n']}/{total}", flush=True)
    return name, iss, len(eps), n_short


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
    print(f"pre-merge check of {len(dirs)} datasets, every episode...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(lambda d: check(d, len(dirs)), dirs))

    bad = [(n, i, ne) for n, i, ne, ns in res if i]
    short = [(n, ns, ne) for n, i, ne, ns in res if ns > 0]
    print(f"\n=== datasets with findings: {len(bad)}/{len(dirs)} ===")
    for n, i, ne in bad:
        print(f"  {n[:46]} ({ne} episodes): {'; '.join(i[:4])}")
    print(f"\n=== episodes shorter than {CHUNK_MIN} frames (action-chunk padding): "
          f"{len(short)} datasets ===")
    for n, ns, ne in sorted(short, key=lambda x: -x[1] / max(x[2], 1))[:12]:
        print(f"  {ns}/{ne} ({ns/ne*100:.0f}% short) {n[:42]}")
    if not bad:
        print("\n  no NaN/inf, frame_index, timestamp or extreme-value findings")


if __name__ == "__main__":
    main()
