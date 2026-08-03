# SPDX-License-Identifier: Apache-2.0
"""Stage 10: global normalization statistics for the merged repository.

Computes, per dimension over every `action` and `observation.state` frame in the merged
repository: min, max, mean, std (with a 1e-8 floor), q01, q99 and count.

`PI05Config.normalization_mapping` is {VISUAL: IDENTITY, STATE: MEAN_STD, ACTION: MEAN_STD},
and the released checkpoint's `config.json` records those same values, so mean and std are
the entries the training run actually consumed. q01/q99 are written as well because they are
the robust alternative for a pool with extreme calibration outliers: a quantile mapping clips
instead of letting a handful of frames stretch the scale. Using them requires setting the
mapping explicitly, and this build does not.

Image statistics are written as dummies because visual normalization is identity and the
policy uses ImageNet constants. The `_mask` columns are not normalized and are excluded by
the trainer.

Input:  <root>/base_unified/data/chunk-*/episode_*.parquet
Output: <root>/base_unified/meta/stats.json

Run:
  python 29_norm_stats_robust.py --root /path/to/workspace
"""

import argparse
import glob
import json
import os


def stat_dim(X):
    import numpy as np

    return {
        "min": X.min(0).tolist(),
        "max": X.max(0).tolist(),
        "mean": X.mean(0).tolist(),
        "std": (X.std(0) + 1e-8).tolist(),
        "q01": np.quantile(X, 0.01, axis=0).tolist(),
        "q99": np.quantile(X, 0.99, axis=0).tolist(),
        "count": [int(len(X))],
    }


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
    files = sorted(glob.glob(f"{dst}/data/chunk-*/episode_*.parquet"))
    print(f"global norm stats over {len(files)} episode parquet files...")
    acts, states = [], []
    for i, p in enumerate(files):
        t = pq.read_table(p, columns=["action", "observation.state"])
        acts.append(np.array(t.column("action").to_pylist(), dtype=np.float32))
        states.append(np.array(t.column("observation.state").to_pylist(), dtype=np.float32))
        if (i + 1) % 3000 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    A = np.concatenate(acts)
    S = np.concatenate(states)
    del acts, states

    stats = {"action": stat_dim(A), "observation.state": stat_dim(S)}

    info = json.load(open(f"{dst}/meta/info.json"))

    def dummy(v):
        return [[[float(v)]] for _ in range(3)]
    for k, f in info["features"].items():
        if f.get("dtype") == "video":
            stats[k] = {"min": dummy(0.0), "max": dummy(1.0), "mean": dummy(0.5),
                        "std": dummy(0.25), "q01": dummy(0.02), "q99": dummy(0.98),
                        "count": [int(len(A))]}

    json.dump(stats, open(f"{dst}/meta/stats.json", "w"), indent=2)
    print(f"\n=== stats.json written (frames={len(A)}) ===")
    print(f"  action q01: {[round(x,1) for x in stats['action']['q01']]}")
    print(f"  action q99: {[round(x,1) for x in stats['action']['q99']]}")
    print(f"  state  q01: {[round(x,1) for x in stats['observation.state']['q01']]}")
    print(f"  state  q99: {[round(x,1) for x in stats['observation.state']['q99']]}")


if __name__ == "__main__":
    main()
