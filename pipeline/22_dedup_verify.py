# SPDX-License-Identifier: Apache-2.0
"""Stage 8b: detect duplicate datasets by action fingerprint, before the merge.

Each dataset is reduced to a multiset of per-episode (length, rounded action mean) pairs,
and duplicates are decided by order-independent subset / identity relations on that
multiset. Episode lengths alone produce coincidental matches, so the action mean is part
of the key. Re-uploads, cumulative uploads and split copies share byte-identical action
values, so their fingerprints match or nest exactly.

  identical: the fingerprints are equal, the first name alphabetically is kept
  subset:    fingerprint(x) is a proper subset of fingerprint(y), x is dropped and the
             largest superset is kept

Owner names are ignored, so cross-account re-uploads are caught as well.

Input:  <root>/base_train/{external/*/*,self/*}/meta/{episodes.jsonl,episodes_stats.jsonl}
Output: artifacts/23_dedup_result.json (keep / drop with the reason) plus a console summary.

Run:
  python 22_dedup_verify.py --root /path/to/workspace
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RND = 1  # rounding of the action mean; re-uploads are byte-identical so 1 decimal suffices


def load_fingerprint(d):
    lengths = {}
    try:
        for line in open(f"{d}/meta/episodes.jsonl"):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            lengths[e["episode_index"]] = e.get("length", 0)
    except Exception:
        return None
    fp = Counter()
    try:
        for line in open(f"{d}/meta/episodes_stats.jsonl"):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            ei = e["episode_index"]
            am = e.get("stats", {}).get("action", {}).get("mean", [])
            # flatten a nested [[...]] action mean
            flat = []
            for v in am:
                if isinstance(v, (list, tuple)):
                    flat.extend(v)
                else:
                    flat.append(v)
            key = (lengths.get(ei, -1), tuple(round(float(x), RND) for x in flat))
            fp[key] += 1
    except Exception:
        return None
    return fp


def is_subset(a, b):
    """Multiset a is contained in b when a - b is empty."""
    return not (a - b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get("ROOT", "."),
                    help="workspace root (default: $ROOT)")
    ap.add_argument("--src", default=None, help="pool directory (default: <root>/base_train)")
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts", "23_dedup_result.json"),
                    help="output JSON")
    args = ap.parse_args()

    base = args.src or os.path.join(args.root, "base_train")
    dirs = sorted(
        d for d in glob.glob(f"{base}/external/*/*") + glob.glob(f"{base}/self/*")
        if os.path.isdir(d)
    )
    fps, epn = {}, {}
    for d in dirs:
        rel = os.path.relpath(d, base)
        fp = load_fingerprint(d)
        if fp is None or sum(fp.values()) == 0:
            print(f"  fingerprint failed or empty: {rel}")
            continue
        fps[rel] = fp
        epn[rel] = sum(fp.values())

    rels = list(fps.keys())
    drop = {}

    # 1. identical groups: one representative per exact fingerprint
    groups = defaultdict(list)
    for rel in rels:
        h = tuple(sorted(fps[rel].items()))
        groups[h].append(rel)
    reps = []
    for h, members in groups.items():
        members.sort()
        reps.append(members[0])
        for m in members[1:]:
            drop[m] = {"kept": members[0], "relation": "identical",
                       "ep": epn[m], "kept_ep": epn[members[0]]}

    # 2. proper subsets: drop when a strictly larger superset exists, keep the largest
    for x in reps:
        best = None
        for y in reps:
            if y == x:
                continue
            if is_subset(fps[x], fps[y]) and not is_subset(fps[y], fps[x]):
                if best is None or epn[y] > epn[best]:
                    best = y
        if best:
            drop[x] = {"kept": best, "relation": "subset",
                       "ep": epn[x], "kept_ep": epn[best]}

    keep = [r for r in rels if r not in drop]

    # clusters: kept dataset -> everything dropped in favour of it
    clusters = defaultdict(list)
    for x, info in drop.items():
        # follow the chain to the final kept dataset (identical -> subset)
        k = info["kept"]
        seen = set()
        while k in drop and k not in seen:
            seen.add(k)
            k = drop[k]["kept"]
        clusters[k].append((x, info["relation"], info["ep"]))

    result = {
        "keep": sorted(keep),
        "drop": {k: drop[k] for k in sorted(drop)},
        "n_total": len(rels),
        "n_keep": len(keep),
        "n_drop": len(drop),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== dedup: {len(rels)} datasets -> keep {len(keep)} / drop {len(drop)} ===")
    print(f"  -> {args.out}\n")
    print("=== duplicate clusters (kept dataset contains the dropped ones) ===")
    for k in sorted(clusters, key=lambda r: -epn[r]):
        members = clusters[k]
        if not members:
            continue
        print(f"\n[KEEP] {k}  (ep={epn[k]})")
        for x, rel, ep in sorted(members, key=lambda m: -m[2]):
            print(f"   drop {rel:10s} ep={ep:4d}  {x}")


if __name__ == "__main__":
    main()
