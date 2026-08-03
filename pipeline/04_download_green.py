# SPDX-License-Identifier: Apache-2.0
"""Stage 4: download every dataset that passed the action-conformity check.

Downloads the full snapshot (videos included) of each `flag == green` repository.
Each finished repository gets a `.download_complete` marker so the script is resumable.
Expect the download to be large; the full green set in the reference build was 303 GB.

The cache directory follows $HF_HOME. Export it to a volume with enough space
(`export HF_HOME=$ROOT/.hf_cache`) before running, otherwise the default cache on the
system disk fills up. Export $HF_TOKEN as well to avoid anonymous rate limits.

Input:  03_action_match.csv
Output: <root>/external_hf/<owner>__<name>/ per repository.

Run:
  python 04_download_green.py --root /path/to/workspace [--limit N]
"""

import argparse
import csv
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HOME", os.path.join(os.environ.get("ROOT", "."), ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")  # Xet fast transfer, recent hub versions

MARKER = ".download_complete"

_lock = threading.Lock()
_n = {"done": 0, "ok": 0, "fail": 0, "skip": 0}


def download_one(repo: str, dest_root: str, total: int, file_workers: int):
    from huggingface_hub import snapshot_download

    dest = f"{dest_root}/{repo.replace('/', '__')}"
    if os.path.exists(f"{dest}/{MARKER}"):
        with _lock:
            _n["skip"] += 1
            _n["done"] += 1
        return None
    for attempt in range(2):  # one retry, for brotli or transient network errors
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=dest, max_workers=file_workers)
            open(f"{dest}/{MARKER}", "w").close()
            with _lock:
                _n["ok"] += 1
                _n["done"] += 1
                used = shutil.disk_usage(dest_root)
                print(f"[{_n['done']}/{total}] OK {repo}  ({used.free // 2**30}G free)", flush=True)
            return None
        except Exception as e:
            if attempt == 0:
                continue
            with _lock:
                _n["fail"] += 1
                _n["done"] += 1
                print(f"[{_n['done']}/{total}] FAIL {repo}: {str(e)[:80]}", flush=True)
            return (repo, str(e)[:120])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get("ROOT", "."),
                    help="workspace root (default: $ROOT)")
    ap.add_argument("--dest", default=None,
                    help="download directory (default: <root>/external_hf)")
    ap.add_argument("--csv", default="03_action_match.csv", help="stage 3 result CSV")
    ap.add_argument("--limit", type=int, default=None, help="download only the first N repos")
    ap.add_argument("--repo-parallel", type=int, default=10, help="repositories in flight")
    ap.add_argument("--file-workers", type=int, default=4, help="parallel files per repository")
    args = ap.parse_args()

    dest_root = args.dest or os.path.join(args.root, "external_hf")
    os.makedirs(dest_root, exist_ok=True)
    green = [r["id"] for r in csv.DictReader(open(args.csv, encoding="utf-8"))
             if r["flag"] == "green"]
    if args.limit:
        green = green[:args.limit]
    print(f"{len(green)} green repositories -> {dest_root} "
          f"({args.repo_parallel} repos x {args.file_workers} files)", flush=True)

    with ThreadPoolExecutor(max_workers=args.repo_parallel) as ex:
        fails = [f for f in ex.map(
            lambda r: download_one(r, dest_root, len(green), args.file_workers), green) if f]

    print(f"\ndone: OK {_n['ok']} / skip {_n['skip']} / fail {_n['fail']}", flush=True)
    for r, e in fails:
        print(f"  FAIL {r}: {e}", flush=True)


if __name__ == "__main__":
    main()
