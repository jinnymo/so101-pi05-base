# SPDX-License-Identifier: Apache-2.0
"""Stage 5: convert LeRobot v3.0 datasets to v2.1 in place.

Drives `v3_to_v2/convert.py` from https://github.com/jinnymo/lerobot-v3-v2-converter,
one subprocess per dataset. That converter vendors NVIDIA's Isaac-GR00T
`scripts/lerobot_conversion/convert_v3_to_v2.py` verbatim (commit
`23ace64f17aa5015259b8609d371eb61a357c776`) and adds a LeRobot 0.5.x import shim, so the
conversion itself is the same code this build ran. It is a stream copy, so the result is
byte-faithful for the action and state columns. v2.1 is what the training stack consumes.

Each original v3.0 tree is left behind as `<name>_v3.0`; the converted tree takes the
original path.

Input:  03_action_match.csv (rows with flag green and codebase_version v3.0)
        and <root>/external_hf/<owner>__<name>/ downloaded by stage 4.
Output: the same directories, rewritten as v2.1.

Run (inside an environment that has LeRobot installed):
  git clone https://github.com/jinnymo/lerobot-v3-v2-converter
  python 05_convert_v3_to_v2.py --root /path/to/workspace \
      --converter /path/to/lerobot-v3-v2-converter
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_lock = threading.Lock()
_n = {"ok": 0, "skip": 0, "fail": 0, "done": 0}


def convert_one(repo_id: str, root: str, total: int, cli: Path):
    p = Path(root) / repo_id.replace("/", "__")
    info = p / "meta" / "info.json"
    if not info.exists():
        with _lock:
            _n["skip"] += 1
            _n["done"] += 1
        return None
    try:
        if json.loads(info.read_text(encoding="utf-8")).get("codebase_version") == "v2.1":
            with _lock:
                _n["skip"] += 1
                _n["done"] += 1
            return None
    except Exception:
        pass
    r = subprocess.run([sys.executable, str(cli), "--input", str(p)],
                       capture_output=True, text=True)
    if r.returncode == 0:
        with _lock:
            _n["ok"] += 1
            _n["done"] += 1
            print(f"[{_n['done']}/{total}] OK {repo_id}", flush=True)
        return None
    tail = (r.stderr or r.stdout).strip().splitlines()
    err = tail[-1] if tail else f"exit {r.returncode}"
    with _lock:
        _n["fail"] += 1
        _n["done"] += 1
        print(f"[{_n['done']}/{total}] FAIL {repo_id}: {err[:90]}", flush=True)
    return (repo_id, err[:140])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get("ROOT", "."),
                    help="workspace root (default: $ROOT)")
    ap.add_argument("--src", default=None, help="dataset directory (default: <root>/external_hf)")
    ap.add_argument("--csv", default="03_action_match.csv", help="stage 3 result CSV")
    ap.add_argument("--converter", default=os.environ.get("CONVERTER", ""),
                    help="lerobot-v3-v2-converter checkout (default: $CONVERTER)")
    ap.add_argument("--parallel", type=int, default=4,
                    help="datasets in flight (each spawns its own ffmpeg per episode)")
    args = ap.parse_args()

    cli = Path(args.converter).expanduser() / "v3_to_v2" / "convert.py"
    if not cli.is_file():
        sys.exit(f"converter CLI not found: {cli}\n"
                 "  git clone https://github.com/jinnymo/lerobot-v3-v2-converter")

    src = args.src or os.path.join(args.root, "external_hf")
    green_v3 = [r["id"] for r in csv.DictReader(open(args.csv, encoding="utf-8"))
                if r["flag"] == "green" and r["codebase_version"] == "v3.0"]
    print(f"{len(green_v3)} green v3.0 datasets -> v2.1 (parallel {args.parallel})", flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        fails = [f for f in ex.map(
            lambda r: convert_one(r, src, len(green_v3), cli), green_v3) if f]

    print(f"\ndone: OK {_n['ok']} / skip {_n['skip']} / fail {_n['fail']}", flush=True)
    for r, e in fails:
        print(f"  FAIL {r}: {e}", flush=True)


if __name__ == "__main__":
    main()
