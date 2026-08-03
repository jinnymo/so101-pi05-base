# SPDX-License-Identifier: Apache-2.0
"""Check that a downloaded so101-pi05-base checkpoint is complete and loadable.

Reads the safetensors header and config.json directly, so it needs no torch, no
lerobot and no vlash - only the Python standard library. It does not run the
model; it establishes that the files are intact and that the policy config is
the one this release documents.

Usage:
    python verify_checkpoint.py /path/to/checkpoint
"""

import argparse
import json
import math
import os
import struct

REQUIRED_FILES = ["config.json", "model.safetensors"]
EXPECTED_SLOTS = [
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
]


def read_safetensors_header(path):
    """Return (header dict, header byte length). The first 8 bytes are its size."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), n


def line(text):
    print(f"      {text}")


def status(n, label, ok):
    print(f"[{n}/4] {label:<38} -- {'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", help="directory holding config.json and model.safetensors")
    args = ap.parse_args()

    ckpt = args.checkpoint
    print(f"checkpoint: {ckpt}\n")
    failed = 0

    missing = [n for n in REQUIRED_FILES if not os.path.isfile(os.path.join(ckpt, n))]
    ok = not missing
    status(1, "required files", ok)
    if ok:
        for n in REQUIRED_FILES:
            line(f"{n:<20} {os.path.getsize(os.path.join(ckpt, n)):,} bytes")
    else:
        line(f"missing: {', '.join(missing)}")
        print("\n0/4 checks passed")
        return 1

    weights = os.path.join(ckpt, "model.safetensors")

    err = None
    header_len, tensors, params, dtypes = 0, {}, 0, []
    try:
        header, header_len = read_safetensors_header(weights)
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        params = sum(math.prod(v["shape"]) for v in tensors.values())
        dtypes = sorted({v["dtype"] for v in tensors.values()})
    except Exception as exc:
        err = exc
    ok = err is None and params > 0
    status(2, "safetensors header", ok)
    if ok:
        line(f"{'tensors':<20} {len(tensors):,}")
        line(f"{'parameters':<20} {params:,}")
        line(f"{'dtypes':<20} {', '.join(dtypes)}")
    else:
        failed += 1
        line(f"header unreadable: {err}")

    with open(os.path.join(ckpt, "config.json")) as f:
        cfg = json.load(f)
    slots = [k for k in cfg.get("input_features", {}) if k.startswith("observation.images.")]
    state = cfg.get("input_features", {}).get("observation.state", {}).get("shape", [None])[0]
    action = cfg.get("output_features", {}).get("action", {}).get("shape", [None])[0]
    ok = cfg.get("type") == "pi05" and slots == EXPECTED_SLOTS and state == 6 and action == 6
    status(3, "policy config", ok)
    line(f"{'type':<20} {cfg.get('type')}")
    for s in slots:
        line(f"{'camera slot':<20} {s.rsplit('.', 1)[-1]}")
    line(f"{'state / action dim':<20} {state} / {action}")
    line(f"{'chunk size':<20} {cfg.get('chunk_size')}")
    line(f"{'image resolution':<20} {cfg.get('image_resolution')}")
    line(f"{'normalization':<20} {cfg.get('normalization_mapping')}")
    line(f"{'empty_cameras':<20} {cfg.get('empty_cameras')}  (override per camera count at run time)")
    if not ok:
        failed += 1

    end = max((v["data_offsets"][1] for v in tensors.values()), default=0)
    expected = 8 + header_len + end
    actual = os.path.getsize(weights)
    ok = expected == actual
    status(4, "header offsets vs file size", ok)
    line(f"{'declared':<20} {expected:,} bytes")
    line(f"{'on disk':<20} {actual:,} bytes")
    if not ok:
        failed += 1
        line("truncated or partially downloaded")

    print(f"\n{4 - failed}/4 checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
