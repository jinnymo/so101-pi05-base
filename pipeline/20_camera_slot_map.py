# SPDX-License-Identifier: Apache-2.0
"""Stage 8a: map each dataset's cameras onto the three fixed pi0.5 image slots.

pi0.5 takes exactly three image slots: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb.
"No camera" is a black dummy video plus image_mask=False, which blocks that slot's
attention keys and isolates it from the rest of the input.

Camera count is data, not a weight: a one-camera and a five-camera dataset train equally,
only the empty slots get masked.

Rules, in order:
  1. depth and IR streams are dropped (the slots are RGB)
  2. external cameras are sorted by priority; the first one takes base_0_rgb
  3. wrist cameras take left_wrist_0_rgb then right_wrist_0_rgb
  4. slots still empty are filled from the remaining cameras, external first
  5. cameras beyond three are dropped
  6. slots still empty are masked at training time

Input:  <root>/base_train/{external/*/*,self/*}/meta/info.json
Output: artifacts/21_camera_slot_map.json, one entry per dataset, plus a console summary.

Run:
  python 20_camera_slot_map.py --root /path/to/workspace
"""

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SLOTS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

# Wrist / on-arm keywords, from the ~50 camera-key suffixes actually observed.
WRIST_KW = ("wrist", "gripper", "endeff", "handeye", "ego", "tip", "hand", "arm", "robo")
# depth / IR: not usable in an RGB slot
DEPTH_IR_KW = ("depth", "_ir", "infrared")
# cannot be classified automatically; flagged for a manual look
AMBIG_KW = ("obs_image", "camera1", "camera2", "camera_1", "camera_2")
# external camera priority, earlier means closer to the base slot
EXT_PRIORITY = ("front", "top", "overhead", "side", "up", "head", "above",
                "context", "main", "fixed", "base", "horizon", "area")


def cam_suffix(key):
    """observation.images.X / observation.images.images.X -> lowercase trailing token."""
    s = key.split("observation.images.", 1)[-1]
    return s.lower().lstrip(".")


def classify(suf):
    if any(d in suf for d in DEPTH_IR_KW):
        return "exclude"
    if any(a in suf for a in AMBIG_KW):
        return "ambig"
    if any(w in suf for w in WRIST_KW):
        return "wrist"
    return "external"


def ext_rank(suf):
    for i, p in enumerate(EXT_PRIORITY):
        if p in suf:
            return i
    return len(EXT_PRIORITY)


def map_dataset(cam_keys):
    wrist, ext, excl, ambig = [], [], [], []
    for c in cam_keys:
        suf = cam_suffix(c)
        t = classify(suf)
        if t == "wrist":
            wrist.append(c)
        elif t == "external":
            ext.append((ext_rank(suf), c))
        elif t == "exclude":
            excl.append(c)
        else:
            ambig.append((ext_rank(suf), c))
    ext = [c for _, c in sorted(ext, key=lambda x: x[0])]
    # ambiguous cameras rank after external ones: mapped automatically, but flagged
    ambig_sorted = [c for _, c in sorted(ambig, key=lambda x: x[0])]

    slots = {s: None for s in SLOTS}
    if ext:
        slots["base_0_rgb"] = ext.pop(0)
    if wrist:
        slots["left_wrist_0_rgb"] = wrist.pop(0)
    if wrist:
        slots["right_wrist_0_rgb"] = wrist.pop(0)

    pool = ext + ambig_sorted + wrist
    for s in SLOTS:
        if slots[s] is None and pool:
            slots[s] = pool.pop(0)

    dropped = excl + pool  # depth/IR plus anything beyond three slots
    masked = [s for s in SLOTS if slots[s] is None]
    return {
        "slots": slots,
        "masked_slots": masked,
        "dropped_cams": dropped,
        "ambig_cams": [c for _, c in ambig],
        "n_present": len(SLOTS) - len(masked),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get("ROOT", "."),
                    help="workspace root (default: $ROOT)")
    ap.add_argument("--src", default=None, help="pool directory (default: <root>/base_train)")
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts", "21_camera_slot_map.json"),
                    help="output JSON")
    args = ap.parse_args()

    base = args.src or os.path.join(args.root, "base_train")
    dirs = sorted(
        d for d in glob.glob(f"{base}/external/*/*") + glob.glob(f"{base}/self/*")
        if os.path.isdir(d)
    )
    result = {}
    summ = {"present_1": 0, "present_2": 0, "present_3": 0,
            "has_dropped": 0, "has_ambig": 0, "base_masked": 0}
    for d in dirs:
        rel = os.path.relpath(d, base)
        try:
            info = json.load(open(f"{d}/meta/info.json"))
        except Exception as e:
            result[rel] = {"error": str(e)[:60]}
            continue
        cams = [k for k, v in info.get("features", {}).items()
                if isinstance(v, dict) and v.get("dtype") == "video"]
        m = map_dataset(cams)
        m["raw_cams"] = cams
        result[rel] = m
        summ[f"present_{m['n_present']}"] = summ.get(f"present_{m['n_present']}", 0) + 1
        if m["dropped_cams"]:
            summ["has_dropped"] += 1
        if m["ambig_cams"]:
            summ["has_ambig"] += 1
        if m["slots"]["base_0_rgb"] is None:
            summ["base_masked"] += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"=== camera slot map: {len(dirs)} datasets -> {args.out} ===\n")
    print("filled slots per dataset:")
    for n in (1, 2, 3):
        print(f"  {n} slot(s): {summ.get(f'present_{n}', 0)}")
    print(f"\ndatasets with dropped cameras: {summ['has_dropped']}")
    print(f"datasets needing a manual look (ambiguous keys): {summ['has_ambig']}")
    print(f"datasets with the base slot masked (no external camera): {summ['base_masked']}")

    print("\n=== ambiguous / dropped detail ===")
    for rel, m in result.items():
        if m.get("ambig_cams") or m.get("dropped_cams"):
            tag = []
            if m.get("ambig_cams"):
                tag.append(f"ambig={m['ambig_cams']}")
            if m.get("dropped_cams"):
                tag.append(f"dropped={m['dropped_cams']}")
            print(f"  {rel}: {'; '.join(tag)}")


if __name__ == "__main__":
    main()
