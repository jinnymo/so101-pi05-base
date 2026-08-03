# SPDX-License-Identifier: Apache-2.0
"""Stage 8d: combine the slot map, the dedup result and the episode drops into one manifest.

The manifest is the instruction sheet the merge executes: for every dataset, whether it is
kept, which camera goes into which of the three slots, which episodes to skip, and which
task strings to rewrite.

Datasets are matched by basename, so the tier directory a dataset sits in does not matter.

Input (from --artifacts):
  21_camera_slot_map.json  per-dataset slot assignment
  23_dedup_result.json     keep / drop by action fingerprint
  25_ep_drop.json          per-episode drops (empty, duration mismatch)
Output:
  27_manifest.json

Run:
  python 26_build_manifest.py
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def bn(rel):
    return os.path.basename(rel)


# rowb1 train/val: tangled metadata hid these from the fingerprint, but the action column
# is raw leakage from the parent dataset, so they are dropped by hand.
MANUAL_DROP = {
    "rowb1__so101_pick_cup1_train_final": "dedup-leakage:rowb1__so101_pick_cup1",
    "rowb1__so101_pick_cup1_validation_final": "dedup-leakage:rowb1__so101_pick_cup1",
}

# Prompt normalisation: basename -> {task_index (as string): replacement task}
_KAZU = ("Grab the cubes in areas A, B, and C vertically and put on the plate. "
         "If there are multiple cubes in A, B, and C, grab them in order of "
         "priority from the top. (Priority: A > B > C)")
_PLACE = "Place the cubes on the bench from left to right in the following order: "
TASK_NORM = {
    "CoRL2026-CSI__SO101-teleop_stack_RGBblock_on_bluedish_150epi_10fps":
        {"0": "Stack red, green, and blue blocks on the blue dish from bottom to top."},
    "Damin3927__so101_pickplace": {
        "4": _PLACE + "black, white and brown",
        "5": _PLACE + "black, brown and black",
        "6": _PLACE + "white, white and black",
        "7": _PLACE + "black and brown",
        "8": _PLACE + "white, white and white"},
    "Kazu1232__record-so101-warp-ABC-reverse_A50": {"0": _KAZU},
    "Kazu1232__record-so101-warp-ABC-reverse_B50": {"0": _KAZU},
    "Kazu1232__record-so101-warp-ABC-reverse_C50": {"0": _KAZU},
    "dohoon2665__so101_pick_and_place": {"0": "Pick up the object and place it in the target location."},
    "dohoon2665__so101_pick_and_place_random": {"0": "Pick up the object and place it at a random target location."},
    "kagyvro48__so101_dataset1_arracher_la_mauvaise_herbe": {"0": "Pull out the weed."},
}

# Junk task labels found by hand (task 29 = "Failure." -> ep 33, task 158 = "." -> ep 201)
EXTRA_EP_DROP = {
    "sabinMlminator__so101_pickplace_cubes_test1": [33, 201],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", default=os.path.join(HERE, "artifacts"),
                    help="directory holding the stage 8a-8c JSON files")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: <artifacts>/27_manifest.json)")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.artifacts, "27_manifest.json")
    cam = json.load(open(f"{args.artifacts}/21_camera_slot_map.json", encoding="utf-8"))
    dedup = json.load(open(f"{args.artifacts}/23_dedup_result.json", encoding="utf-8"))
    try:
        epdrop = json.load(open(f"{args.artifacts}/25_ep_drop.json", encoding="utf-8"))
    except FileNotFoundError:
        epdrop = {}
        print("25_ep_drop.json missing: run 24_ep_drop.py first. "
              "Continuing with no episode drops.")

    manifest = {}
    drop_set = set(dedup.get("drop", {}).keys())

    for rel, c in cam.items():
        name = bn(rel)
        entry = {
            "keep": True,
            "drop_reason": None,
            "camera_slots": c.get("slots", {}),
            "masked_slots": c.get("masked_slots", []),
            "dropped_cams": c.get("dropped_cams", []),
            "ambig_cams": c.get("ambig_cams", []),
            "drop_eps": [],
            "task_normalize": None,
        }
        if rel in drop_set:
            entry["keep"] = False
            d = dedup["drop"][rel]
            entry["drop_reason"] = f"dedup-{d['relation']}:{bn(d['kept'])}"
        elif name in MANUAL_DROP:
            entry["keep"] = False
            entry["drop_reason"] = MANUAL_DROP[name]
        # Episode drops. A video that is longer than the action stream is harmless: the
        # loader only decodes the timestamp range the action stream covers and never
        # touches the tail. Only truncated videos and empty episodes are dropped.
        rec = epdrop.get(rel) or next((v for k, v in epdrop.items() if bn(k) == name), {})
        reasons = rec.get("reasons", {})
        eps = set()
        for ep in rec.get("drop_eps", []):
            r = reasons.get(str(ep), "")
            if "duration mismatch" in r:
                m = re.search(r"video([\d.]+)s.*?action([\d.]+)s", r)
                if m and float(m.group(1)) >= float(m.group(2)) * 0.95:
                    continue
            eps.add(ep)
        eps |= set(EXTRA_EP_DROP.get(name, []))
        entry["drop_eps"] = sorted(eps)
        if name in TASK_NORM:
            entry["task_normalize"] = TASK_NORM[name]
        manifest[rel] = entry

    keep = {r: e for r, e in manifest.items() if e["keep"]}
    drop = {r: e for r, e in manifest.items() if not e["keep"]}
    total_ep_drop = sum(len(e["drop_eps"]) for e in keep.values())

    out = {
        "datasets": manifest,
        "summary": {
            "n_total": len(manifest),
            "n_keep": len(keep),
            "n_drop": len(drop),
            "n_ep_drop_in_kept": total_ep_drop,
            "n_task_normalize": sum(1 for e in keep.values() if e["task_normalize"]),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    s = out["summary"]
    print(f"=== manifest -> {out_path} ===")
    print(f"  total {s['n_total']} / keep {s['n_keep']} / drop {s['n_drop']}")
    print(f"  episodes dropped inside kept datasets: {s['n_ep_drop_in_kept']}")
    print(f"  prompt rewrites: {s['n_task_normalize']} datasets")
    print("\n=== kept datasets with episode drops ===")
    for r, e in sorted(keep.items()):
        if e["drop_eps"]:
            print(f"  {bn(r)}: {len(e['drop_eps'])} {e['drop_eps'][:6]}"
                  f"{'...' if len(e['drop_eps'])>6 else ''}")
    print("\n=== kept datasets with the base slot masked (no external camera) ===")
    for r, e in sorted(keep.items()):
        if e["camera_slots"].get("base_0_rgb") is None:
            print(f"  {bn(r)}: masked={e['masked_slots']}")


if __name__ == "__main__":
    main()
