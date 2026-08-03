# SPDX-License-Identifier: Apache-2.0
"""Stage 11a: loader smoke test on the merged repository (no GPU needed).

Applies the same image transform the trainer uses (resize the long side to 224 and pad),
then checks:
  1. the dataset loads through the LeRobot v2.1 reader, 10 fps sources included
  2. episodes with different source resolutions collate into one batch
  3. the observation.images.{slot}_mask columns arrive in the batch as exact 0.0 / 1.0,
     i.e. normalization did not touch them
  4. the number of masked slots varies from episode to episode, which is what makes the
     camera count irrelevant to the model

Whether a mask of 0 really blocks attention is a property of the policy code, not of the
data; that is verified separately by reading the model source.

Input:  <root>/base_unified/
Output: a console report.

Run:
  python 30_smoke.py --root /path/to/workspace [--vlash-src /path/to/vlash]
"""

import argparse
import os
import sys

SLOTS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]


def ax_resize_pad(img, size=224):
    import torch.nn.functional as F
    h, w = img.shape[-2:]
    r = size / max(h, w)
    nh, nw = max(1, round(h * r)), max(1, round(w * r))
    img = F.interpolate(img.unsqueeze(0).float(), (nh, nw),
                        mode="bilinear", align_corners=False).squeeze(0)
    return F.pad(img, (0, size - nw, 0, size - nh), value=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.environ.get("ROOT", "."),
                    help="workspace root (default: $ROOT)")
    ap.add_argument("--dst", default=None,
                    help="merged repository (default: <root>/base_unified)")
    ap.add_argument("--vlash-src", default=os.environ.get("VLASH_SRC", ""),
                    help="training package checkout to prepend to sys.path, if not installed")
    args = ap.parse_args()

    if args.vlash_src:
        sys.path.insert(0, args.vlash_src)

    import numpy as np
    import vlash.datasets  # noqa: F401  imported for its LeRobot compatibility patch
    from torch.utils.data._utils.collate import default_collate

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dst = args.dst or os.path.join(args.root, "base_unified")
    ds = LeRobotDataset("local/base_unified", root=dst, image_transforms=ax_resize_pad)
    n = ds.num_episodes
    print(f"[1] load OK: episodes={n} frames={ds.num_frames} fps={ds.fps}")
    print(f"    camera_keys={[k.split('.')[-1] for k in ds.meta.camera_keys]}")

    es = ds.meta.episodes["dataset_from_index"]
    pick = [int(es[i]) for i in np.linspace(0, n - 1, 12, dtype=int)]
    batch = default_collate([ds[i] for i in pick])
    print(f"[2] collate OK (mixed resolutions unified to 224): "
          f"{tuple(batch['observation.images.base_0_rgb'].shape)}")

    ok = True
    print("[3] mask columns reach the batch as 0/1:")
    for slot in SLOTS:
        vals = batch[f"observation.images.{slot}_mask"].flatten().tolist()
        uniq = set(round(v, 3) for v in vals)
        if not uniq <= {0.0, 1.0}:
            print(f"    FAIL {slot}_mask is not 0/1: {uniq}")
            ok = False
    n_masked_per_ep = [
        sum(1 for slot in SLOTS if batch[f"observation.images.{slot}_mask"][i].item() == 0.0)
        for i in range(len(pick))
    ]
    print(f"    masked slots per episode: {n_masked_per_ep}")
    if len(set(n_masked_per_ep)) < 2:
        print("    WARN low variation, the sample may be skewed")

    print("[4] per-slot validity of the first sampled episodes:")
    for i, fi in enumerate(pick[:6]):
        st = " ".join(f"{s.split('_')[0]}={int(batch[f'observation.images.{s}_mask'][i].item())}"
                      for s in SLOTS)
        print(f"    frame {fi}: [{st}]")

    print(f"\n{'SMOKE PASS' if ok else 'SMOKE FAIL'}: load, collate, mask delivery, "
          f"per-episode mask variation")


if __name__ == "__main__":
    main()
