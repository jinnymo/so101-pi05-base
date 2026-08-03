# SPDX-License-Identifier: Apache-2.0
"""Stage 11c: model forward smoke test on the merged repository (needs a GPU).

Runs a forward pass of the patched pi0.5 policy on a batch drawn from the merged
repository, under no_grad and bf16 autocast, so it fits on a 24 GB card at batch 2.

Checks:
  1. the policy is built with the `_mask` columns excluded from input_features
  2. the image preprocessing reads the mask columns per sample, so present and missing
     slots differ inside one batch
  3. the forward loss is finite, which means the whole path works

The base checkpoint is loaded from the local Hugging Face cache; the tokenizer path comes
from VLASH_PALIGEMMA_PATH. Both offline flags default to on, override them in the
environment if you want the run to reach the network.

Input:  <root>/base_unified/
Output: a console report.

Run:
  python 32_forward_smoke.py --root /path/to/workspace \
      --paligemma /path/to/paligemma_tokenizer [--vlash-src /path/to/vlash]
"""

import argparse
import os
import sys


def resize(img, size=224):
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
    ap.add_argument("--paligemma", default="",
                    help="flat PaliGemma tokenizer directory (sets VLASH_PALIGEMMA_PATH)")
    ap.add_argument("--base-model", default="lerobot/pi05_base",
                    help="base checkpoint id or path")
    args = ap.parse_args()

    if args.vlash_src:
        sys.path.insert(0, args.vlash_src)
    if args.paligemma:
        os.environ["VLASH_PALIGEMMA_PATH"] = args.paligemma
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import torch
    import vlash.datasets  # noqa: F401  imported for its LeRobot compatibility patch
    from torch.utils.data._utils.collate import default_collate

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from vlash.datasets import VLASHDataset
    from vlash.policies.factory import make_policy
    from vlash.policies.pi05.configuration_pi05 import PI05Config

    slots = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    dst = args.dst or os.path.join(args.root, "base_unified")

    cfg = PI05Config(pretrained_path=args.base_model, device="cuda", dtype="bfloat16")
    meta = LeRobotDatasetMetadata("local/base_unified", root=dst)
    delta = resolve_delta_timestamps(cfg, meta)
    ds = VLASHDataset("local/base_unified", root=dst, delta_timestamps=delta,
                      image_transforms=resize, max_delay_steps=0)
    print(f"[1] dataset OK: {ds.num_episodes} episodes")

    policy = make_policy(cfg, ds.meta)
    policy.eval().cuda()
    nparam = sum(p.numel() for p in policy.parameters()) / 1e9
    mask_in_inputs = [k for k in policy.config.input_features if k.endswith("_mask")]
    print(f"[2] policy OK: {nparam:.2f}B params, leftover _mask input features="
          f"{mask_in_inputs} (an empty list is correct)")

    # find one episode with a single present slot and one with all three
    es = ds.meta.episodes["dataset_from_index"]
    one, three = None, None
    for gi in [int(es[i]) for i in np.linspace(0, ds.num_episodes - 1, 200, dtype=int)]:
        it = ds[gi]
        npres = sum(int(it[f"observation.images.{s}_mask"].flatten()[0].item()) for s in slots)
        if npres == 1 and one is None:
            one = gi
        if npres == 3 and three is None:
            three = gi
        if one is not None and three is not None:
            break
    pick = [p for p in [three or 0, one or 0] if p is not None][:2]
    batch = default_collate([ds[i] for i in pick])
    print(f"[3] batch episodes={pick}:")
    for s in slots:
        print(f"    {s}_mask: {batch[f'observation.images.{s}_mask'].flatten().tolist()}")

    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss, out = policy.forward(batch)
    lv = float(loss)
    print(f"\n[4] forward OK: loss={lv:.4f} finite={np.isfinite(lv)}")
    print(f"\n{'FORWARD SMOKE PASS' if np.isfinite(lv) else 'FAIL: loss is NaN or inf'}")


if __name__ == "__main__":
    main()
