# SPDX-License-Identifier: Apache-2.0
"""Stage 11d: one training step on the merged repository (needs a GPU).

Runs forward, backward and, if memory allows, one optimizer step of the full fine-tune on
a single 24 GB card: batch 1, gradient checkpointing, bf16, and a bitsandbytes AdamW8bit
optimizer (a full-precision Adam over 3.6B parameters does not fit).

The load-bearing signal is that loss and grad_norm come out finite after backward. The
optimizer step is reported when it fits and skipped when it does not; the real run happens
on a larger card.

Input:  <root>/base_unified/
Output: a console report.

Run:
  python 33_train_smoke.py --root /path/to/workspace \
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
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    if args.vlash_src:
        sys.path.insert(0, args.vlash_src)
    if args.paligemma:
        os.environ["VLASH_PALIGEMMA_PATH"] = args.paligemma
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch
    import vlash.datasets  # noqa: F401  imported for its LeRobot compatibility patch
    from torch.utils.data._utils.collate import default_collate

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from vlash.datasets import VLASHDataset
    from vlash.policies.factory import make_policy
    from vlash.policies.pi05.configuration_pi05 import PI05Config

    dst = args.dst or os.path.join(args.root, "base_unified")

    cfg = PI05Config(pretrained_path=args.base_model, device="cuda",
                     dtype="bfloat16", gradient_checkpointing=True)
    meta = LeRobotDatasetMetadata("local/base_unified", root=dst)
    delta = resolve_delta_timestamps(cfg, meta)
    ds = VLASHDataset("local/base_unified", root=dst, delta_timestamps=delta,
                      image_transforms=resize, max_delay_steps=0)
    print(f"[1] dataset OK: {ds.num_episodes} episodes")

    policy = make_policy(cfg, ds.meta)
    policy.train().cuda()
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad) / 1e9
    print(f"[2] policy in train mode: {n_train:.2f}B trainable (full fine-tune)")

    es = ds.meta.episodes["dataset_from_index"]
    gi = int(es[0])
    batch = default_collate([ds[gi]])
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

    opt = None
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(policy.parameters(), lr=args.lr)
        print("[3] optimizer: bitsandbytes AdamW8bit")
    except Exception as e:
        print(f"[3] 8-bit optimizer unavailable ({str(e)[:60]}), stopping after backward")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy.forward(batch)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    lv, gv = float(loss), float(gnorm)
    print(f"[4] forward+backward OK: loss={lv:.4f} grad_norm={gv:.4f}")

    step_ok = False
    if opt is not None:
        try:
            opt.step()
            opt.zero_grad()
            step_ok = True
            print("[5] optimizer.step() OK, weights updated")
        except torch.cuda.OutOfMemoryError as e:
            print(f"[5] optimizer.step() out of memory, backward signal is enough: "
                  f"{str(e)[:60]}")
        except Exception as e:
            print(f"[5] optimizer.step() failed: {type(e).__name__}: {str(e)[:80]}")

    mem = torch.cuda.max_memory_allocated() / 1e9
    ok = np.isfinite(lv) and np.isfinite(gv)
    print(f"\n[mem] peak {mem:.1f}GB")
    print(f"{'TRAIN SMOKE PASS' if ok else 'FAIL'}: loss and grad_norm finite"
          f"{', optimizer step included' if step_ok else ', backward only'}")


if __name__ == "__main__":
    main()
