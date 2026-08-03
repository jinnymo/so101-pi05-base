#!/usr/bin/env python3
"""Back up and restore checkpoint slots on S3, for preemptible instances.

The pure selection logic below is covered by pytest; the aws calls and the CLI
need a real S3 round trip.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

LATEST = "LATEST.json"
MANIFEST = "manifest.json"


def build_manifest(root: Path) -> list[dict]:
    """List every file under root as {name (relative path), size}, excluding manifest.json."""
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != MANIFEST:
            out.append({"name": str(p.relative_to(root)), "size": p.stat().st_size})
    return out


def verify_manifest(root: Path, manifest: list[dict]) -> bool:
    """True when every file in the manifest exists under root with a matching size."""
    for entry in manifest:
        f = root / entry["name"]
        if not f.is_file() or f.stat().st_size != entry["size"]:
            return False
    return True


def select_rolling_to_delete(resume_root: Path, keep: int = 2) -> list[Path]:
    """Local ckpt-<step> directories to delete, keeping the newest `keep`."""
    ckpts = sorted(
        (p for p in resume_root.glob("ckpt-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    return ckpts[:-keep] if len(ckpts) > keep else []


def steps_to_delete(steps: list[int], keep: int = 2) -> list[int]:
    """Step numbers to delete from S3, keeping the newest `keep`."""
    s = sorted(steps)
    return s[:-keep] if len(s) > keep else []


def select_local_rolling_to_delete(
    checkpoints_dir: Path, keep: int = 2, protect: str | None = None
) -> list[Path]:
    """Step directories (six-digit names) to delete from a local lerobot checkpoints/ dir.

    The newest `keep` directories, and the one `protect` names (the target of the
    `last` symlink), are never deleted.
    """
    dirs = sorted(
        (p for p in Path(checkpoints_dir).iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    survivors = {d.name for d in dirs[-keep:]} if keep > 0 else set()
    if protect:
        survivors.add(protect)
    return [d for d in dirs if d.name not in survivors]


def _sh(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def _s3_cp(src: str, dst: str) -> None:
    _sh("aws", "s3", "cp", src, dst)


def _s3_sync(src: str, dst: str) -> None:
    _sh("aws", "s3", "sync", "--only-show-errors", src, dst)


def _s3_rm(uri: str) -> None:
    _sh("aws", "s3", "rm", "--recursive", uri)


def _read_latest(base: str, slot: str) -> dict | None:
    uri = f"{base}/resume/{slot}/{LATEST}"
    try:
        return json.loads(_sh("aws", "s3", "cp", uri, "-"))
    except subprocess.CalledProcessError:
        return None  # empty slot: start from scratch


def _list_ckpt_steps(base: str, slot: str) -> list[int]:
    """Step numbers taken from the ckpt-<step>/ prefixes under resume/<slot>/."""
    try:
        out = _sh("aws", "s3", "ls", f"{base}/resume/{slot}/")
    except subprocess.CalledProcessError:
        return []
    steps = []
    for line in out.splitlines():
        m = re.search(r"PRE ckpt-(\d+)/", line)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def download(base: str, slot: str, output_dir: str) -> int:
    """Restore a slot into output_dir/checkpoints/<step>/ and point `last` at it.

    Falls back to the previous checkpoint when the newest one fails verification.
    Returns the restored step, or -1 when nothing could be restored.
    """
    meta = _read_latest(base, slot)
    if meta is None:
        print(f"[s3ckpt] no LATEST for slot={slot} -> fresh start")
        return -1
    ckpt_root = Path(output_dir) / "checkpoints"

    # Candidates: the LATEST step first, then the one rolling checkpoint below it
    candidates = [int(meta["step"])]
    for s in sorted(_list_ckpt_steps(base, slot), reverse=True):
        if s < int(meta["step"]):
            candidates.append(s)
            break

    for step in candidates:
        # lerobot layout: the checkpoint lives in <step>/ and `last` is a symlink
        # to it, which keeps update_last_checkpoint working.
        step_dir = ckpt_root / f"{step:06d}"
        if step_dir.exists():
            shutil.rmtree(step_dir)
        step_dir.mkdir(parents=True, exist_ok=True)
        _s3_sync(f"{base}/resume/{slot}/ckpt-{step}", str(step_dir))
        try:
            mf = json.loads((step_dir / MANIFEST).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            mf = []
        if verify_manifest(step_dir, mf):
            last = ckpt_root / "last"
            if last.is_symlink() or last.is_file():
                last.unlink()
            elif last.is_dir():
                shutil.rmtree(last)
            last.symlink_to(f"{step:06d}")  # relative, same parent directory
            print(f"[s3ckpt] restored step={step} -> {step_dir} (last -> {step:06d})")
            return step
        print(f"[s3ckpt] WARN ckpt-{step} manifest mismatch -> trying the previous checkpoint")

    print(f"[s3ckpt] WARN slot={slot} every candidate failed -> fresh start")
    return -1


def upload_resume(base: str, slot: str, ckpt_dir: str, step: int, keep: int = 2) -> None:
    """Upload ckpt_dir with its manifest, then PUT LATEST.json, then trim to `keep`."""
    ckpt = Path(ckpt_dir)
    # Written after build_manifest, so the manifest excludes itself
    (ckpt / MANIFEST).write_text(json.dumps(build_manifest(ckpt)))
    rel = f"resume/{slot}/ckpt-{step}"
    _s3_sync(str(ckpt), f"{base}/{rel}")

    tmp = ckpt.parent / f".{LATEST}.{step}"
    tmp.write_text(json.dumps({"step": step, "ckpt_path": rel}))
    _s3_cp(str(tmp), f"{base}/resume/{slot}/{LATEST}")  # last PUT, so it is atomic
    tmp.unlink()

    for s in steps_to_delete(_list_ckpt_steps(base, slot), keep):
        _s3_rm(f"{base}/resume/{slot}/ckpt-{s}")
    print(f"[s3ckpt] uploaded resume ckpt step={step}")


def upload_archive(base: str, slot: str, adapter_dir: str, step: int) -> None:
    _s3_sync(adapter_dir, f"{base}/archive/{slot}/{step}")
    print(f"[s3ckpt] uploaded archive step={step}")


def cleanup(base: str, slot: str) -> None:
    _s3_rm(f"{base}/resume/{slot}/")
    print(f"[s3ckpt] cleaned resume slot={slot}")


def prune_local(checkpoints_dir: str, keep: int = 2) -> None:
    """Trim a local lerobot checkpoints/ dir to the newest `keep` plus the `last` target."""
    root = Path(checkpoints_dir)
    if not root.is_dir():
        return
    last = root / "last"
    protect = os.path.basename(os.readlink(last)) if last.is_symlink() else None
    for d in select_local_rolling_to_delete(root, keep, protect):
        shutil.rmtree(d, ignore_errors=True)
        print(f"[s3ckpt] pruned local ckpt {d.name}")


USAGE = "usage: s3_checkpoint.py {download|upload-resume|upload-archive|cleanup|prune-local} ..."


def main() -> None:
    if len(sys.argv) < 2:
        print(USAGE, file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)
    try:
        if cmd == "prune-local":
            prune_local(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
        elif cmd == "download":
            download(os.environ["S3_CKPT_BASE"], sys.argv[2], sys.argv[3])
        elif cmd == "upload-resume":
            upload_resume(os.environ["S3_CKPT_BASE"], sys.argv[2], sys.argv[3], int(sys.argv[4]))
        elif cmd == "upload-archive":
            upload_archive(os.environ["S3_CKPT_BASE"], sys.argv[2], sys.argv[3], int(sys.argv[4]))
        elif cmd == "cleanup":
            cleanup(os.environ["S3_CKPT_BASE"], sys.argv[2])
        else:
            print(f"unknown cmd: {cmd}", file=sys.stderr)
            sys.exit(2)
    except subprocess.CalledProcessError as e:
        # An aws failure is best effort: report the cause and skip this step.
        # Training continues, because train.py invokes this with check=False.
        msg = (e.stderr or "").strip() or str(e)
        print(f"[s3ckpt] aws call failed, skipping this step: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
