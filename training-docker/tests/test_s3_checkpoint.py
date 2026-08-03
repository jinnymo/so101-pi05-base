import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import s3_checkpoint as sc
from s3_checkpoint import select_local_rolling_to_delete

S3CKPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "s3_checkpoint.py")


def test_verify_manifest_ok(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    manifest = [{"name": "a.bin", "size": 10}]
    assert sc.verify_manifest(tmp_path, manifest) is True


def test_verify_manifest_size_mismatch(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 5)
    manifest = [{"name": "a.bin", "size": 10}]
    assert sc.verify_manifest(tmp_path, manifest) is False


def test_verify_manifest_missing_file(tmp_path):
    manifest = [{"name": "ghost.bin", "size": 10}]
    assert sc.verify_manifest(tmp_path, manifest) is False


def test_build_manifest_lists_files_with_size(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f.bin").write_bytes(b"abc")
    manifest = sc.build_manifest(tmp_path)
    assert {"name": "sub/f.bin", "size": 3} in manifest


def test_select_rolling_to_delete_keeps_n(tmp_path):
    for s in (1000, 2000, 3000):
        (tmp_path / f"ckpt-{s}").mkdir()
    to_del = sc.select_rolling_to_delete(tmp_path, keep=2)
    assert [p.name for p in to_del] == ["ckpt-1000"]


def test_select_rolling_to_delete_under_limit(tmp_path):
    (tmp_path / "ckpt-1000").mkdir()
    assert sc.select_rolling_to_delete(tmp_path, keep=2) == []


def test_steps_to_delete_keeps_n():
    assert sc.steps_to_delete([1000, 2000, 3000], keep=2) == [1000]


def test_steps_to_delete_unsorted():
    assert sc.steps_to_delete([3000, 1000, 2000], keep=2) == [1000]


def test_steps_to_delete_under_limit():
    assert sc.steps_to_delete([1000], keep=2) == []


def test_build_manifest_excludes_manifest_json(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 4)
    (tmp_path / "manifest.json").write_text("[]")
    names = [e["name"] for e in sc.build_manifest(tmp_path)]
    assert "a.bin" in names and "manifest.json" not in names


def _mk(tmp_path, steps):
    for s in steps:
        (tmp_path / f"{s:06d}").mkdir()
    return tmp_path


def test_local_rolling_keep2(tmp_path):
    _mk(tmp_path, [0, 2000, 4000, 6000])
    got = sorted(p.name for p in select_local_rolling_to_delete(tmp_path, keep=2))
    assert got == ["000000", "002000"]


def test_local_rolling_protect_never_deleted(tmp_path):
    _mk(tmp_path, [0, 2000, 4000])
    got = [p.name for p in select_local_rolling_to_delete(tmp_path, keep=1, protect="000000")]
    assert got == ["002000"]


def test_local_rolling_ignores_nondigit(tmp_path):
    _mk(tmp_path, [0])
    (tmp_path / "last").mkdir()
    assert select_local_rolling_to_delete(tmp_path, keep=2) == []


def test_prune_local_cli_no_s3_env(tmp_path):
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for s in [0, 2000, 4000, 6000]:
        (ckpts / f"{s:06d}").mkdir()
    (ckpts / "last").symlink_to("006000")
    env = {k: v for k, v in os.environ.items() if k != "S3_CKPT_BASE"}
    r = subprocess.run([sys.executable, S3CKPT, "prune-local", str(ckpts), "2"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    remaining = sorted(p.name for p in ckpts.iterdir() if p.is_dir() and p.name.isdigit())
    assert remaining == ["004000", "006000"]
