"""The image transform must stay picklable, or DataLoader workers cannot start.

Requires the vlash package on sys.path with the patched train.py applied. Set
VLASH_SRC to the vlash source tree; the default is the path used inside the
image.
"""
import functools
import os
import pickle
import sys

import pytest
import torch

sys.path.insert(0, os.environ.get("VLASH_SRC", "/opt/vlash"))
pytest.importorskip("vlash")

import vlash.datasets  # noqa: F401  applies the compatibility layer first
from vlash.train import _resize_with_pad, _resize_with_pad_then_aug


def test_resize_pad_is_module_level_and_picklable():
    pickle.loads(pickle.dumps(_resize_with_pad))
    out = _resize_with_pad(torch.zeros(3, 480, 640))
    assert out.shape == (3, 224, 224)


def test_compose_partial_picklable():
    functools.partial(_resize_with_pad_then_aug, lambda x: x)  # aug=identity
    pickle.loads(pickle.dumps(_resize_with_pad))  # the aug=None path is what matters
    out = _resize_with_pad_then_aug(lambda x: x, torch.zeros(3, 1080, 1920))
    assert out.shape == (3, 224, 224)
