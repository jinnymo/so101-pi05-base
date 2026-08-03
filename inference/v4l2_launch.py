#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Launcher that forces LeRobot's OpenCV camera backend to V4L2.

LeRobot's get_cv2_backend() returns CAP_ANY on Linux, which fails on some UVC
cameras with set(fourcc/width)=False followed by VIDIOC_QBUF: Bad file descriptor.
CAP_V4L2 works. Replacing the function at import time avoids modifying
site-packages and is reversible.

Usage: python v4l2_launch.py run <config.yaml> [--overrides...]
"""
import cv2
import lerobot.cameras.utils as _utils
import lerobot.cameras.opencv.camera_opencv as _cam


def _v4l2_backend():
    return int(cv2.CAP_V4L2)


_utils.get_cv2_backend = _v4l2_backend
_cam.get_cv2_backend = _v4l2_backend

from vlash.cli import main   # noqa: E402  (import after the patch is applied)

if __name__ == "__main__":
    main()
