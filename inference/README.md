# Inference

The two files the closed-loop hardware evaluation is driven by. There is no inference code of its
own here: the policy runs from the patched VLASH files in `../training-docker/patched/`, and these
two supply the launch path and the configuration.

[`../docs/06-inference.md`](../docs/06-inference.md) is the reference for both — environment,
checkpoint layout, every runtime option, the evaluation procedure and what the policy did.

| File | What it is |
|---|---|
| `v4l2_launch.py` | Launcher that replaces LeRobot's OpenCV backend selector with `cv2.CAP_V4L2` before importing the VLASH CLI. Needed only where `CAP_ANY` fails on the camera; see `docs/06-inference.md` 5.4 |
| `eval.yaml` | Robot, camera and policy configuration. This is the configuration section 7 of that document reports as validated on hardware |

`eval.yaml` ships with `<...>` placeholders. Fill in the serial port, both camera device nodes, the
calibration directory, the robot id and the checkpoint root before the first run. Identify the two
camera nodes by capturing a frame from each rather than by enumeration order — swapping the
overhead and wrist views raises nothing and degrades the policy.

Run from the package root, so the two paths below resolve as written:

```bash
python inference/v4l2_launch.py run inference/eval.yaml --policy.path=<CKPT_ROOT>/40000
```

Every key under `policy:` other than `path` becomes a config override on top of the checkpoint's own
`config.json`; anything absent keeps the checkpoint's value. Command-line flags override the YAML,
which is how the dry run and the checkpoint sweep in section 8 vary one setting at a time.

Where `CAP_ANY` opens the cameras correctly the launcher is unnecessary and the same command is
`vlash run inference/eval.yaml ...`.

**Dry run before commanding motors.** `VLASH_DRY_RUN=1` runs cameras, policy and action decoding
and prints the action dict without calling `send_action`. Torque is still enabled and the arm still
holds position, so it is live hardware either way.
