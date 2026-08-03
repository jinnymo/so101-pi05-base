#!/usr/bin/env bash
# Cloud GPU container init, provider independent.
#
# Providers covered:
#   - containers that inject a key through PUBLIC_KEY / SSH_PUBLIC_KEY / AUTHORIZED_KEYS
#   - VMs where you SSH into the host and use docker exec (image independent)
#   - batch training services that override CMD
#   - local runs (no key set: sshd is skipped)
#
# Two modes:
#   1. interactive (default): start sshd, then sleep, and run commands over SSH
#   2. batch: override CMD, e.g. docker run <image> /opt/scripts/train-base ...

set -e

# A volume mounted over /workspace hides directories created at build time, so
# they have to be created again after the mount.
mkdir -p /workspace/datasets /workspace/checkpoints /workspace/logs /workspace/.cache
chmod -R 777 /workspace 2>/dev/null || true
echo "[container-init] /workspace directories ready"
ls -la /workspace 2>&1 | head -10

# The key variable differs by provider.
SSH_KEY="${PUBLIC_KEY:-${SSH_PUBLIC_KEY:-${AUTHORIZED_KEYS:-}}}"

mkdir -p /root/.ssh
chmod 700 /root/.ssh

if [[ -n "$SSH_KEY" ]]; then
    echo "$SSH_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    echo "[container-init] SSH key installed to authorized_keys"
else
    echo "[container-init] no SSH key variable set - skipping sshd (local run, or SSH into the host)"
fi

if [[ -n "$SSH_KEY" ]] && command -v /usr/sbin/sshd >/dev/null 2>&1; then
    mkdir -p /run/sshd
    /usr/sbin/sshd
    echo "[container-init] sshd listening on port 22"
fi

echo "[container-init] idling (sleep infinity)"
exec sleep infinity
