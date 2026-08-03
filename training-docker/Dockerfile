# syntax=docker/dockerfile:1.4
# Full fine-tune image for pi0.5 on a unified 3-slot SO-101 dataset
# (lora.enable=false). NOTICE lists what the patched files change.

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Pinned vlash commit. The patched files were written against this revision.
ARG VLASH_COMMIT=22cbabfee0f57874987c75a35a7dac129e695db0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget ca-certificates \
        ffmpeg libsm6 libxext6 libgl1 \
        openssh-client openssh-server sshpass \
        unzip jq gettext-base \
        linux-libc-dev build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

# AWS CLI v2, for S3 datasets and checkpoints
RUN curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp/ \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscli.zip

# Miniforge, not the Anaconda installer: conda-forge only, no Anaconda ToS.
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/conda.sh \
    && bash /tmp/conda.sh -b -p $CONDA_DIR \
    && rm /tmp/conda.sh \
    && $CONDA_DIR/bin/conda init bash
ENV PATH="$CONDA_DIR/bin:$PATH"

RUN conda config --remove channels defaults 2>/dev/null || true \
    && conda config --set channel_priority strict

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy

RUN git clone https://github.com/mit-han-lab/vlash.git /opt/vlash \
    && cd /opt/vlash \
    && git checkout $VLASH_COMMIT

COPY patched/train.py /opt/vlash/vlash/train.py
COPY patched/run.py /opt/vlash/vlash/run.py
COPY patched/modeling_pi05.py /opt/vlash/vlash/policies/pi05/modeling_pi05.py

RUN conda run -n vlash pip install -e /opt/vlash

# lerobot patch: bounded video decoder cache and a decode fallback.
# lerobot arrives as a transitive dependency of vlash, so this overwrite has to
# happen after `pip install -e`. Cache size: VLASH_DECODER_CACHE_MAX (default 64).
COPY patched/video_utils.py /opt/conda/envs/vlash/lib/python3.10/site-packages/lerobot/datasets/video_utils.py

# Base weights are baked in, so a long run needs no hub access.
# local_dir_use_symlinks is not passed: it was deprecated in huggingface_hub 0.23
# and removed in 1.0. Since 0.23 a local_dir download already writes real files.
RUN mkdir -p /opt/models/pi05_base \
    && conda run -n vlash python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('lerobot/pi05_base', local_dir='/opt/models/pi05_base')"

# The PaliGemma tokenizer is not vendored in this package: google/paligemma-3b-pt-224
# is a gated repository. Run scripts/fetch-tokenizer.sh once, with a Hugging Face
# account that has accepted the Gemma terms and been granted access, before building.
COPY paligemma_tokenizer_flat /opt/models/paligemma_tokenizer

COPY configs/base.yaml /opt/configs/base.yaml
COPY scripts/train-base /opt/scripts/train-base
COPY scripts/prepare-dataset /opt/scripts/prepare-dataset
COPY scripts/inject-stats /opt/scripts/inject-stats
COPY scripts/container-init.sh /opt/scripts/container-init.sh
COPY scripts/s3_checkpoint.py /opt/scripts/s3_checkpoint.py

RUN chmod +x /opt/scripts/*
ENV PATH="/opt/scripts:/opt/conda/envs/vlash/bin:$PATH"

# Applies PATH and the cache environment to SSH login shells as well
RUN cat > /etc/profile.d/train-env.sh <<'PROFILE_EOF' \
 && chmod +x /etc/profile.d/train-env.sh
export PATH="/opt/scripts:/opt/conda/envs/vlash/bin:/opt/conda/bin:${PATH}"
export HOME=${HOME:-/workspace}
export XDG_CACHE_HOME=/workspace/.cache
export TRITON_CACHE_DIR=/workspace/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torch_inductor
export TORCH_HOME=/workspace/.cache/torch
export TRANSFORMERS_CACHE=/workspace/.cache/transformers
export MPLCONFIGDIR=/workspace/.cache/matplotlib
export HF_HOME=/opt/models/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLASH_PALIGEMMA_PATH=/opt/models/paligemma_tokenizer
PROFILE_EOF

ENV HOME=/workspace
ENV XDG_CACHE_HOME=/workspace/.cache
ENV TRITON_CACHE_DIR=/workspace/.cache/triton
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torch_inductor
ENV TORCH_HOME=/workspace/.cache/torch
ENV TRANSFORMERS_CACHE=/workspace/.cache/transformers
ENV MPLCONFIGDIR=/workspace/.cache/matplotlib

ENV HF_HOME=/opt/models/hf_cache
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV VLASH_PALIGEMMA_PATH=/opt/models/paligemma_tokenizer

RUN mkdir -p /workspace/checkpoints /workspace/.cache /workspace/logs \
    && chmod -R 777 /workspace /tmp \
    && chmod -R 777 /root/.cache /root/.config 2>/dev/null || true

WORKDIR /workspace

# Default CMD is the interactive SSH init. For batch runs, override it:
#   docker run <image> /opt/scripts/train-base --dataset-url=... --steps=...
ENTRYPOINT []
CMD ["/opt/scripts/container-init.sh"]
