# DGX Spark (GB10 Grace-Blackwell, aarch64). Use NVIDIA NGC PyTorch as the base:
# it ships a CUDA torch built for aarch64 + Blackwell (sm_121), which pip wheels do not.
# Bump BASE_IMAGE to a newer tag if torch reports "no kernel image available for device".
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

# ffmpeg: video decoding; libgl1+glib: opencv runtime deps.
# (python, torch, cuda, numpy, pillow already come from the NGC base.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Only the libs the base image doesn't already provide. torch is NOT reinstalled.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
