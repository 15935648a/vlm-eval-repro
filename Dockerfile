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
# Pin numpy to the NGC base's exact version so transformers/pandas can't replace it —
# a mismatched numpy breaks torch's C bridge ("Numpy is not available"). BASE_NP is read
# from the pristine base numpy at the start of this layer (prior layers were apt-only).
COPY requirements.txt .
RUN BASE_NP="$(python -c 'import numpy; print(numpy.__version__)')" \
    && echo "pinning numpy==${BASE_NP}" \
    && pip install --no-cache-dir "numpy==${BASE_NP}" -r requirements.txt \
    && python -c "import torch, numpy as np; torch.from_numpy(np.zeros(3)); print('numpy bridge OK')"

COPY . .

CMD ["bash"]
