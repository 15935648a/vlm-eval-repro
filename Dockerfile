# CUDA runtime base; torch wheels bundle their own CUDA libs, host just needs the driver.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

# ffmpeg: video decoding (decord/opencv); libgl1+glib: opencv runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
