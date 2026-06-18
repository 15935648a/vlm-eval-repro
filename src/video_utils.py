"""Uniform frame sampling. Deterministic: same video + same NUM_FRAMES -> same frames."""
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def _uniform_indices(total: int, num_frames: int) -> List[int]:
    if total <= 0:
        raise ValueError("video has no frames")
    if total <= num_frames:
        # repeat-pad short clips so we always return exactly num_frames frames
        return [min(i, total - 1) for i in range(num_frames)]
    return [int(round(x)) for x in np.linspace(0, total - 1, num_frames)]


def sample_frames(video_path, num_frames: int = 8) -> List[Image.Image]:
    """Return `num_frames` PIL.Image RGB frames sampled uniformly across the clip."""
    video_path = str(video_path)
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        idx = _uniform_indices(len(vr), num_frames)
        batch = vr.get_batch(idx).asnumpy()  # (N, H, W, 3) uint8 RGB
        return [Image.fromarray(f).convert("RGB") for f in batch]
    except ImportError:
        return _sample_frames_cv2(video_path, num_frames)


def _sample_frames_cv2(video_path: str, num_frames: int) -> List[Image.Image]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = _uniform_indices(total, num_frames)
    frames: List[Image.Image] = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read frames from {video_path}")
    while len(frames) < num_frames:  # pad if some reads failed
        frames.append(frames[-1])
    return frames[:num_frames]


def iter_videos(video_dir) -> List[Path]:
    return sorted(Path(video_dir).rglob("*.mp4"))
