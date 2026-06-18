"""Uniform frame sampling. Deterministic: same video + same NUM_FRAMES -> same frames.

Primary decoder is ffmpeg-via-subprocess decoding straight to PIL — this deliberately avoids
numpy/opencv, whose prebuilt aarch64 wheels clash with the NGC base image's numpy ABI
("numpy.core.multiarray failed to import"). decord/opencv remain as fallbacks only.
"""
import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import List

from PIL import Image


def _duration_seconds(video_path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", video_path],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def _grab_frame_at(video_path: str, ts: float) -> Image.Image:
    # fast seek (-ss before -i), decode a single frame to PNG on stdout.
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{ts:.3f}",
         "-i", video_path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
        capture_output=True, check=True,
    )
    if not proc.stdout:  # seek landed past the end; fall back to the first frame
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", video_path,
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
            capture_output=True, check=True,
        )
    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")


def _sample_frames_ffmpeg(video_path: str, num_frames: int) -> List[Image.Image]:
    dur = _duration_seconds(video_path)
    # sample at the midpoint of each of num_frames equal segments (avoids t=0 and the very end).
    timestamps = [dur * (i + 0.5) / num_frames for i in range(num_frames)]
    return [_grab_frame_at(video_path, t) for t in timestamps]


def sample_frames(video_path, num_frames: int = 8) -> List[Image.Image]:
    """Return `num_frames` PIL.Image RGB frames sampled uniformly across the clip."""
    video_path = str(video_path)
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return _sample_frames_ffmpeg(video_path, num_frames)
    return _sample_frames_cv2(video_path, num_frames)  # last-resort fallback


def _sample_frames_cv2(video_path: str, num_frames: int) -> List[Image.Image]:
    import cv2  # noqa: only reached if ffmpeg is unavailable

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = ([min(i, total - 1) for i in range(num_frames)] if total <= num_frames
           else [round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)])
    frames: List[Image.Image] = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(frame[:, :, ::-1]))  # BGR->RGB
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read frames from {video_path}")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def iter_videos(video_dir) -> List[Path]:
    return sorted(Path(video_dir).rglob("*.mp4"))
