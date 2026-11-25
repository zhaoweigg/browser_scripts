"""Fetch a YouTube video's audio track and decode it to 16 kHz mono float32."""

import os
import re
from pathlib import Path

import av
import numpy as np
import yt_dlp

SAMPLE_RATE = 16000
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def check_video_id(video_id: str) -> str:
    if not VIDEO_ID_RE.match(video_id or ""):
        raise ValueError(f"not a YouTube video id: {video_id!r}")
    return video_id


def fetch(video_id: str, out_dir: Path, progress=None):
    """Download the smallest Opus audio stream (plenty for speech).

    Returns (audio_path, info); `info` is yt-dlp's metadata, which also lists the captions."""
    check_video_id(video_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out_dir.glob(f"{video_id}.*") if not p.name.endswith(".part")]

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress(d.get("downloaded_bytes", 0) / total)

    opts = {
        "format": "249/250/251/bestaudio",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=not existing)
        return (existing[0] if existing else Path(ydl.prepare_filename(info))), info


def download_audio(video_id: str, out_dir: Path, progress=None) -> Path:
    existing = [p for p in out_dir.glob(f"{check_video_id(video_id)}.*") if not p.name.endswith(".part")]
    return existing[0] if existing else fetch(video_id, out_dir, progress)[0]


def decode(path: Path) -> np.ndarray:
    """Decode any audio file to mono float32 at 16 kHz."""
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
