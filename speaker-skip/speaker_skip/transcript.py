"""Fetch a video's transcript timing: when each caption line is spoken.

Only the timing is used (to know where lines start and end); the text is ignored.
"""

import json

import yt_dlp

from .audio import check_video_id


def _pick_track(info):
    """Prefer the uploader's own Chinese captions, then any of theirs, then auto-generated."""
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    manual = {k: v for k, v in manual.items() if k != "live_chat"}
    orig = [k for k in auto if k.endswith("-orig")]
    candidates = (
        [("manual", k, manual[k]) for k in manual if k.startswith("zh")]
        + [("manual", k, manual[k]) for k in manual]
        + [("auto", k, auto[k]) for k in orig]
        + [("auto", k, auto[k]) for k in auto if k.startswith("zh")]
    )
    for kind, lang, formats in candidates:
        for f in formats:
            if f.get("ext") == "json3" and f.get("url"):
                return kind, lang, f["url"]
    return None


def parse_json3(data):
    """json3 caption events -> sorted [(start_s, end_s), ...], one per caption line."""
    cues = []
    for ev in data.get("events", []):
        text = "".join(s.get("utf8", "") for s in ev.get("segs") or [])
        if not text.strip() or "tStartMs" not in ev:
            continue
        start = ev["tStartMs"] / 1000
        cues.append([start, start + ev.get("dDurationMs", 0) / 1000])
    cues.sort()
    out = []
    for i, (s, e) in enumerate(cues):
        if i + 1 < len(cues):
            e = min(e, cues[i + 1][0])  # auto captions overlap the next line
        if out and s - out[-1][0] < 0.05:
            out[-1][1] = max(out[-1][1], e)  # duplicate start: same line
        elif e - s > 0.05:
            out.append([s, e])
    return [(s, e) for s, e in out]


def fetch_cues(video_id, info=None):
    """Return (cues, description) or (None, reason) when the video has no usable transcript."""
    check_video_id(video_id)
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        if info is None:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        track = _pick_track(info)
        if not track:
            return None, "no transcript"
        kind, lang, url = track
        try:
            data = json.loads(ydl.urlopen(url).read().decode("utf-8"))
        except Exception as e:
            return None, f"transcript download failed: {e}"
    cues = parse_json3(data)
    duration = info.get("duration") or (cues[-1][1] if cues else 0)
    covered = sum(e - s for s, e in cues)
    # A transcript that covers little of the video can't tell us where people talk.
    if len(cues) < 20 or (duration and covered < 0.4 * duration):
        return None, f"transcript too sparse ({len(cues)} lines)"
    return cues, f"{kind} {lang}, {len(cues)} lines"
