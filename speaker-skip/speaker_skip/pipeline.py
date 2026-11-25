"""End-to-end: audio samples -> who talks when -> skip ranges per enrolled person."""

import json
from pathlib import Path

import numpy as np

from . import audio, speakers

RESULT_VERSION = 2  # bump when analysis changes, so cached results get redone

ROOT = speakers.ROOT
DATA = ROOT / "data"
AUDIO_DIR = DATA / "audio"
RESULTS_DIR = DATA / "results"


def analyze(samples: np.ndarray, progress=None) -> dict:
    segs = speakers.diarize(samples, progress=progress)
    embs = speakers.speaker_embeddings(samples, segs)
    vps = speakers.load_voiceprints()
    labels = speakers.label_speakers(embs, vps)

    talk = {}
    for s, e, spk in segs:
        talk[spk] = talk.get(spk, 0.0) + (e - s)

    info = []
    for spk in sorted(talk, key=talk.get, reverse=True):
        name, sim, sims = labels.get(spk, (None, 0.0, {}))
        info.append({
            "speaker": int(spk),
            "name": name,
            "similarity": round(sim, 3),
            "similarities": {k: round(v, 3) for k, v in sims.items()},
            "seconds": round(talk[spk], 1),
        })

    def label(spk):
        return labels.get(spk, (None,))[0] or f"speaker{spk}"

    return {
        "version": RESULT_VERSION,
        "duration": round(len(samples) / audio.SAMPLE_RATE, 2),
        "speakers": info,
        "skip": {name: speakers.skip_ranges(segs, labels, name) for name in vps},
        "segments": [[round(s, 2), round(e, 2), label(spk)] for s, e, spk in segs],
    }


def result_path(video_id: str) -> Path:
    return RESULTS_DIR / f"{audio.check_video_id(video_id)}.json"


def load_result(video_id: str):
    p = result_path(video_id)
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r if r.get("version") == RESULT_VERSION else None


def analyze_video(video_id: str, status=None, keep_audio=False) -> dict:
    """Download, analyze and cache one video. `status(state, progress)` reports progress."""
    status = status or (lambda *_: None)
    status("downloading", 0.0)
    path = audio.download_audio(video_id, AUDIO_DIR, progress=lambda p: status("downloading", p))
    status("decoding", 0.0)
    samples = audio.decode(path)
    status("analyzing", 0.0)
    result = analyze(samples, progress=lambda p: status("analyzing", p))
    result["videoId"] = video_id
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path(video_id).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    if not keep_audio:
        path.unlink(missing_ok=True)
    status("done", 1.0)
    return result


def enroll(name: str, video_ids, keep_audio=True) -> dict:
    """Build a voiceprint for `name` from videos where they are the main speaker
    (e.g. their own solo channel). The dominant voice in each video is taken."""
    prints, report = [], []
    for vid in video_ids:
        path = audio.download_audio(vid, AUDIO_DIR)
        samples = audio.decode(path)
        segs = speakers.diarize(samples)
        talk = {}
        for s, e, spk in segs:
            talk[spk] = talk.get(spk, 0.0) + (e - s)
        main = max(talk, key=talk.get)
        emb = speakers.speaker_embeddings(samples, [x for x in segs if x[2] == main], max_segments=40)[main]
        prints.append(emb)
        report.append({"video": vid, "main_share": round(talk[main] / sum(talk.values()), 2),
                       "speakers": len(talk)})
        if not keep_audio:
            path.unlink(missing_ok=True)

    if len(prints) > 1:
        sims = [float(np.dot(a, b)) for i, a in enumerate(prints) for b in prints[i + 1:]]
        report.append({"cross_video_similarity": [round(s, 3) for s in sims]})
    vp = np.mean(prints, axis=0)
    vp /= np.linalg.norm(vp) + 1e-9
    speakers.VOICEPRINTS.mkdir(parents=True, exist_ok=True)
    np.save(speakers.VOICEPRINTS / f"{name}.npy", vp)
    return {"name": name, "videos": report}
