"""End-to-end: audio samples -> who talks when -> skip ranges per enrolled person."""

import json
import time
from pathlib import Path

import numpy as np

from . import audio, speakers, transcript

RESULT_VERSION = 4  # bump when analysis changes, so cached results get redone

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


def _skips(cues, labels, sims, names):
    segs = [(s, e, i) for i, (s, e) in enumerate(cues)]
    lab = {i: (name,) for i, name in enumerate(labels)}
    return {name: speakers.confident_ranges(speakers.skip_ranges(segs, lab, name), cues, sims, k)
            for k, name in enumerate(names)}


def _transcript_result(embs, cues, duration) -> dict:
    vps = speakers.load_voiceprints()
    names = list(vps)
    labels, sims = speakers.label_cues(embs, cues, vps)

    info = []
    for k, name in enumerate(names + [None]):
        idx = [i for i, x in enumerate(labels) if x == name]
        if not idx:
            continue
        dur = np.array([cues[i][1] - cues[i][0] for i in idx])
        entry = {"name": name, "seconds": round(float(dur.sum()), 1)}
        if name is not None:
            entry["similarity"] = round(float((sims[idx, k] * dur).sum() / dur.sum()), 3)
        info.append(entry)
    info.sort(key=lambda x: -x["seconds"])

    merged = []  # consecutive lines by the same person become one segment
    for (s, e), name in zip(cues, labels):
        name = name or "unknown"
        if merged and merged[-1][2] == name and s - merged[-1][1] < 1.0:
            merged[-1][1] = round(e, 2)
        else:
            merged.append([round(s, 2), round(e, 2), name])

    return {
        "version": RESULT_VERSION,
        "method": "transcript",
        "duration": round(duration, 2),
        "speakers": info,
        "skip": _skips(cues, labels, sims, names),
        "segments": merged,
    }


def analyze_transcript(samples: np.ndarray, cues, progress=None) -> dict:
    """Fast path: the transcript says when each line is spoken, the audio says by whom."""
    embs = speakers.cue_embeddings(samples, cues, progress=progress)
    return _transcript_result(embs, cues, len(samples) / audio.SAMPLE_RATE)


PARTIAL_EVERY = 2.0  # seconds between partial results
PARTIAL_MIN_LINES = 30
PARTIAL_HOLD_BACK = 6  # newest lines can still change label, so they are not published yet


def analyze_streaming(download, cues, progress=None, partial=None) -> dict:
    """Like analyze_transcript, but works through the audio while it downloads and reports
    skip ranges for the part done so far via partial({"upto": seconds, "skip": {...}})."""
    vps = speakers.load_voiceprints()
    names = list(vps)
    sr = audio.SAMPLE_RATE
    total = float(download.info.get("duration") or cues[-1][1])
    buf = np.zeros(int((total + 30) * sr), dtype=np.float32)
    embs = np.zeros((len(cues), speakers.extractor().dim), dtype=np.float32)
    have = 0   # samples decoded so far
    done = 0   # caption lines embedded so far
    last = time.time()

    def advance(final=False):
        nonlocal done, last
        limit = have / sr
        while done < len(cues):
            a, b = speakers.cue_window(*cues[done], limit if final else total)
            if b > limit and not final:
                break
            embs[done] = speakers.embed_window(buf[:have], a, b)
            done += 1
        if final or time.time() - last < PARTIAL_EVERY or done < PARTIAL_MIN_LINES:
            return
        last = time.time()
        if progress:
            progress(done / len(cues))
        if partial:
            labels, sims = speakers.label_cues(embs[:done], cues[:done], vps)
            n = done - PARTIAL_HOLD_BACK
            partial({"upto": round(cues[n - 1][1], 2),
                     "skip": _skips(cues[:n], labels[:n], sims[:n], names)})

    for chunk in download.samples():
        if have + len(chunk) > len(buf):
            buf = np.concatenate([buf, np.zeros(len(buf) // 4 + len(chunk), dtype=np.float32)])
        buf[have:have + len(chunk)] = chunk
        have += len(chunk)
        advance()
    advance(final=True)
    return _transcript_result(embs, cues, have / sr)


def result_path(video_id: str) -> Path:
    return RESULTS_DIR / f"{audio.check_video_id(video_id)}.json"


def load_result(video_id: str):
    p = result_path(video_id)
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r if r.get("version") == RESULT_VERSION else None


def analyze_video(video_id: str, status=None, keep_audio=False, method="auto", partial=None) -> dict:
    """Download, analyze and cache one video. `status(state, progress)` reports progress.

    method "auto" uses the transcript when the video has one, analyzing while the audio
    downloads and reporting early skip ranges through `partial`; without a transcript it
    falls back to full diarization. "full" forces that slow method."""
    status = status or (lambda *_: None)
    status("downloading", 0.0)
    download = audio.Download(video_id, AUDIO_DIR)
    cues, note = None, "full analysis requested"
    if method != "full":
        try:
            cues, note = transcript.fetch_cues(video_id, download.info)
        except Exception as e:  # any transcript trouble just means the slow path
            cues, note = None, f"transcript unavailable: {e}"
    download.start()
    result = None
    if cues:
        status("analyzing", 0.0)
        try:
            result = analyze_streaming(download, cues, progress=lambda p: status("analyzing", p), partial=partial)
        except Exception as e:
            if download.error:
                raise
            note += f"; could not analyze during download ({e})"
    path = download.wait()
    if result is None:
        status("decoding", 0.0)
        samples = audio.decode(path)
        status("analyzing", 0.0)
        report = lambda p: status("analyzing", p)
        if cues:
            result = analyze_transcript(samples, cues, progress=report)
        else:
            result = analyze(samples, progress=report)
            result["method"] = "full"
    result["note"] = note
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
