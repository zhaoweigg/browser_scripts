"""Who-spoke-when, and which enrolled voice each speaker is."""

from pathlib import Path

import numpy as np
import sherpa_onnx

from .audio import SAMPLE_RATE

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SEG_MODEL = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB_MODEL = MODELS / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
VOICEPRINTS = ROOT / "data" / "voiceprints"

NUM_THREADS = 6
# Cosine similarity a speaker must reach against a voiceprint to be labelled as that person.
MATCH_THRESHOLD = 0.6

_extractor = None


def extractor():
    global _extractor
    if _extractor is None:
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL), num_threads=NUM_THREADS)
        _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    return _extractor


def embed(samples: np.ndarray) -> np.ndarray:
    ext = extractor()
    stream = ext.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    stream.input_finished()
    e = np.asarray(ext.compute(stream), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


def diarize(samples: np.ndarray, progress=None, threshold: float = 0.5, num_speakers: int = -1):
    """Return [(start_s, end_s, speaker_index), ...] sorted by start."""
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL)),
            num_threads=NUM_THREADS,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL), num_threads=NUM_THREADS),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    def cb(done, total):
        if progress and total:
            progress(done / total)
        return 0

    result = sd.process(samples, callback=cb).sort_by_start_time()
    return [(r.start, r.end, r.speaker) for r in result]


def speaker_embeddings(samples: np.ndarray, segments, max_segments: int = 25):
    """Average embedding per diarized speaker, from their longest turns."""
    by_spk = {}
    for s, e, spk in segments:
        by_spk.setdefault(spk, []).append((s, e))
    out = {}
    for spk, segs in by_spk.items():
        segs = sorted(segs, key=lambda x: x[0] - x[1])[:max_segments]  # longest first
        embs = []
        for s, e in segs:
            if e - s < 1.0:
                continue
            chunk = samples[int(s * SAMPLE_RATE): int(min(e, s + 20) * SAMPLE_RATE)]
            embs.append(embed(chunk))
        if embs:
            m = np.mean(embs, axis=0)
            out[spk] = m / (np.linalg.norm(m) + 1e-9)
    return out


def load_voiceprints():
    return {p.stem: np.load(p) for p in sorted(VOICEPRINTS.glob("*.npy"))}


def label_speakers(spk_embs, voiceprints, threshold=MATCH_THRESHOLD):
    """Map diarized speaker index -> (name or None, best similarity, all similarities)."""
    labels = {}
    for spk, e in spk_embs.items():
        sims = {name: float(np.dot(e, vp)) for name, vp in voiceprints.items()}
        best = max(sims, key=sims.get) if sims else None
        name = best if best and sims[best] >= threshold else None
        labels[spk] = (name, sims.get(best, 0.0) if best else 0.0, sims)
    return labels


def skip_ranges(segments, labels, target, merge_gap=1.0, min_len=2.0, other_min=1.0):
    """Time ranges where `target` talks and nobody else does.

    The target's turns are merged across short pauses, then real speech from anyone
    else (turns >= other_min seconds; short "嗯/对" back-channels are ignored) is cut out,
    and leftovers shorter than min_len are dropped so a quick interjection isn't skipped.
    """
    mine = sorted((s, e) for s, e, spk in segments if labels.get(spk, (None,))[0] == target)
    others = sorted((s, e) for s, e, spk in segments
                    if labels.get(spk, (None,))[0] != target and e - s >= other_min)

    merged = []
    for s, e in mine:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    result = []
    for s, e in merged:
        pieces = [(s, e)]
        for os_, oe in others:
            if oe <= s or os_ >= e:
                continue
            nxt = []
            for ps, pe in pieces:
                if oe <= ps or os_ >= pe:
                    nxt.append((ps, pe))
                    continue
                if os_ > ps:
                    nxt.append((ps, os_))
                if oe < pe:
                    nxt.append((oe, pe))
            pieces = nxt
        result.extend(p for p in pieces if p[1] - p[0] >= min_len)
    return [[round(s, 2), round(e, 2)] for s, e in result]
