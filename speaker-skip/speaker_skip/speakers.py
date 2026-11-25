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


# ---------- transcript-based labelling ----------
# Each caption line is matched against the enrolled voices directly, instead of
# diarizing the whole recording first.

MIN_WINDOW = 2.0      # seconds of audio used to judge a line (shorter lines are widened)
MAX_WINDOW = 6.0
UNKNOWN_SCORE = 0.45  # a line whose best match is below this stays unlabelled
SWITCH_PENALTY = 0.3  # how much evidence it takes to change speaker between lines
ADAPT_MARGIN = 0.15   # lines this clearly one person's are used to refine their voiceprint
ADAPT_MIN_SIM = 0.55
ADAPT_MIN_SECONDS = 30.0
ADAPT_MAX_DRIFT = 0.8  # a refined voiceprint must stay this similar to the enrolled one
GUEST_BELOW = 0.45     # lines matching every enrolled voice worse than this may be a guest
GUEST_COHERENCE = 0.5  # how alike lines must sound to be grouped as one guest
GUEST_MIN_SECONDS = 45.0
GUEST_KNOWN_SHARE = 0.5  # a found voice is a guest only if less than this share of its lines match enrolled people
GUEST_MAX = 4


def cue_window(s, e, total):
    """The stretch of audio used to judge the caption line (s, e)."""
    half = min(max(e - s, MIN_WINDOW), MAX_WINDOW) / 2
    mid = (s + e) / 2
    return max(0.0, mid - half), min(total, mid + half)


def embed_window(samples: np.ndarray, a, b) -> np.ndarray:
    """Voice embedding of samples[a:b] (seconds); zeros when there is no usable audio."""
    chunk = samples[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)]
    if len(chunk) >= SAMPLE_RATE // 2 and float(np.sqrt(np.mean(chunk ** 2))) > 1e-3:
        return embed(chunk)
    return np.zeros(extractor().dim, dtype=np.float32)


def cue_embeddings(samples: np.ndarray, cues, progress=None) -> np.ndarray:
    """One voice embedding per caption line; rows are zero where there is no usable audio."""
    total = len(samples) / SAMPLE_RATE
    out = np.zeros((len(cues), extractor().dim), dtype=np.float32)
    for i, (s, e) in enumerate(cues):
        out[i] = embed_window(samples, *cue_window(s, e, total))
        if progress and i % 20 == 0:
            progress(i / len(cues))
    return out


def _viterbi(scores: np.ndarray, weights: np.ndarray, penalty: float) -> np.ndarray:
    """Best label per line: maximise weighted score minus a penalty for every change."""
    n, k = scores.shape
    best = np.zeros((n, k), dtype=np.float64)
    back = np.zeros((n, k), dtype=np.int32)
    best[0] = scores[0] * weights[0]
    for t in range(1, n):
        stay = best[t - 1]
        j = int(np.argmax(stay))
        switch = stay[j] - penalty
        back[t] = np.where(stay >= switch, np.arange(k), j)
        best[t] = np.maximum(stay, switch) + scores[t] * weights[t]
    path = np.zeros(n, dtype=np.int32)
    path[-1] = int(np.argmax(best[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def find_guests(embs, weights, candidates, max_guests=GUEST_MAX):
    """Voiceprints for people in this recording who match no enrolled voice.

    `candidates` marks the lines that match nobody well. Lines that sound alike are
    grouped; a group with enough speech becomes a guest voice, so that person competes
    for their own lines instead of being mistaken for the nearest enrolled voice."""
    guests = []
    left = candidates.copy()
    for _ in range(max_guests):
        if weights[left].sum() < GUEST_MIN_SECONDS:
            break
        # Seed with the line that has the most similar neighbours, then tighten.
        sub = np.flatnonzero(left)
        pick = sub if len(sub) <= 400 else sub[np.linspace(0, len(sub) - 1, 400).astype(int)]
        density = ((embs[pick] @ embs[sub].T) > GUEST_COHERENCE).astype(np.float32) @ weights[sub]
        centre = embs[pick[int(np.argmax(density))]]
        for _ in range(4):
            members = left & (embs @ centre >= GUEST_COHERENCE)
            if not members.any():
                break
            centre = _unit((embs[members] * weights[members, None]).sum(axis=0))
        members = left & (embs @ centre >= GUEST_COHERENCE)
        if weights[members].sum() < GUEST_MIN_SECONDS:
            break
        guests.append(centre)
        left &= ~members
    return guests


def label_cues(embs: np.ndarray, cues, voiceprints, adapt=True):
    """Return (labels, sims): labels[i] is an enrolled voice name, or None (guest/unknown),
    for caption line i; sims[i] are the line's similarities to the enrolled voices."""
    names = list(voiceprints)
    enrolled = np.stack([voiceprints[n] for n in names])
    dur = np.array([e - s for s, e in cues])
    weights = np.clip(dur, 0.5, 3.0)
    valid = np.linalg.norm(embs, axis=1) > 0

    # Guests: voices in this recording that match no enrolled voice. Each gets its own
    # voiceprint so the guest competes for their own lines instead of being mistaken for
    # the nearest enrolled voice. A found voice whose lines mostly do match enrolled
    # people is just leftovers (crosstalk, mumbling) and is ignored.
    best = (embs @ enrolled.T).max(axis=1)
    prints = list(enrolled)
    for g in find_guests(embs, weights, valid & (best < GUEST_BELOW)):
        alike = valid & (embs @ g >= GUEST_COHERENCE)
        if dur[alike & (best >= GUEST_BELOW)].sum() < GUEST_KNOWN_SHARE * dur[alike].sum():
            prints.append(g)
    prints = np.stack(prints)

    def run(prints):
        sims = embs @ prints.T
        scores = np.concatenate([sims, np.full((len(cues), 1), UNKNOWN_SCORE)], axis=1)
        scores[~valid, :-1] = 0.0
        return sims, _viterbi(scores, weights, SWITCH_PENALTY)

    sims, path = run(prints)
    if adapt:
        # Voices sound a bit different in every recording. Refine each enrolled voiceprint
        # with the lines that are clearly that person's here, then label again. Only strong
        # matches count, and a refined print that drifted away from the original is dropped.
        ranked = np.sort(sims, axis=1)
        margin = ranked[:, -1] - (ranked[:, -2] if sims.shape[1] > 1 else 0.0)
        refined = prints.copy()
        for k in range(len(names)):
            sure = (valid & (path == k) & (np.argmax(sims, axis=1) == k)
                    & (sims[:, k] >= ADAPT_MIN_SIM) & (margin >= ADAPT_MARGIN))
            if dur[sure].sum() >= ADAPT_MIN_SECONDS:
                mean = _unit((embs[sure] * weights[sure, None]).sum(axis=0))
                candidate = _unit(enrolled[k] + mean)
                if float(candidate @ enrolled[k]) >= ADAPT_MAX_DRIFT:
                    refined[k] = candidate
        sims, path = run(refined)

    labels = [names[k] if k < len(names) else None for k in path]  # guests and unknown: None
    return labels, sims[:, :len(names)]


SHORT_RANGE = 6.0        # skips shorter than this must be a clear match
SHORT_MIN_SIM = 0.58
SHORT_MIN_MARGIN = 0.10
ANY_MIN_SIM = 0.50       # no skip of any length on a match weaker than this


def confident_ranges(ranges, cues, sims, k):
    """Drop short skip ranges whose voice match is weak or close to another person's.

    Cutting someone else off by mistake is worse than letting a short remark through."""
    starts = np.array([c[0] for c in cues])
    ends = np.array([c[1] for c in cues])
    keep = []
    for s, e in ranges:
        w = np.clip(np.minimum(ends, e) - np.maximum(starts, s), 0, None)
        if w.sum() <= 0:
            continue
        m = (sims * w[:, None]).sum(axis=0) / w.sum()
        if m[k] < ANY_MIN_SIM:
            continue
        if e - s < SHORT_RANGE:
            others = np.delete(m, k)
            if m[k] < SHORT_MIN_SIM or (len(others) and m[k] - others.max() < SHORT_MIN_MARGIN):
                continue
        keep.append([s, e])
    return keep
