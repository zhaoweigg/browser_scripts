"""Find where people talk by loudness alone, for videos that have no transcript.

Produces (start, end) "lines" of speech, the same shape as caption lines, so the fast
transcript method can run without captions. Audio can be fed as it arrives.
"""

import numpy as np

from .audio import SAMPLE_RATE

FRAME = 0.05          # seconds per loudness measurement
FLOOR_DB = -55.0      # quieter than this is never speech
BELOW_SPEECH = 20.0   # this many dB under the usual speech level counts as a pause
HANGOVER = 0.3        # a pause shorter than this doesn't end a line
LINE = 2.0            # longest line; longer stretches of speech are cut into several
MIN_LINE = 0.2        # shorter blips (clicks) are ignored
WARMUP = 5.0          # seconds heard before the speech level is trusted


class SpeechSlicer:
    def __init__(self):
        self._n = int(FRAME * SAMPLE_RATE)
        self._tail = np.zeros(0, dtype=np.float32)  # samples that don't fill a frame yet
        self._hist = np.zeros(140, dtype=np.int64)  # frame loudness, 1 dB bins from -110 dB
        self._held = []      # loudness of frames waiting for the warm-up to finish
        self._seen = 0       # frames measured
        self._i = 0          # frames decided
        self._level = None   # usual speech level (90th percentile of non-silent frames)
        self._start = None   # first frame of the line being built
        self._voiced = None  # last frame that had speech in it

    def _update_level(self):
        total = self._hist.sum()
        if total:
            self._level = int(np.searchsorted(np.cumsum(self._hist), 0.9 * total)) - 110.0

    def _decide(self, db, out):
        i = self._i
        self._i += 1
        if db > max(FLOOR_DB, self._level - BELOW_SPEECH):
            if self._start is None:
                self._start = i
            self._voiced = i
            if (i + 1 - self._start) * FRAME >= LINE:
                out.append((self._start * FRAME, (i + 1) * FRAME))
                self._start = None
        elif self._start is not None and (i - self._voiced) * FRAME >= HANGOVER:
            self._close(out)

    def _close(self, out):
        if self._start is not None and (self._voiced + 1 - self._start) * FRAME >= MIN_LINE:
            out.append((self._start * FRAME, (self._voiced + 1) * FRAME))
        self._start = None

    def feed(self, samples: np.ndarray):
        """Add audio; returns the speech lines completed by it, as [(start_s, end_s), ...]."""
        data = np.concatenate([self._tail, samples]) if len(self._tail) else samples
        whole = len(data) // self._n
        self._tail = data[whole * self._n:].copy()
        if not whole:
            return []
        frames = data[:whole * self._n].reshape(whole, self._n)
        dbs = 20 * np.log10(np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) + 1e-9)
        out = []
        for db in dbs:
            if db > FLOOR_DB:
                self._hist[min(139, int(db + 110))] += 1
            self._seen += 1
            if self._seen % 20 == 0 and self._seen * FRAME >= WARMUP:
                self._update_level()
            if self._level is None:
                self._held.append(db)
                continue
            for held in self._held:
                self._decide(held, out)
            self._held = []
            self._decide(db, out)
        return out

    def flush(self):
        """Call once at the end of the audio; returns the last lines."""
        out = []
        if self._level is None:
            self._update_level()
        if self._level is not None:
            for held in self._held:
                self._decide(held, out)
        self._held = []
        self._close(out)
        return out


def speech_lines(samples: np.ndarray):
    """All speech lines of a complete recording."""
    s = SpeechSlicer()
    return s.feed(samples) + s.flush()
