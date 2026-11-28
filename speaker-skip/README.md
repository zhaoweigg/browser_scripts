# speaker-skip

Skip every part of a YouTube discussion where one particular person talks.

Two pieces:

- **`../speaker-skip.user.js`**: Tampermonkey script. Adds a **Skip 伍雷** button to the
  player on configured channels (三個水槍手). Turn it on and it asks the helper to analyze
  the episode, shows progress on the button, marks the skipped parts in yellow on the
  progress bar, then jumps past them while you watch. The on/off state is remembered, so
  later episodes get analyzed as soon as you open them.
- **This folder**: a small helper on `127.0.0.1:8787` that does the analysis, because a web
  page can't get a YouTube video's full audio on its own. For each video it downloads the
  audio track (yt-dlp) and works out where each enrolled person talks. Results are cached in
  `data/results/`, so each video is analyzed once. The audio is deleted afterwards.

How the analysis works:

- **With a transcript (the normal case, under a minute for a 75-minute episode):** the
  video's captions say when each line is spoken. Each line's audio is matched against the
  enrolled voiceprints (3D-Speaker CAM++ embeddings via sherpa-onnx), the labels are smoothed
  across neighbouring lines, and each voiceprint is refined with the lines that are clearly
  that person's in this episode. Only the caption timing is used, never the text.
  - **Streaming:** this runs while the audio is still downloading. The first minutes are
    ready about 8 seconds after you press the button, and the helper hands the browser the
    skip ranges found so far every 2 seconds, so skipping works before the analysis is done.
  - **Guests:** a voice that matches none of the enrolled people (an interview guest) gets
    its own voiceprint for that episode, so it is never mistaken for the closest-sounding
    host. Guests and unclear lines are never skipped.
- **Without a transcript (about 4 min):** full speaker diarization (pyannote segmentation +
  clustering), then each cluster is matched against the voiceprints. Force this with
  `analyze VIDEO_ID --full`.

The helper runs as a process named **SpeakerSkip** and shows an icon in the macOS menu bar:
a skip-forward symbol while idle, a waveform with a percentage while analyzing. Its menu
shows what it's doing, when the browser script last checked in, the enrolled voices, the log,
and **Quit**. After Quit it stays stopped until the next login or `./install-helper.sh`.

Chrome needs **Allow User Scripts** switched on for Tampermonkey (chrome://extensions →
Tampermonkey → Details), or no userscript runs at all.

## Setup

```bash
cd speaker-skip
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "yt-dlp[default]" sherpa-onnx av numpy deno pyobjc-framework-Cocoa
./download-models.sh
./install-helper.sh          # starts now and on login; logs to data/helper.log
```

Enroll voices from videos where that person is the main speaker (their own channel):

```bash
.venv/bin/python -m speaker_skip enroll 伍雷 A-Mf1oaELIw auIe60tibgY
.venv/bin/python -m speaker_skip enroll 李厚辰 mv0CPDtNo5A abRAHhY6ALM
.venv/bin/python -m speaker_skip enroll 贾葭 txfH3IVjIwY
```

Try one video from the terminal:

```bash
.venv/bin/python -m speaker_skip analyze _YNaN-JAnc8
```

## Changing who gets skipped

Edit `CHANNELS` at the top of `speaker-skip.user.js`: the key is the channel's @handle, `skip` is
the enrolled voice name. The helper computes skip ranges for every enrolled voice.

## Tuning

In `speaker_skip/speakers.py`:

- `skip_ranges(...)`: `min_len` (shortest stretch worth skipping, default 2 s), `other_min`
  (how long someone else must talk to cut a skip short; shorter "嗯/对" back-channels are
  ignored), and `PAUSE_MAX` (a pause up to this long between two of the person's stretches
  is skipped with them, as long as nobody else talks in it; default 8 s).
- Transcript method: `SWITCH_PENALTY` (higher = fewer speaker changes between lines),
  `UNKNOWN_SCORE` (lines matching nobody this well are left alone), `GUEST_*` (when an
  unmatched voice counts as a guest), `ADAPT_*` (which lines may refine a voiceprint, and
  how far it may drift), `ANY_MIN_SIM` (no skip on a weaker match than this), and
  `SHORT_MIN_SIM` / `SHORT_MIN_MARGIN` (how clear a match must be before a skip shorter
  than `SHORT_RANGE` seconds is allowed; raise them to skip less during crosstalk).
- Full method: `MATCH_THRESHOLD` (how similar a voice cluster must be to a voiceprint).

After changing analysis code, bump `RESULT_VERSION` in `pipeline.py` so cached results are redone.

Stop the helper: `./install-helper.sh uninstall`.
