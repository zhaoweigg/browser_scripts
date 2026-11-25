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
  audio (yt-dlp), finds who speaks when (pyannote segmentation + 3D-Speaker CAM++ voice
  embeddings, via sherpa-onnx), and matches each voice against enrolled voiceprints. Results
  are cached in `data/results/`, so each video is analyzed once. The audio is deleted afterwards.

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

- `MATCH_THRESHOLD`: how similar a voice must be to a voiceprint to count as that person.
- `skip_ranges(...)`: `min_len` (shortest stretch worth skipping, default 2 s), `merge_gap`
  (pauses bridged inside one turn), `other_min` (how long someone else must talk to cut a
  skip short; shorter "嗯/对" back-channels are ignored).

After changing analysis code, bump `RESULT_VERSION` in `pipeline.py` so cached results are redone.

Stop the helper: `./install-helper.sh uninstall`.
