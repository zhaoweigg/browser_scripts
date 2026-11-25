#!/bin/bash
# Install (or reinstall) the speaker-skip helper as a macOS login agent, so it starts
# on login and restarts if it crashes. It shows an icon in the menu bar while running.
# Run again after moving this folder, or to start it again after quitting from the menu.
# Remove with: ./install-helper.sh uninstall
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="com.github.zhaoweigg.speaker-skip"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [[ "${1:-}" == "uninstall" ]]; then
  rm -f "$PLIST"
  echo "removed $LABEL"
  exit 0
fi

mkdir -p "$DIR/data" "$HOME/Library/LaunchAgents"
# Run Python under its own name so the process shows up as "SpeakerSkip", not "python".
ln -sf python "$DIR/.venv/bin/SpeakerSkip"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$DIR/.venv/bin/SpeakerSkip</string>
    <string>-m</string>
    <string>speaker_skip</string>
    <string>serve</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$DIR/.venv/bin:/usr/bin:/bin</string>
    <key>PYTHONUNBUFFERED</key><string>1</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <!-- Restart after a crash, but stay stopped after Quit from the menu bar (exit 0). -->
  <key>KeepAlive</key>
  <dict><key>SuccessfulExit</key><false/></dict>
  <key>StandardOutPath</key><string>$DIR/data/helper.log</string>
  <key>StandardErrorPath</key><string>$DIR/data/helper.log</string>
</dict>
</plist>
EOF
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $LABEL (log: $DIR/data/helper.log)"
