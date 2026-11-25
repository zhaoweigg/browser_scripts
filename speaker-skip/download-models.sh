#!/bin/bash
# Fetch the ONNX models used for speaker analysis (about 35 MB) into ./models.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models && cd models
B=https://github.com/k2-fsa/sherpa-onnx/releases/download
[[ -f 3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx ]] ||
  curl -fL -O "$B/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
[[ -f sherpa-onnx-pyannote-segmentation-3-0/model.onnx ]] ||
  curl -fL "$B/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" | tar xj
ls -la
