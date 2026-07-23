#!/bin/bash
# av-scribeのセットアップ: venv作成 + 依存インストール + 話者分離モデルのダウンロード。
# 前提: macOS (Apple Silicon), uv, ffmpeg
#   brew install uv ffmpeg
set -euo pipefail
cd "$(dirname "$0")"

echo "== venv作成 + 依存インストール =="
uv venv --python 3.12
uv pip install mlx-whisper sherpa-onnx soundfile

echo "== 話者分離モデルのダウンロード (計 約45MB) =="
mkdir -p models
# セグメンテーション: pyannote segmentation 3.0 (ONNX変換版, k2-fsa配布)
if [ ! -f models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx ]; then
  curl -SL -o models/seg.tar.bz2 \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  tar -xjf models/seg.tar.bz2 -C models/
  rm models/seg.tar.bz2
fi
# 話者埋め込み: 3D-Speaker eres2net
# (リリースタグの"recongition"はk2-fsa側のタイポのまま正)
if [ ! -f models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx ]; then
  curl -SL -o models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
fi

echo "== 完了 =="
echo "使い方: .venv/bin/python av_scribe.py input.mp4 -o out.md"
echo "(初回実行時にWhisperモデル(mlx-community/whisper-large-v3-turbo)がHugging Faceから自動ダウンロードされます)"
