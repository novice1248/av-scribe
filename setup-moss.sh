#!/bin/bash
# av-scribe --engine moss のセットアップ(オプション)。
# 話者帰属の精度を重視するときだけ実行する追加セットアップ。
# 既存の.venv(setup.sh作成)にtorch/transformers等を追加インストールする。
# 依存サイズの目安: torch等で約2GB + MOSS-Transcribe-Diarize本体のモデル重み約1.8GB(初回実行時にHFから自動DL)。
# CPU推論のみ(MLX/MPS非対応)で、mlxエンジンより低速(実測: 5分素材で約65秒 vs mlxの約137秒。
# ただし90分素材では実時間の約0.46倍かかる想定)。
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "エラー: .venvが見つかりません。先に ./setup.sh を実行してください。" >&2
  exit 1
fi

echo "== MOSS-Transcribe-Diarize用の依存を追加インストール =="
uv pip install torch transformers accelerate soundfile librosa
uv pip install "git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git"

echo "== 完了 =="
echo "使い方: .venv/bin/python av_scribe.py input.mp4 -o out.md --engine moss"
echo "(初回実行時にモデル本体(約1.8GB, OpenMOSS-Team/MOSS-Transcribe-Diarize)がHugging Faceから自動ダウンロードされます)"
