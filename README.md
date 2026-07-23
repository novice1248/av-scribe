# av-scribe

動画/音声 → 話者ラベル付き文字起こしを**完全ローカル**で行う単一ファイルの CLI（macOS / Apple Silicon）。

音声データを外部 API に一切送りません。会議録・インタビュー・LT の録音など、外に出したくない音声のための道具です。

- **ASR**: [mlx-whisper](https://github.com/ml-explore/mlx-examples)（`whisper-large-v3-turbo`、Metal で動く）
- **話者分離**: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（pyannote segmentation 3.0 + 3D-Speaker eres2net）— Hugging Face トークン不要
- 実装は [av_scribe.py](av_scribe.py) の約 120 行だけ。ASR と話者分離を独立に走らせ、時間の重なりが最大の話者を各セグメントに割り当てる

## セットアップ

```bash
brew install uv ffmpeg
./setup.sh   # venv作成 + 依存 + 話者分離モデル(約45MB)のダウンロード
```

Whisper モデルは初回実行時に Hugging Face から自動ダウンロードされます（以後はローカルキャッシュ。**モデル取得後は音声処理に通信は発生しません**）。

## 使い方

```bash
.venv/bin/python av_scribe.py 会議録.mp4 -o out.md
.venv/bin/python av_scribe.py interview.wav --speakers 2   # 話者数が分かっているなら指定(精度が上がる)
.venv/bin/python av_scribe.py talk.mov -o out.jsonl        # 拡張子.jsonlで構造化出力
```

出力例（Markdown）:

```
**speaker_00** [0s] 今日はよろしくお願いします。
[3s] まず自己紹介から始めましょうか。

**speaker_01** [8s] はい、〇〇です。普段はバックエンドを書いています。
```

`~/.local/bin` などに置くラッパーを作ると `av-scribe input.mp4` で呼べます:

```bash
#!/bin/zsh
exec /path/to/av-scribe/.venv/bin/python /path/to/av-scribe/av_scribe.py "$@"
```

## Limitations（正直な限界）

- **macOS / Apple Silicon 専用**（mlx が Metal 前提）。Linux/Windows なら ASR を faster-whisper 等に差し替える改造が必要
- **話者埋め込みモデルは中国語話者で学習された eres2net**（zh-cn）。日本語話者でも実用になっているが、声質が近い話者の分離精度は保証しない。精度が足りなければ WeSpeaker / NeMo 系の ONNX に `EMB_MODEL` を差し替える
- 話者数の自動推定（`--speakers 0`）はクラスタリングしきい値頼みで、外すことがある。**話者数が分かっているなら明示指定を推奨**
- ASR と話者分離は独立に走るため、発話の境界とセグメントの境界がずれると短い相槌が隣の話者に吸われることがある

## モデルのライセンス

- pyannote segmentation 3.0: MIT（[k2-fsa による ONNX 変換版](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models)を setup.sh がダウンロード）
- 3D-Speaker eres2net: Apache-2.0

## License

MIT
