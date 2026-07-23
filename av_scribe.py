#!/usr/bin/env python3
"""av-scribe: 動画/音声 → 話者ラベル付き文字起こし（完全ローカル）

ASR: mlx-whisper (large-v3-turbo, Metal)
話者分離: sherpa-onnx (pyannote segmentation 3.0 + 3D-Speaker eres2net)
入力は外部APIに一切送らない。

usage:
  av-scribe input.mp4 [-o out.md] [--speakers N] [--lang ja] [--model REPO]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
SEG_MODEL = os.path.join(BASE, "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
EMB_MODEL = os.path.join(BASE, "models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")
DEFAULT_ASR = "mlx-community/whisper-large-v3-turbo"


def extract_wav(src: str, dst: str) -> None:
    """ffmpegで16kHzモノラルwavに変換（動画でも音声でも受ける）"""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-vn", "-ar", "16000", "-ac", "1", dst],
        check=True,
    )


def diarize(wav: str, num_speakers: int):
    import sherpa_onnx
    import soundfile as sf

    if num_speakers > 0:
        clustering = sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers)
    else:
        # 話者数不明時は距離しきい値でクラスタ数を自動決定
        clustering = sherpa_onnx.FastClusteringConfig(threshold=1.3)
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL)),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB_MODEL),
        clustering=clustering,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == sd.sample_rate, f"sample rate mismatch: {sr} != {sd.sample_rate}"
    return sd.process(audio).sort_by_start_time()


def transcribe(wav: str, lang: str, model: str):
    import mlx_whisper
    r = mlx_whisper.transcribe(wav, path_or_hf_repo=model, language=lang, condition_on_previous_text=False)
    return r["segments"]


def assign_speaker(seg, turns) -> str:
    """ASRセグメントに、時間の重なりが最大の話者を割り当てる"""
    best, best_ov = None, 0.0
    for t in turns:
        ov = min(seg["end"], t.end) - max(seg["start"], t.start)
        if ov > best_ov:
            best, best_ov = t.speaker, ov
    return f"speaker_{best:02d}" if best is not None else "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="出力先 (.md / .jsonl)。省略時はstdout")
    ap.add_argument("--speakers", type=int, default=0, help="話者数（0=自動推定）")
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--model", default=DEFAULT_ASR)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        extract_wav(args.input, wav)
        print("== 文字起こし中 (mlx-whisper) ==", file=sys.stderr)
        segments = transcribe(wav, args.lang, args.model)
        print("== 話者分離中 (sherpa-onnx) ==", file=sys.stderr)
        turns = diarize(wav, args.speakers)

    rows = []
    for s in segments:
        text = s["text"].strip()
        if not text:
            continue
        rows.append({
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "speaker": assign_speaker(s, turns),
            "text": text,
        })

    if args.output and args.output.endswith(".jsonl"):
        out = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    else:
        lines, prev = [], None
        for r in rows:
            head = "" if r["speaker"] == prev else f"\n**{r['speaker']}** "
            lines.append(f"{head}[{r['start']:.0f}s] {r['text']}")
            prev = r["speaker"]
        out = "\n".join(lines).strip() + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
        print(f"wrote {args.output} ({len(rows)} segments)", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
