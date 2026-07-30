#!/usr/bin/env python3
"""av-scribe: 動画/音声 → 話者ラベル付き文字起こし（完全ローカル）

ASR: mlx-whisper (large-v3-turbo, Metal)
話者分離: sherpa-onnx (pyannote segmentation 3.0 + 3D-Speaker eres2net)
入力は外部APIに一切送らない。

--engine moss: MOSS-Transcribe-Diarize 0.9B(CPU) で文字起こし+話者分離を単一パスで行う。
話者帰属の精度は上がるが漢字選択は劣り、CPU推論のみで低速（要 setup-moss.sh）。

usage:
  av-scribe input.mp4 [-o out.md] [--speakers N] [--lang ja] [--model REPO] [--engine {mlx,moss}]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

MOSS_MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
MOSS_MAX_DURATION_SEC = 90 * 60  # モデル側の1パス上限（超過時は警告のみ、処理は続行）

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


def moss_speaker_label(raw: str) -> str:
    """MOSSの話者ラベル("S01"等)をav-scribe既存形式("speaker_00")に変換。

    "S01" → 0番目 → "speaker_00" のように、ラベル中の数字部分を1引いて0始まりにする。
    数字が読めない/想定外の形式なら生のラベルをそのまま使う（既存ラベルとの衝突を避けるためprefixを付ける）。
    """
    m = re.search(r"(\d+)", raw)
    if m is None:
        return f"speaker_{raw}"
    return f"speaker_{int(m.group(1)) - 1:02d}"


def get_audio_duration_sec(wav: str) -> float:
    """wavの長さ(秒)をsoundfileのヘッダ情報だけから求める(全読み込み不要)"""
    import soundfile as sf

    info = sf.info(wav)
    return info.frames / info.samplerate


def transcribe_diarize_moss(wav: str):
    """MOSS-Transcribe-Diarize 0.9B で文字起こし+話者分離を単一パスで行う。

    CPU推論のみ（MLX/MPS非対応）。90分/パスの上限があるため呼び出し側で警告を出す。
    依存はsetup-moss.shでオプションインストールする前提（本体のセットアップには含めない）。
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        from moss_transcribe_diarize import parse_transcript
        from moss_transcribe_diarize.inference_utils import (
            build_transcription_messages,
            generate_transcription,
            resolve_device,
        )
    except ImportError:
        sys.exit(
            "エラー: --engine moss の依存が見つかりません。"
            " ./setup-moss.sh を実行してから再度お試しください。"
        )

    device = resolve_device("auto")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = (
        AutoModelForCausalLM.from_pretrained(MOSS_MODEL_ID, trust_remote_code=True, dtype="auto")
        .to(dtype=dtype)
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(MOSS_MODEL_ID, trust_remote_code=True)

    duration_sec = get_audio_duration_sec(wav)
    # 日本語の実測で約7トークン/秒(5分≒2000トークン)。相槌・言い直し等の余裕を見て12トークン/秒で概算。
    # 下限4096(短尺での安全マージン)、上限32768(過大な生成時間・メモリを避けるキャップ)。
    max_new_tokens = min(32768, max(4096, int(duration_sec * 12)))

    messages = build_transcription_messages(wav)
    result = generate_transcription(
        model, processor, messages, max_new_tokens=max_new_tokens, do_sample=False, device=device, dtype=dtype,
    )

    rows = []
    for seg in parse_transcript(result["text"]):
        text = seg.text.strip()
        if not text:
            continue
        rows.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "speaker": moss_speaker_label(seg.speaker),
            "text": text,
        })

    if rows and duration_sec > 0:
        last_end = rows[-1]["end"]
        if last_end < duration_sec * 0.8:
            print(
                f"警告: 出力の最終セグメントが{last_end:.0f}秒で終わっていますが、"
                f"音声本体は{duration_sec:.0f}秒あります。max_new_tokens上限に達して"
                " 出力が途中で切れている可能性があります。",
                file=sys.stderr,
            )
    return rows


def warn_if_too_long(wav: str) -> None:
    """--engine moss選択時、90分上限を超えていたら警告する（処理は止めない）"""
    duration_sec = get_audio_duration_sec(wav)
    if duration_sec > MOSS_MAX_DURATION_SEC:
        print(
            f"警告: 音声が90分を超えています({duration_sec / 60:.0f}分)。"
            " MOSSは90分/パスが上限のため、末尾が欠落する可能性があります。",
            file=sys.stderr,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="出力先 (.md / .jsonl)。省略時はstdout")
    ap.add_argument("--speakers", type=int, default=0, help="話者数（0=自動推定）")
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--model", default=DEFAULT_ASR)
    ap.add_argument("--engine", choices=["mlx", "moss"], default="mlx",
                     help="mlx=mlx-whisper+sherpa-onnx(既定), moss=MOSS-Transcribe-Diarize単一パス(話者帰属重視・低速・CPU限定)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        extract_wav(args.input, wav)

        if args.engine == "moss":
            warn_if_too_long(wav)
            print("== 文字起こし+話者分離中 (MOSS-Transcribe-Diarize, CPU) ==", file=sys.stderr)
            rows = transcribe_diarize_moss(wav)
        else:
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
