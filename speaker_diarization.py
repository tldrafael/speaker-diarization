import sys
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import os
import math
import whisper


AUDIO_FILE = sys.argv[1]
LANG = "pt"
MODEL = "large"  # try: small / medium / large
HF_TOKEN = os.environ.get("HF_TOKEN")


def srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000.0))
    h = ms // 3_600_000
    ms -= h * 3_600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1_000
    ms -= s * 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def pick_speaker(output, seg_start: float, seg_end: float) -> str:
    # Prefer exclusive diarization when available (better for aligning to ASR segments)
    diar = getattr(output, "exclusive_speaker_diarization", None) or getattr(output, "speaker_diarization", None)

    if diar is None:
        return "UNKNOWN"

    scores = {}
    # diar yields (turn, speaker)
    for turn, speaker in diar:
        ov = overlap(seg_start, seg_end, float(turn.start), float(turn.end))
        if ov > 0:
            scores[speaker] = scores.get(speaker, 0.0) + ov

    if not scores:
        return "UNKNOWN"
    return max(scores.items(), key=lambda kv: kv[1])[0]


def main():
    # 1) Diarize
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=HF_TOKEN,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    with ProgressHook() as hook:
        diarization = pipeline(AUDIO_FILE, hook=hook)  # runs locally

    # 2) Transcribe (with timestamps per segment)
    asr_model = whisper.load_model(MODEL, device=device)
    result = asr_model.transcribe(AUDIO_FILE, language=LANG, task="transcribe")

    # 3) Build SRT lines with speaker labels
    srt_lines = []
    idx = 1
    for seg in result.get("segments", []):
        start = float(seg["start"])
        end = float(seg["end"])
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        speaker = pick_speaker(diarization, start, end)

        srt_lines.append(str(idx))
        srt_lines.append(f"{srt_timestamp(start)} --> {srt_timestamp(end)}")
        srt_lines.append(f"{speaker}: {text}")
        srt_lines.append("")  # blank line
        idx += 1

    out_path = os.path.splitext(AUDIO_FILE)[0] + ".speaker.pt.srt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()

