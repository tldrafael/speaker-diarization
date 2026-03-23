import argparse
import os
import sys
from bisect import bisect_left, bisect_right

import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


def srt_timestamp(seconds: float) -> str:
    h, rem = divmod(int(round(seconds * 1000)), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_speaker_index(diarization):
    """Pre-sort diarization turns so speaker lookup is O(log n) per segment."""
    turns: list[tuple[float, float, str]] = []

    diar = getattr(diarization, "itertracks", None)
    if diar is None:
        return [], []

    for turn, _, speaker in diar(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))

    turns.sort(key=lambda t: t[0])
    starts = [t[0] for t in turns]
    return turns, starts


def pick_speaker(turns, starts, seg_start: float, seg_end: float) -> str:
    if not turns:
        return "UNKNOWN"

    # Only consider turns that could overlap [seg_start, seg_end):
    #   turn.start < seg_end  AND  turn.end > seg_start
    right = bisect_left(starts, seg_end)
    scores: dict[str, float] = {}
    for i in range(right):
        t_start, t_end, speaker = turns[i]
        if t_end <= seg_start:
            continue
        ov = min(seg_end, t_end) - max(seg_start, t_start)
        if ov > 0:
            scores[speaker] = scores.get(speaker, 0.0) + ov

    if not scores:
        return "UNKNOWN"
    return max(scores, key=scores.__getitem__)


def parse_args():
    p = argparse.ArgumentParser(description="Speaker diarization + transcription → SRT")
    p.add_argument("audio", help="Path to the audio file")
    p.add_argument("-l", "--lang", default="pt", help="Language code (default: pt)")
    p.add_argument(
        "-m", "--model", default="large-v3",
        help="Whisper model size (default: large-v3). Options: tiny, base, small, medium, large-v3",
    )
    p.add_argument(
        "--compute-type", default=None,
        help="CTranslate2 compute type (default: float16 on CUDA, int8 on CPU)",
    )
    p.add_argument(
        "--num-speakers", type=int, default=None,
        help="Hint for the diarization pipeline (speeds it up if known)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1) Diarize ----------------------------------------------------------
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(torch.device(device))

    diar_kwargs: dict = {}
    if args.num_speakers is not None:
        diar_kwargs["num_speakers"] = args.num_speakers

    with ProgressHook() as hook:
        diarization = pipeline(args.audio, hook=hook, **diar_kwargs)

    turns, starts = build_speaker_index(diarization)

    # Free diarization model before loading Whisper to reduce peak memory
    del pipeline
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- 2) Transcribe -------------------------------------------------------
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")
    asr = WhisperModel(args.model, device=device, compute_type=compute_type)

    segments, _info = asr.transcribe(
        args.audio,
        language=args.lang,
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,
    )

    # --- 3) Build SRT --------------------------------------------------------
    srt_lines: list[str] = []
    idx = 1
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        speaker = pick_speaker(turns, starts, seg.start, seg.end)
        srt_lines.append(str(idx))
        srt_lines.append(f"{srt_timestamp(seg.start)} --> {srt_timestamp(seg.end)}")
        srt_lines.append(f"{speaker}: {text}")
        srt_lines.append("")
        idx += 1

    out_path = os.path.splitext(args.audio)[0] + ".speaker.srt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    print(f"Wrote {idx - 1} subtitles → {out_path}")


if __name__ == "__main__":
    main()
