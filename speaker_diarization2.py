import argparse
import os
import subprocess
from bisect import bisect_left

import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

MAX_CUE_SECONDS = 7.0
MAX_GAP_SECONDS = 0.6


def _probe_audio(path: str) -> tuple[str, str] | None:
    """Return (sample_rate, channels) for the first audio stream, if ffprobe can read it."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "csv=p=0",
            path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return None
    parts = probe.stdout.strip().split(",")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def ensure_wav(path: str) -> str:
    """Return a 16 kHz mono WAV. Community-1 expects that input."""
    if path.lower().endswith(".wav"):
        probed = _probe_audio(path)
        if probed == ("16000", "1"):
            return path

    wav_path = os.path.splitext(path)[0] + ".16k.wav"
    if os.path.abspath(wav_path) == os.path.abspath(path):
        wav_path = path + ".16k.wav"

    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
        capture_output=True,
    )
    return wav_path


def srt_timestamp(seconds: float) -> str:
    h, rem = divmod(int(round(seconds * 1000)), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_speaker_index(diarization):
    """Pre-sort exclusive diarization turns so speaker lookup is O(log n) per word."""
    exclusive = getattr(diarization, "exclusive_speaker_diarization", None)
    if exclusive is None:
        return [], []

    turns: list[tuple[float, float, str]] = []
    for turn, speaker in exclusive:
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


def iter_words(segments):
    """Yield (start, end, text) at word level, falling back to the segment when needed."""
    for seg in segments:
        words = getattr(seg, "words", None) or []
        emitted = False
        for word in words:
            text = (word.word or "").strip()
            if not text or word.start is None or word.end is None:
                continue
            emitted = True
            yield float(word.start), float(word.end), text
        if emitted:
            continue
        text = (seg.text or "").strip()
        if text:
            yield float(seg.start), float(seg.end), text


def build_cues(segments, turns, starts) -> list[dict]:
    """Merge same-speaker words into subtitle cues."""
    cues: list[dict] = []
    current: dict | None = None

    for start, end, text in iter_words(segments):
        speaker = pick_speaker(turns, starts, start, end)
        if current is None:
            current = {"speaker": speaker, "start": start, "end": end, "words": [text]}
            continue

        gap = start - current["end"]
        duration_if_added = end - current["start"]
        if (
            speaker != current["speaker"]
            or gap > MAX_GAP_SECONDS
            or duration_if_added > MAX_CUE_SECONDS
        ):
            cues.append(current)
            current = {"speaker": speaker, "start": start, "end": end, "words": [text]}
            continue

        current["end"] = end
        current["words"].append(text)

    if current is not None:
        cues.append(current)
    return cues


def parse_args():
    p = argparse.ArgumentParser(description="Speaker diarization + transcription → SRT")
    p.add_argument("audio", help="Path to the audio file")
    p.add_argument("-l", "--lang", default="pt", help="Language code (default: pt)")
    p.add_argument(
        "-m", "--model", default="large-v3",
        help="Whisper model size (default: large-v3). Options: tiny, base, small, medium, large-v3, large-v3-turbo",
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
    audio_path = ensure_wav(args.audio)
    hf_token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1) Diarize ----------------------------------------------------------
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    pipeline.to(torch.device(device))

    diar_kwargs: dict = {}
    if args.num_speakers is not None:
        diar_kwargs["num_speakers"] = args.num_speakers

    with ProgressHook() as hook:
        diarization = pipeline(audio_path, hook=hook, **diar_kwargs)

    turns, starts = build_speaker_index(diarization)

    # Free diarization model before loading Whisper to reduce peak memory
    del pipeline
    del diarization
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- 2) Transcribe -------------------------------------------------------
    compute_type = args.compute_type or ("float16" if device == "cuda" else "int8")
    asr = WhisperModel(args.model, device=device, compute_type=compute_type)

    segments, _info = asr.transcribe(
        audio_path,
        language=args.lang,
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,
        word_timestamps=True,
    )

    # --- 3) Build SRT --------------------------------------------------------
    srt_lines: list[str] = []
    idx = 1
    for cue in build_cues(segments, turns, starts):
        text = " ".join(cue["words"]).strip()
        if not text:
            continue
        srt_lines.append(str(idx))
        srt_lines.append(f"{srt_timestamp(cue['start'])} --> {srt_timestamp(cue['end'])}")
        srt_lines.append(f"{cue['speaker']}: {text}")
        srt_lines.append("")
        idx += 1

    out_path = os.path.splitext(args.audio)[0] + ".speaker.srt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    print(f"Wrote {idx - 1} subtitles → {out_path}")


if __name__ == "__main__":
    main()
