# Speaker diarization

Portuguese speaker-labeled SRT from one audio file.

Needs `ffmpeg`, pyenv, Python 3.11, and `HF_TOKEN`. Accept the Hugging Face conditions for `pyannote/speaker-diarization-community-1` first.

Install each requirements file with the PyTorch CUDA index so pip does not pick a CPU build of torch:

```bash
pip install -r <requirements-file> \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple
export HF_TOKEN=hf_...
```

## Input

Both scripts take one audio path. `ffmpeg` must be able to decode it (`wav`, `mp3`, `m4a`, `flac`, and similar).

`speaker_diarization1.py` passes that file through unchanged. `speaker_diarization2.py` uses a 16 kHz mono WAV as-is and otherwise writes `{stem}.16k.wav` before diarization.

## speaker_diarization1.py

Requirements: [requirements-speaker_diarization1.txt](requirements-speaker_diarization1.txt) (OpenAI Whisper and pyannote 4).

```bash
pyenv virtualenv 3.11.9 speaker-diarization1
pyenv activate speaker-diarization1
pip install -r requirements-speaker_diarization1.txt \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple

python speaker_diarization1.py audio.mp3
```

Whisper `large`, Portuguese, segment-level speaker labels. Output: `{stem}.speaker.pt.srt`.

## speaker_diarization2.py

Requirements: [requirements-speaker_diarization2.txt](requirements-speaker_diarization2.txt) (faster-whisper and pyannote 4).

```bash
pyenv virtualenv 3.11.9 speaker-diarization2
pyenv activate speaker-diarization2
pip install -r requirements-speaker_diarization2.txt \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple

python speaker_diarization2.py audio.mp3
python speaker_diarization2.py audio.mp3 -l pt -m large-v3-turbo --num-speakers 2
```

faster-whisper `large-v3` and word-level speaker labels. Output: `{stem}.speaker.srt`.

`-l` is the language (default `pt`). `-m` is the Whisper model. `--num-speakers` is optional when the speaker count is known. `--compute-type` defaults to `float16` on CUDA and `int8` on CPU.

This directory's `.python-version` selects the existing `speaker-diarization` environment, which already has the script 2 packages. Activate `speaker-diarization1` when running script 1.