# whisper-batch-transcribe

Transcribe thousands of audio files on your own GPU with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), into a single CSV.

Built for large folders of recordings, from thousands of short clips to long phone calls, in any language Whisper supports.

## Features

- **One command, one CSV:** every file becomes a row with its path, duration and transcript.
- **Batched GPU decoding:** short clips are transcribed several at a time, which is much faster for datasets of many small files.
- **Resumable:** stop it any time (Ctrl+C, reboot, crash) and run it again; finished files are skipped.
- **Fault tolerant:** a broken file is reported and skipped, not fatal.
- **Loop protection:** transcripts that run far longer than the clip could contain (Whisper repeating itself) are cut short, so one bad clip can't stall the run.
- **Local and private:** audio never leaves your machine, and there are no API costs.
- **Zero CUDA setup:** the CUDA libraries come from pip, so you only need an NVIDIA driver.
- **Live progress with ETA.**

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU with a recent driver (recommended), or a CPU (works, but much slower)
- Linux, WSL2 or Windows. On macOS it runs on the CPU.

## Quick start

```bash
git clone https://github.com/george-dina/whisper-batch-transcribe.git
cd whisper-batch-transcribe
./scripts/setup.sh
source .venv/bin/activate

python transcribe.py /path/to/recordings --language ro
```

The first run downloads the model (about 3 GB for `large-v3`). Results go to `transcripts.csv`.

<details>
<summary>Windows (PowerShell, without WSL)</summary>

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python transcribe.py C:\path\to\recordings --language ro
```

</details>

## Usage

```
python transcribe.py INPUT [options]
```

| Option | Default | Description |
|---|---|---|
| `INPUT` | | Folder with audio files, searched recursively |
| `-o, --output` | `transcripts.csv` | Output CSV file |
| `-l, --language` | auto-detect | Language code such as `ro` or `en`. Setting it is strongly recommended. |
| `-m, --model` | `large-v3` | Any faster-whisper model, e.g. `large-v3-turbo`, `medium`, or a local path |
| `--device` | `auto` | `auto`, `cuda` or `cpu` |
| `--compute-type` | `float16` on GPU, `int8` on CPU | Use `int8_float16` on GPUs with 6 GB of VRAM or less |
| `-b, --batch-size` | `8` on GPU, `1` on CPU | Clips decoded at once. Try `16` on 8 GB GPUs; `1` disables batching |
| `--beam-size` | `5` | `1` is faster, `5` is more accurate |
| `--no-vad` | off | Disable the voice activity filter |
| `--workers` | `4` | Threads that decode audio in the background |
| `--max-tokens-per-second` | `12` | Length cap for clips up to 30 s, which stops repetition loops early. `0` disables it |
| `--no-fallback` | off | Decode each clip once instead of retrying at higher temperatures. Faster, slightly less robust |
| `--ext` | `wav mp3 m4a flac ogg opus` | File extensions to include |
| `--cache-dir` | Hugging Face cache | Where to store downloaded models |

### Examples

```bash
# Romanian, best accuracy
python transcribe.py ./audio -l ro

# Faster model, separate output file
python transcribe.py ./audio -l ro -m large-v3-turbo -o transcripts_turbo.csv

# Bigger batches on an 8 GB GPU
python transcribe.py ./audio -l ro -b 16

# Keep models inside the project folder
python transcribe.py ./audio -l ro --cache-dir ./models
```

## Output

A UTF-8 CSV (with BOM, so Excel shows diacritics correctly):

| file | duration_s | text |
|---|---|---|
| `day1/clip_0001.wav` | 12.48 | Bună ziua, vă rog să-mi spuneți... |
| `day1/clip_0002.wav` | 21.03 | ... |

Paths are relative to the input folder.

## Second pass for short transcripts

Sometimes a transcript is much shorter than its recording should produce: a quiet speaker cut out by the silence filter, or Whisper skipping a stretch it was unsure about. `repass.py` finds those files by comparing transcript length to duration and re-transcribes only them, with the most thorough settings (full audio with no silence filter, one window at a time, no text carried between windows).

```bash
# 1. See how many files look too short, without transcribing anything
python repass.py ./audio -c transcripts.csv -l ro --dry-run

# 2. Re-transcribe them
python repass.py ./audio -c transcripts.csv -l ro
```

- A file counts as suspicious below `--min-cps` characters per second (default `6`; normal conversation is usually 10 to 15). The dry run prints the distribution for your data, so you can pick a threshold that fits.
- The new transcript replaces the old one only if it is longer after repetition loops are removed, so the second pass can't make a row worse by that measure.
- If transcripts are still too short, run it again with `--no-skip --fresh`. `--no-skip` stops Whisper from skipping 30-second windows it judges to be silence, which recovers more speech but can add filler text (such as "Mulțumesc") where there really is silence. `--fresh` retries files the earlier second pass already handled; a result is still only kept if it is longer.
- The original CSV is saved as `transcripts_before_repass.csv`, and every result goes to `transcripts_repass_log.csv` as it is produced. If the run is interrupted, run the same command again and it continues.

### Removing phrases Whisper invents

On silence or unclear audio, Whisper sometimes writes stock phrases from the YouTube videos and subtitles it was trained on, such as "Nu uitați să dați like", "Mulțumesc pentru vizionare" or "Subtitrarea realizată de comunitatea Amara.org". `--hallucinations` finds every transcript containing one of these, transcribes those files again, removes the invented sentences from both the old and the new version, and keeps the longer cleaned one:

```bash
# See how many files are affected, with a short excerpt of each
python repass.py ./audio -c transcripts.csv -l ro --hallucinations --dry-run

# Clean them
python repass.py ./audio -c transcripts.csv -l ro --hallucinations
```

The built-in list (in `repass.py`) only contains phrases specific to videos and subtitles; a plain "Mulțumesc" is never touched. To target your own phrase, use `--match` with a regular expression, for example `--match 'dați like'`. Check the excerpts from the dry run first, in case a phrase also occurs legitimately in your recordings.

## How batching works

Whisper reads audio in 30-second windows, so a folder of short clips normally keeps the GPU busy with one small clip at a time. This tool instead:

1. Decodes files and trims leading and trailing silence in background threads, so the GPU never waits on disk.
2. Joins a group of clips into one signal separated by silence, and marks each clip as its own window using faster-whisper's `clip_timestamps`. Every clip is still transcribed in isolation, never mixed with its neighbors.
3. Decodes `--batch-size` windows at once on the GPU and maps each result back to its file.
4. Applies Whisper's usual quality checks. Batched decoding skips Whisper's retry-at-higher-temperature step, so any clip that looks unreliable (low confidence or repetitive output) is transcribed again on its own with the full checks.

Files longer than 30 seconds are handled differently: each one is split into 30-second windows (at pauses, using the voice activity filter), and the windows are decoded `--batch-size` at a time. On CPU, everything takes the regular one-file-at-a-time path.

## Performance

Rough expectations for clips of 20 to 30 seconds with `large-v3` on a laptop RTX 4070 (8 GB):

| Mode | Per clip | 6000 clips |
|---|---|---|
| One at a time (`-b 1`) | 0.5 to 1 s | about 1 to 1.5 hours |
| Batched (`-b 8` or `-b 16`) | roughly 2 to 3x faster | about 20 to 40 minutes |

These are estimates; speed depends heavily on cooling, power mode and audio length, so test on a small batch first. `large-v3-turbo` is several times faster again, with slightly lower accuracy.

Tips:

- **Always set `--language`.** Auto-detection on short clips can pick the wrong language, and in batched mode a whole group shares one detected language.
- **Compare models on 50 files** before a long run. If `large-v3-turbo` is good enough for your audio, it saves a lot of time.
- **Laptops:** keep the charger in and use a performance power profile.

## Troubleshooting

**`Failed to initialize NVML` or `nvidia-smi` can't reach the driver**
The NVIDIA GPU is off or not visible. On laptops with hybrid graphics (e.g. ASUS G-Helper or Armoury Crate), switch the GPU mode from Eco to Standard. In WSL, run `wsl --shutdown` from PowerShell afterwards and reopen the terminal. Do not install an NVIDIA driver inside WSL; it uses the Windows one.

**`CUDA out of memory`**
Lower `--batch-size`, use `--compute-type int8_float16`, or pick a smaller model.

**Errors about `libcublas` or `libcudnn`**
Reinstall the CUDA wheels inside the virtual environment: `pip install --force-reinstall -r requirements.txt`.

**Every file fails with `unexpected keyword argument 'metadata_errors'`**
PyAV 19 is incompatible with faster-whisper 1.2.x. The pinned requirements avoid it; if you installed packages by hand, run `pip install "av<19"`.

**Much slower than expected while the GPU is busy**
Whisper is probably looping on some clips: it repeats a phrase until its length limit, then retries up to five more times at higher temperatures. The length cap (on by default) bounds that work. For more speed, add `--no-fallback`, or switch to `-m large-v3-turbo`, whose decoder is much lighter.

**Words missing at the start or end of clips**
Try `--no-vad`. The voice activity filter prevents hallucinated text on silence, but can occasionally trim quiet speech.

## Development

The tests run the real faster-whisper pipeline with the neural network replaced by a small deterministic stub, so they need no GPU and no model download:

```bash
pip install pytest
pytest
```

## License

[MIT](LICENSE)
