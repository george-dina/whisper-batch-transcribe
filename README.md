# whisper-batch-transcribe

Transcribe thousands of audio files on your own GPU with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), into a single CSV.

Built for large folders of short recordings (thousands of clips under 30 seconds each), but works with files of any length and any language Whisper supports.

## Features

- **One command, one CSV:** every file becomes a row with its path, duration and transcript.
- **Resumable:** stop it any time (Ctrl+C, reboot, crash) and run it again; finished files are skipped.
- **Fault tolerant:** a broken file is reported and skipped, not fatal.
- **Local and private:** audio never leaves your machine, and there are no API costs.
- **Zero CUDA setup:** the CUDA libraries come from pip, so you only need an NVIDIA driver.
- **Live progress with ETA.**

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU with a recent driver (recommended), or a CPU (works, but much slower)
- Linux, WSL2 or Windows. On macOS it runs on the CPU.

## Quick start

```bash
git clone https://github.com/<your-username>/whisper-batch-transcribe.git
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
| `--beam-size` | `5` | `1` is faster, `5` is more accurate |
| `--no-vad` | off | Disable the voice activity filter |
| `--ext` | `wav mp3 m4a flac ogg opus` | File extensions to include |
| `--cache-dir` | Hugging Face cache | Where to store downloaded models |

### Examples

```bash
# Romanian, best accuracy
python transcribe.py ./audio -l ro

# Faster model, separate output file
python transcribe.py ./audio -l ro -m large-v3-turbo -o transcripts_turbo.csv

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

## Performance

Rough expectations for clips of 20 to 30 seconds with `large-v3` on a laptop RTX 4070 (8 GB): around 0.5 to 1 second per clip, so a few thousand clips in about an hour. `large-v3-turbo` is several times faster with slightly lower accuracy. Speed depends heavily on cooling and power mode, so test on a small batch first.

Tips:

- **Always set `--language`.** Auto-detection on short clips can pick the wrong language.
- **Compare models on 50 files** before a long run. If `large-v3-turbo` is good enough for your audio, it saves a lot of time.
- **Laptops:** keep the charger in and use a performance power profile.

## Troubleshooting

**`Failed to initialize NVML` or `nvidia-smi` can't reach the driver**
The NVIDIA GPU is off or not visible. On laptops with hybrid graphics (e.g. ASUS G-Helper or Armoury Crate), switch the GPU mode from Eco to Standard. In WSL, run `wsl --shutdown` from PowerShell afterwards and reopen the terminal. Do not install an NVIDIA driver inside WSL; it uses the Windows one.

**`CUDA out of memory`**
Use `--compute-type int8_float16` or a smaller model.

**Errors about `libcublas` or `libcudnn`**
Reinstall the CUDA wheels inside the virtual environment: `pip install --force-reinstall -r requirements.txt`.

**Every file fails with `unexpected keyword argument 'metadata_errors'`**
PyAV 19 is incompatible with faster-whisper 1.2.x. The pinned requirements avoid it; if you installed packages by hand, run `pip install "av<19"`.

**Words missing at the start or end of clips**
Try `--no-vad`. The voice activity filter prevents hallucinated text on silence, but can occasionally trim quiet speech.

## License

[MIT](LICENSE)
