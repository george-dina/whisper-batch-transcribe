#!/usr/bin/env python3
"""Batch-transcribe a folder of audio files with faster-whisper.

Writes one CSV row per file (file, duration_s, text). Safe to stop and
restart: files already in the output CSV are skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

DEFAULT_EXTENSIONS = ("wav", "mp3", "m4a", "flac", "ogg", "opus")
FIELDS = ("file", "duration_s", "text")


# --------------------------------------------------------------------------- #
# CUDA library discovery
# --------------------------------------------------------------------------- #
def _nvidia_lib_dirs(subdir: str) -> list[Path]:
    """Return cuBLAS/cuDNN library folders installed via pip (nvidia-*-cu12)."""
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return []
    dirs = []
    for root in spec.submodule_search_locations:
        for pkg in ("cublas", "cudnn"):
            d = Path(root) / pkg / subdir
            if d.is_dir():
                dirs.append(d)
    return dirs


def ensure_cuda_libs() -> None:
    """Make pip-installed CUDA libraries visible to CTranslate2.

    On Linux the dynamic loader only reads LD_LIBRARY_PATH at process start,
    so we set it and re-exec once. On Windows we register the DLL folders.
    """
    if sys.platform.startswith("linux"):
        if os.environ.get("_WBT_REEXEC"):
            return
        dirs = [str(d) for d in _nvidia_lib_dirs("lib")]
        current = os.environ.get("LD_LIBRARY_PATH", "")
        missing = [d for d in dirs if d not in current.split(os.pathsep)]
        if missing:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(missing + [current]).rstrip(os.pathsep)
            os.environ["_WBT_REEXEC"] = "1"
            os.execv(sys.executable, [sys.executable, *sys.argv])
    elif sys.platform == "win32":
        for d in _nvidia_lib_dirs("bin"):
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-transcribe audio files with faster-whisper into a single CSV.",
    )
    p.add_argument("input", type=Path, help="folder with audio files (searched recursively)")
    p.add_argument("-o", "--output", type=Path, default=Path("transcripts.csv"), help="output CSV file (default: transcripts.csv)")
    p.add_argument("-l", "--language", default=None, help="language code such as ro or en (default: auto-detect)")
    p.add_argument("-m", "--model", default="large-v3", help="model name or local path, e.g. large-v3-turbo (default: large-v3)")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"), help="inference device (default: auto)")
    p.add_argument(
        "--compute-type",
        default=None,
        help="CTranslate2 compute type (default: float16 on GPU, int8 on CPU; use int8_float16 for GPUs with 6 GB or less)",
    )
    p.add_argument("--beam-size", type=int, default=5, help="beam size: 1 is faster, 5 is more accurate (default: 5)")
    p.add_argument("--no-vad", action="store_true", help="disable the voice activity filter")
    p.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTENSIONS), help="file extensions to include (default: wav mp3 m4a flac ogg opus)")
    p.add_argument("--cache-dir", type=Path, default=None, help="where to store downloaded models (default: Hugging Face cache)")
    return p.parse_args()


def find_audio(folder: Path, extensions: list[str]) -> list[Path]:
    exts = {"." + e.lower().lstrip(".") for e in extensions}
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def load_done(output: Path) -> set[str]:
    if not output.exists():
        return set()
    with output.open(encoding="utf-8-sig", newline="") as fh:
        return {row["file"] for row in csv.DictReader(fh) if row.get("file")}


def resolve_device(device: str, compute_type: str | None) -> tuple[str, str]:
    if device == "auto":
        import ctranslate2

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def fmt_eta(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    ensure_cuda_libs()

    from faster_whisper import WhisperModel

    if not args.input.is_dir():
        print(f"error: {args.input} is not a folder", file=sys.stderr)
        return 1

    files = find_audio(args.input, args.ext)
    done = load_done(args.output)
    todo = [f for f in files if str(f.relative_to(args.input)) not in done]

    print(f"Found {len(files)} files, {len(files) - len(todo)} already done, {len(todo)} to go.")
    if not todo:
        return 0

    device, compute_type = resolve_device(args.device, args.compute_type)
    print(f"Loading {args.model} on {device} ({compute_type})...")
    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute_type,
        download_root=str(args.cache_dir) if args.cache_dir else None,
    )

    new_file = not args.output.exists() or args.output.stat().st_size == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    start = time.monotonic()

    with args.output.open("a", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for i, path in enumerate(todo, 1):
            rel = str(path.relative_to(args.input))
            try:
                segments, info = model.transcribe(
                    str(path),
                    language=args.language,
                    beam_size=args.beam_size,
                    vad_filter=not args.no_vad,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception as exc:  # keep going; failed files are retried on the next run
                failed += 1
                print(f"[{i}/{len(todo)}] FAILED {rel}: {exc}", file=sys.stderr)
                continue

            writer.writerow({"file": rel, "duration_s": f"{info.duration:.2f}", "text": text})
            fh.flush()

            elapsed = time.monotonic() - start
            eta = elapsed / i * (len(todo) - i)
            print(f"[{i}/{len(todo)}] {rel}  (ETA {fmt_eta(eta)})")

    total = time.monotonic() - start
    print(f"Done in {fmt_eta(total)}. {len(todo) - failed} transcribed, {failed} failed.")
    if failed:
        print("Run the same command again to retry the failed files.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
