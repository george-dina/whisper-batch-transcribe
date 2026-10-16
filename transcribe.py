#!/usr/bin/env python3
"""Batch-transcribe a folder of audio files with faster-whisper.

Writes one CSV row per file (file, duration_s, text). Safe to stop and
restart: files already in the output CSV are skipped.

Short clips (30 s or less after trimming silence) are transcribed several at
a time on the GPU. Longer files are split into 30 s windows that are decoded
in parallel. Any short clip whose batched result looks unreliable is
transcribed again on its own with Whisper's full quality checks.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from bisect import bisect_right
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar

DEFAULT_EXTENSIONS = ("wav", "mp3", "m4a", "flac", "ogg", "opus")
FIELDS = ("file", "duration_s", "text")

SAMPLE_RATE = 16000
MAX_BATCHED_CLIP_S = 30.0  # Whisper's context window
GAP_S = 1.0  # silence placed between clips when they are grouped

MAX_TOKENS = 440  # Whisper allows 448 tokens per window, including the prompt
TOKEN_HEADROOM = 32  # extra tokens allowed on top of the per-second budget

# Whisper's own quality thresholds, applied to batched results
NO_SPEECH_THRESHOLD = 0.6
LOG_PROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4


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
# Command line
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    p.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=None,
        help="clips decoded at once on the GPU; 1 disables batching (default: 8 on GPU, 1 on CPU)",
    )
    p.add_argument("--beam-size", type=int, default=5, help="beam size: 1 is faster, 5 is more accurate (default: 5)")
    p.add_argument("--no-vad", action="store_true", help="disable the voice activity filter")
    p.add_argument(
        "--max-tokens-per-second",
        type=float,
        default=12.0,
        help="cap on transcript length for clips up to 30 s, which stops repetition loops early; 0 disables (default: 12)",
    )
    p.add_argument(
        "--no-fallback",
        action="store_true",
        help="decode each clip once, skipping Whisper's retries at higher temperature (faster, slightly less robust)",
    )
    p.add_argument("--workers", type=int, default=4, help="threads that decode audio in the background (default: 4)")
    p.add_argument("--ext", nargs="+", default=list(DEFAULT_EXTENSIONS), help="file extensions to include (default: wav mp3 m4a flac ogg opus)")
    p.add_argument("--cache-dir", type=Path, default=None, help="where to store downloaded models (default: Hugging Face cache)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Files and output
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Audio loading (runs in background threads)
# --------------------------------------------------------------------------- #
@dataclass
class Clip:
    rel: str
    audio: object = None  # numpy float32 array at 16 kHz
    duration: float = 0.0
    span: tuple[int, int] | None = None  # speech region in samples; None means no speech
    error: str | None = None
    segments: list = field(default_factory=list)

    @property
    def span_seconds(self) -> float:
        return 0.0 if self.span is None else (self.span[1] - self.span[0]) / SAMPLE_RATE


def load_clip(path: Path, root: Path, use_vad: bool) -> Clip:
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    clip = Clip(rel=str(path.relative_to(root)))
    try:
        audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    except Exception as exc:
        clip.error = str(exc) or exc.__class__.__name__
        return clip

    clip.audio = audio
    clip.duration = len(audio) / SAMPLE_RATE
    if len(audio) == 0:
        return clip
    if use_vad:
        speech = get_speech_timestamps(audio, VadOptions())
        if speech:
            clip.span = (speech[0]["start"], speech[-1]["end"])
    else:
        clip.span = (0, len(audio))
    return clip


T = TypeVar("T")
R = TypeVar("R")


def prefetch(items: Iterable[T], fn: Callable[[T], R], workers: int, window: int) -> Iterator[R]:
    """Like map(fn, items), run in threads, in order, with at most `window` results in memory."""
    it = iter(items)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        pending = deque(ex.submit(fn, item) for item in itertools.islice(it, window))
        while pending:
            fut = pending.popleft()
            for item in itertools.islice(it, 1):
                pending.append(ex.submit(fn, item))
            yield fut.result()


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
def segments_text(segments: Iterable) -> str:
    return " ".join(s.text.strip() for s in segments if s.text.strip()).strip()


def is_no_speech(seg) -> bool:
    return seg.no_speech_prob > NO_SPEECH_THRESHOLD and seg.avg_logprob < LOG_PROB_THRESHOLD


def is_unreliable(seg) -> bool:
    """Batched decoding skips Whisper's temperature fallback, so flag results it would have retried."""
    return seg.compression_ratio > COMPRESSION_RATIO_THRESHOLD or seg.avg_logprob < LOG_PROB_THRESHOLD


def token_cap(seconds: float, args) -> int | None:
    """Most tokens a clip of this length may produce, or None for no cap.

    Normal speech needs well under 10 tokens per second, so a transcript that
    runs past the budget is almost always the model repeating itself. Files
    longer than one window are left alone, since their prompt carries earlier
    text and the budget would not apply cleanly.
    """
    if args.max_tokens_per_second <= 0 or seconds > MAX_BATCHED_CLIP_S:
        return None
    return min(MAX_TOKENS, int(args.max_tokens_per_second * seconds) + TOKEN_HEADROOM)


def hit_cap(segments: list, cap: int | None) -> bool:
    return cap is not None and sum(len(s.tokens) for s in segments) >= cap - 2


def transcribe_single(model, clip: Clip, args, stats: dict | None = None) -> str:
    cap = token_cap(clip.span_seconds or clip.duration, args)
    kwargs = {"temperature": 0.0} if args.no_fallback else {}
    segments, _ = model.transcribe(
        clip.audio,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        max_new_tokens=cap,
        **kwargs,
    )
    segments = list(segments)
    if stats is not None and hit_cap(segments, cap):
        stats["capped"] += 1
    return segments_text(segments)


def transcribe_group(pipe, clips: list[Clip], args) -> None:
    """Transcribe several short clips in one batched call; fills clip.segments.

    The clips are joined into one signal separated by silence, and each clip is
    passed as its own chunk via clip_timestamps, so the model sees every clip
    in isolation. Each resulting segment starts at its chunk's offset, which
    maps it back to its clip.
    """
    import numpy as np

    gap = np.zeros(int(GAP_S * SAMPLE_RATE), dtype=np.float32)
    pieces, stamps, offsets = [], [], []
    pos = 0
    for clip in clips:
        start, end = clip.span
        speech = clip.audio[start:end]
        pieces += [speech, gap]
        offsets.append(pos / SAMPLE_RATE)
        stamps.append({"start": pos / SAMPLE_RATE, "end": (pos + len(speech)) / SAMPLE_RATE})
        pos += len(speech) + len(gap)

    segments, _ = pipe.transcribe(
        np.concatenate(pieces),
        language=args.language,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        clip_timestamps=stamps,
        without_timestamps=True,
        max_new_tokens=token_cap(max(c.span_seconds for c in clips), args),
    )
    for clip in clips:
        clip.segments = []
    for seg in segments:
        idx = bisect_right(offsets, seg.start + GAP_S / 2) - 1
        clips[max(idx, 0)].segments.append(seg)


def transcribe_long_batched(pipe, clip: Clip, args) -> str:
    """Transcribe one long file by decoding its 30 s windows in parallel."""
    kwargs = {}
    if args.no_vad:
        # Without VAD the pipeline needs explicit windows: plain 30 s slices
        step = MAX_BATCHED_CLIP_S
        kwargs["clip_timestamps"] = [
            {"start": t, "end": min(t + step, clip.duration)}
            for t in (i * step for i in range(int(clip.duration // step) + 1))
            if clip.duration - t > 0.1
        ]
    else:
        kwargs["vad_filter"] = True
    segments, _ = pipe.transcribe(
        clip.audio,
        language=args.language,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        **kwargs,
    )
    return segments_text(segments)


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #
class Progress:
    def __init__(self, total: int, every_s: float = 2.0):
        self.total, self.done, self.every_s = total, 0, every_s
        self.start = self.last = time.monotonic()

    def update(self, n: int, note: str = "") -> None:
        self.done += n
        now = time.monotonic()
        if now - self.last < self.every_s and self.done < self.total:
            return
        self.last = now
        elapsed = now - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else 0.0
        msg = f"[{self.done}/{self.total}] {rate:.1f} files/s, ETA {fmt_time(eta)}"
        print(f"{msg}  {note}" if note else msg, flush=True)


def fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(args, model, pipe=None) -> int:
    """Transcribe everything not yet in the output CSV. Returns the number of failed files."""
    files = find_audio(args.input, args.ext)
    done = load_done(args.output)
    todo = [f for f in files if str(f.relative_to(args.input)) not in done]
    print(f"Found {len(files)} files, {len(files) - len(todo)} already done, {len(todo)} to go.")
    if not todo:
        return 0

    batched = pipe is not None and args.batch_size > 1
    group_size = args.batch_size * 4 if batched else 1
    stats = {"failed": 0, "retried": 0, "silent": 0, "capped": 0}
    progress = Progress(len(todo))

    new_file = not args.output.exists() or args.output.stat().st_size == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        def write(clip: Clip, text: str) -> None:
            writer.writerow({"file": clip.rel, "duration_s": f"{clip.duration:.2f}", "text": text})

        def fail(clip: Clip, reason: str) -> None:
            stats["failed"] += 1
            print(f"FAILED {clip.rel}: {reason}", file=sys.stderr, flush=True)

        def long_batched(clip: Clip) -> None:
            try:
                write(clip, transcribe_long_batched(pipe, clip, args))
            except Exception as exc:
                print(f"Batched decoding failed for {clip.rel} ({exc}); retrying one window at a time.", file=sys.stderr, flush=True)
                single(clip)

        def single(clip: Clip) -> None:
            try:
                write(clip, transcribe_single(model, clip, args, stats))
            except Exception as exc:
                fail(clip, str(exc))

        def flush_group(group: list[Clip]) -> None:
            try:
                transcribe_group(pipe, group, args)
            except Exception as exc:
                print(f"Batch failed ({exc}); retrying {len(group)} files one by one.", file=sys.stderr, flush=True)
                for clip in group:
                    single(clip)
            else:
                for clip in group:
                    if not clip.segments or all(is_no_speech(s) for s in clip.segments):
                        stats["silent"] += 1
                        write(clip, "")
                    elif not args.no_fallback and any(is_unreliable(s) for s in clip.segments):
                        stats["retried"] += 1
                        single(clip)
                    else:
                        write(clip, segments_text(clip.segments))
            fh.flush()
            for clip in group:
                clip.audio = None
            progress.update(len(group))

        group: list[Clip] = []
        loader = prefetch(
            todo,
            lambda p: load_clip(p, args.input, use_vad=not args.no_vad),
            workers=args.workers,
            window=max(group_size * 2, 8),
        )
        for clip in loader:
            if clip.error:
                fail(clip, clip.error)
                progress.update(1)
            elif clip.span is None:
                stats["silent"] += 1
                write(clip, "")
                fh.flush()
                progress.update(1)
            elif not batched or clip.span_seconds > MAX_BATCHED_CLIP_S:
                long_batched(clip) if batched else single(clip)
                fh.flush()
                clip.audio = None
                progress.update(1)
            else:
                group.append(clip)
                if len(group) >= group_size:
                    flush_group(group)
                    group = []
        if group:
            flush_group(group)

    total = time.monotonic() - progress.start
    ok = len(todo) - stats["failed"]
    print(
        f"Done in {fmt_time(total)}. {ok} transcribed "
        f"({stats['silent']} without speech, {stats['retried']} re-checked one by one), {stats['failed']} failed."
    )
    if stats["capped"]:
        print(f"{stats['capped']} transcripts were cut short because they looked like repetition loops.")
    if stats["failed"]:
        print("Run the same command again to retry the failed files.")
    return stats["failed"]


def main() -> int:
    args = parse_args()
    ensure_cuda_libs()

    if not args.input.is_dir():
        print(f"error: {args.input} is not a folder", file=sys.stderr)
        return 1

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    device, compute_type = resolve_device(args.device, args.compute_type)
    if args.batch_size is None:
        args.batch_size = 8 if device == "cuda" else 1
    if args.batch_size > 1 and args.language is None:
        print("Note: without --language, each batch uses the language detected from its first clips.")

    if not find_audio(args.input, args.ext):
        print(f"No audio files found in {args.input}.")
        return 0

    print(f"Loading {args.model} on {device} ({compute_type}), batch size {args.batch_size}...")
    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute_type,
        download_root=str(args.cache_dir) if args.cache_dir else None,
    )
    pipe = BatchedInferencePipeline(model=model) if args.batch_size > 1 else None
    return 1 if run(args, model, pipe) else 0


if __name__ == "__main__":
    sys.exit(main())
