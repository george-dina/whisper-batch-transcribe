#!/usr/bin/env python3
"""Second pass: re-transcribe files whose transcript looks too short for their duration.

Reads the CSV written by transcribe.py, picks rows with fewer characters per
second than --min-cps, and transcribes those files again with the most
thorough settings: the full audio (no silence filter, so quiet speakers are
not dropped), one 30 s window at a time, without carrying text from one
window to the next (which can make Whisper skip whole stretches).

A new transcript replaces the old one only if it is longer once repetition
loops are removed. Results are logged as they come in, so an interrupted run
resumes where it stopped. The original CSV is backed up before it is updated.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import transcribe as tx

COMPRESSION_RATIO_THRESHOLD = 2.4  # Whisper's own repetition threshold
LOG_FIELDS = ("file", "old_chars", "new_chars", "replaced", "text")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-transcribe files whose transcript is too short for their duration.")
    p.add_argument("input", type=Path, help="the audio folder used for the first pass")
    p.add_argument("-c", "--csv", type=Path, default=Path("transcripts.csv"), help="transcript CSV to check and update (default: transcripts.csv)")
    p.add_argument("--min-cps", type=float, default=6.0, help="re-transcribe rows with fewer characters per second than this (default: 6)")
    p.add_argument("--min-duration", type=float, default=5.0, help="ignore files shorter than this many seconds (default: 5)")
    p.add_argument("--dry-run", action="store_true", help="only report and list the candidates, transcribe nothing")
    p.add_argument("-l", "--language", default=None, help="language code such as ro or en (default: auto-detect)")
    p.add_argument("-m", "--model", default="large-v3", help="model name or local path (default: large-v3)")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"), help="inference device (default: auto)")
    p.add_argument("--compute-type", default=None, help="CTranslate2 compute type (default: float16 on GPU, int8 on CPU)")
    p.add_argument("--beam-size", type=int, default=5, help="beam size (default: 5)")
    p.add_argument("--vad", action="store_true", help="keep the silence filter on (default: off, so quiet speech is not cut)")
    p.add_argument("--cache-dir", type=Path, default=None, help="where to store downloaded models (default: Hugging Face cache)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #
def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows_atomic(path: Path, rows: list[dict], fields) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def chars_per_second(row: dict) -> float:
    try:
        duration = float(row["duration_s"])
    except (KeyError, ValueError):
        return float("inf")
    return len(row["text"]) / duration if duration > 0 else float("inf")


def find_candidates(rows: list[dict], min_cps: float, min_duration: float) -> list[dict]:
    return [
        r for r in rows
        if float(r.get("duration_s") or 0) >= min_duration and chars_per_second(r) < min_cps
    ]


def summarize(rows: list[dict], candidates: list[dict], args) -> None:
    rates = sorted(chars_per_second(r) for r in rows if float(r.get("duration_s") or 0) >= args.min_duration)
    if not rates:
        print("No rows long enough to check.")
        return
    q = statistics.quantiles(rates, n=20) if len(rates) >= 20 else rates
    print(f"Characters per second across {len(rates)} files: "
          f"5th percentile {q[0]:.1f}, median {statistics.median(rates):.1f}, 95th percentile {q[-1]:.1f}")
    hours = sum(float(r["duration_s"]) for r in candidates) / 3600
    print(f"{len(candidates)} files below {args.min_cps:g} chars/s ({hours:.1f} hours of audio) will be re-transcribed.")


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
def thorough_transcript(model, clip: tx.Clip, args) -> str:
    from faster_whisper.transcribe import get_compression_ratio

    segments, _ = model.transcribe(
        clip.audio,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=args.vad,
        condition_on_previous_text=False,
    )
    # Drop repetition loops so they can't win the "longer is better" comparison
    kept = [s for s in segments if s.text.strip() and get_compression_ratio(s.text.strip()) <= COMPRESSION_RATIO_THRESHOLD]
    return tx.segments_text(kept)


def load_log(path: Path) -> dict[str, dict]:
    return {r["file"]: r for r in read_rows(path)} if path.exists() else {}


def run(args, model) -> int:
    rows = read_rows(args.csv)
    candidates = find_candidates(rows, args.min_cps, args.min_duration)
    summarize(rows, candidates, args)

    if args.dry_run:
        out = args.csv.with_name(args.csv.stem + "_repass_candidates.csv")
        listing = [{"file": r["file"], "duration_s": r["duration_s"], "chars": len(r["text"]),
                    "chars_per_s": f"{chars_per_second(r):.2f}"} for r in candidates]
        write_rows_atomic(out, listing, ("file", "duration_s", "chars", "chars_per_s"))
        print(f"Candidate list written to {out}. Nothing was transcribed.")
        return 0

    log_path = args.csv.with_name(args.csv.stem + "_repass_log.csv")
    log = load_log(log_path)
    todo = [r for r in candidates if r["file"] not in log]
    if log:
        print(f"{len(candidates) - len(todo)} already re-transcribed in an earlier run, {len(todo)} to go.")

    failed = 0
    if todo:
        new_log = not log_path.exists()
        progress = tx.Progress(len(todo))
        with log_path.open("a", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            if new_log:
                writer.writeheader()
            for row in todo:
                clip = tx.load_clip(args.input / row["file"], args.input, use_vad=False)
                if clip.error:
                    failed += 1
                    print(f"FAILED {row['file']}: {clip.error}", file=sys.stderr, flush=True)
                    progress.update(1)
                    continue
                try:
                    new = thorough_transcript(model, clip, args)
                except Exception as exc:
                    failed += 1
                    print(f"FAILED {row['file']}: {exc}", file=sys.stderr, flush=True)
                    progress.update(1)
                    continue
                old = row["text"]
                entry = {"file": row["file"], "old_chars": len(old), "new_chars": len(new),
                         "replaced": int(len(new) > len(old)), "text": new}
                writer.writerow(entry)
                fh.flush()
                log[row["file"]] = entry
                progress.update(1, f"{row['file']}: {len(old)} -> {len(new)} chars")

    # Apply the improvements to the main CSV, keeping row order
    improved = {f: e["text"] for f, e in log.items() if str(e["replaced"]) == "1"}
    changed = 0
    for row in rows:
        if row["file"] in improved and row["text"] != improved[row["file"]]:
            row["text"] = improved[row["file"]]
            changed += 1
    if changed:
        backup = args.csv.with_name(args.csv.stem + "_before_repass.csv")
        if not backup.exists():
            backup.write_bytes(args.csv.read_bytes())
            print(f"Original saved as {backup}.")
        write_rows_atomic(args.csv, rows, tx.FIELDS)

    total_replaced = len(improved)
    print(f"Done. {total_replaced} of {len(log)} re-transcribed files got a longer transcript"
          f" ({changed} updated in {args.csv.name} now), {failed} failed.")
    print(f"Per-file details: {log_path}")
    return failed


def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        print(f"error: {args.csv} not found", file=sys.stderr)
        return 1
    if args.dry_run:
        return run(args, model=None)

    tx.ensure_cuda_libs()
    from faster_whisper import WhisperModel

    device, compute_type = tx.resolve_device(args.device, args.compute_type)
    print(f"Loading {args.model} on {device} ({compute_type})...")
    start = time.monotonic()
    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute_type,
        download_root=str(args.cache_dir) if args.cache_dir else None,
    )
    failed = run(args, model)
    print(f"Total time {tx.fmt_time(time.monotonic() - start)}.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
