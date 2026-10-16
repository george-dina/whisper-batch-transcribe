#!/usr/bin/env python3
"""Second pass: re-transcribe files whose transcript looks too short for their duration.

Reads the CSV written by transcribe.py, picks rows with fewer characters per
second than --min-cps, and transcribes those files again with the most
thorough settings: the full audio (no silence filter, so quiet speakers are
not dropped), one 30 s window at a time, without carrying text from one
window to the next (which can make Whisper skip whole stretches).

A new transcript replaces the old one only if it is longer once repetition
loops are removed.

With --match, rows are picked by a text pattern instead, for example a phrase
Whisper invents over unclear audio ("Nu uitați să dați like"). Sentences that
match are removed from both the old and the new transcript, and the longer of
the two cleaned versions is kept, so the invented phrase disappears either way.

Each change is written to the CSV as soon as a file is done and logged, so an
interrupted run resumes where it stopped. The original CSV is backed up before
the first change.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import sys
import time
from pathlib import Path

import transcribe as tx

COMPRESSION_RATIO_THRESHOLD = 2.4  # Whisper's own repetition threshold

# Phrases Whisper is known to invent over silence or unclear audio: leftovers
# from the YouTube videos and subtitle files it was trained on. Each is
# specific to video or subtitle content, so it should not occur in a normal
# phone call. Plain "mulțumesc" is deliberately not on the list.
HALLUCINATIONS = {
    "like": r"\blike\b",
    "subscribe / abonare": r"\bsubscribe\b|\babona(?:ți|ti)[- ]v[ăa]\b|\babonea(?:z[ăa]|za)[- ]te\b|s[ăa] v[ăa] abona(?:ți|ti)",
    "pentru vizionare": r"pentru vizionare",
    "clopoțel / notificări": r"clopo[țţt]el|activa(?:ți|ti) notific[ăa]ril",
    "comentariu în secțiune": r"l[ăa]sa(?:ți|ti) un comentariu|sec(?:ț|ţ|t)iunea de comentarii",
    "canalul meu / nostru": r"canalul (?:meu|nostru)",
    "subtitrare": r"subtitr|subtitl",
    "amara.org": r"amara\.org",
    "thanks for watching": r"thanks? (?:you )?for watching",
}
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
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="never skip a window as 'no speech'; recovers missed speech, but may add filler text on real silence",
    )
    p.add_argument("--fresh", action="store_true", help="ignore earlier second-pass results and try every candidate again")
    p.add_argument(
        "--hallucinations",
        action="store_true",
        help="pick rows containing any of Whisper's known invented phrases (like, subscribe, "
        "'pentru vizionare', subtitle credits, ...) and remove those sentences",
    )
    p.add_argument(
        "--match",
        metavar="REGEX",
        default=None,
        help="pick rows whose text matches this pattern (case-insensitive) instead of short rows, "
        "and remove the matching sentences, e.g. --match '\\blike\\b'",
    )
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


SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def compile_pattern(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def find_matches(rows: list[dict], pattern: re.Pattern) -> list[dict]:
    return [r for r in rows if pattern.search(r["text"])]


def remove_matching_sentences(text: str, pattern: re.Pattern) -> str:
    return " ".join(s for s in SENTENCE_SPLIT.split(text.strip()) if s and not pattern.search(s)).strip()


def snippet(text: str, pattern: re.Pattern, width: int = 60) -> str:
    m = pattern.search(text)
    if not m:
        return ""
    return text[max(0, m.start() - width): m.end() + width].replace("\n", " ")


def build_pattern(args) -> re.Pattern | None:
    parts = []
    if args.hallucinations:
        parts += list(HALLUCINATIONS.values())
    if args.match:
        parts.append(args.match)
    return compile_pattern("|".join(f"(?:{p})" for p in parts)) if parts else None


def summarize_matches(rows: list[dict], candidates: list[dict], pattern: re.Pattern, args=None) -> None:
    hours = sum(float(r.get("duration_s") or 0) for r in candidates) / 3600
    label = "known Whisper phrases" if args is not None and args.hallucinations and not args.match else f"/{pattern.pattern}/"
    print(f"{len(candidates)} of {len(rows)} files match {label} ({hours:.1f} hours of audio).")
    if args is not None and args.hallucinations:
        for name, rx in HALLUCINATIONS.items():
            n = sum(1 for r in candidates if compile_pattern(rx).search(r["text"]))
            if n:
                print(f"  {name}: {n} files")
    for r in candidates[:5]:
        print(f"  {r['file']}: ...{snippet(r['text'], pattern)}...")
    if len(candidates) > 5:
        print(f"  ... and {len(candidates) - 5} more")


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
        **({"no_speech_threshold": None} if args.no_skip else {}),
    )
    # Drop repetition loops so they can't win the "longer is better" comparison
    kept = [s for s in segments if s.text.strip() and get_compression_ratio(s.text.strip()) <= COMPRESSION_RATIO_THRESHOLD]
    return tx.segments_text(kept)


def load_log(path: Path) -> dict[str, dict]:
    return {r["file"]: r for r in read_rows(path)} if path.exists() else {}


def run(args, model) -> int:
    rows = read_rows(args.csv)
    pattern = build_pattern(args)
    if pattern:
        candidates = find_matches(rows, pattern)
        summarize_matches(rows, candidates, pattern, args)
    else:
        candidates = find_candidates(rows, args.min_cps, args.min_duration)
        summarize(rows, candidates, args)

    if args.dry_run:
        out = args.csv.with_name(args.csv.stem + "_repass_candidates.csv")
        listing = [{"file": r["file"], "duration_s": r["duration_s"], "chars": len(r["text"]),
                    "chars_per_s": f"{chars_per_second(r):.2f}",
                    "snippet": snippet(r["text"], pattern) if pattern else ""} for r in candidates]
        write_rows_atomic(out, listing, ("file", "duration_s", "chars", "chars_per_s", "snippet"))
        print(f"Candidate list written to {out}. Nothing was transcribed.")
        return 0

    suffix = "_repass_match_log.csv" if pattern else "_repass_log.csv"
    log_path = args.csv.with_name(args.csv.stem + suffix)
    log = {} if args.fresh else load_log(log_path)
    todo = [r for r in candidates if r["file"] not in log]
    if len(todo) < len(candidates):
        print(f"{len(candidates) - len(todo)} already re-transcribed in an earlier run, {len(todo)} to go.")

    backup = args.csv.with_name(args.csv.stem + "_before_repass.csv")
    failed = replaced = recovered = 0
    if todo:
        new_log = not log_path.exists()
        progress = tx.Progress(len(todo))
        with log_path.open("a", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            if new_log:
                writer.writeheader()
            for row in todo:
                clip = tx.load_clip(args.input / row["file"], args.input, use_vad=False)
                try:
                    if clip.error:
                        raise RuntimeError(clip.error)
                    new = thorough_transcript(model, clip, args)
                except Exception as exc:
                    failed += 1
                    print(f"FAILED {row['file']}: {exc}", file=sys.stderr, flush=True)
                    progress.update(1)
                    continue

                old = row["text"]
                if pattern:
                    old_clean = remove_matching_sentences(old, pattern)
                    new_clean = remove_matching_sentences(new, pattern)
                    use_new = len(new_clean) > len(old_clean)
                    chosen = new_clean if use_new else old_clean
                    recovered += use_new
                else:
                    chosen = new if len(new) > len(old) else old
                changed = chosen != old

                # Update the CSV first, then the log: a crash in between only means redoing one file
                if changed:
                    if not backup.exists():
                        backup.write_bytes(args.csv.read_bytes())
                        print(f"Original saved as {backup}.", flush=True)
                    row["text"] = chosen
                    write_rows_atomic(args.csv, rows, tx.FIELDS)
                    replaced += 1
                writer.writerow({"file": row["file"], "old_chars": len(old), "new_chars": len(chosen),
                                 "replaced": int(changed), "text": chosen})
                fh.flush()
                progress.update(1, f"{row['file']}: {len(old)} -> {len(chosen)} chars")

    done = len(todo) - failed
    if pattern:
        print(f"Done. {replaced} of {done} files cleaned: {recovered} with a fresh transcript, "
              f"{replaced - recovered} by removing the matching sentences. {failed} failed.")
    else:
        print(f"Done. {replaced} of {done} re-transcribed files got a longer transcript, {failed} failed.")
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
