import csv
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import transcribe  # noqa: E402
from fake_model import SAMPLE_RATE, make_model, tone  # noqa: E402
from faster_whisper import BatchedInferencePipeline  # noqa: E402

FREQS = [float(f) for f in np.geomspace(150, 6000, 20).round()]


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def read_csv(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    files = [r["file"] for r in rows]
    assert len(files) == len(set(files)), "duplicate rows"
    return {r["file"]: r["text"] for r in rows}


def make_dataset(root: Path, freqs, seconds=None) -> dict[str, float]:
    """Write one tone per file with varied lengths and folders; return file -> frequency."""
    mapping = {}
    for i, f in enumerate(freqs):
        dur = seconds[i] if seconds else 1.0 + (i * 7) % 24  # 1 to 24 s
        rel = f"part{i % 3}/clip_{i:03d}.wav"
        write_wav(root / rel, tone(f, dur))
        mapping[rel] = f
    return mapping


def run(tmp_path, audio_dir, out_name, batch_size, freqs, bad=frozenset(), vad=False, loop=frozenset(), extra=(), stub=None):
    model, words = make_model(freqs, bad, loop)
    args = transcribe.parse_args([str(audio_dir), "-o", str(tmp_path / out_name), "-l", "ro", "-b", str(batch_size), "--workers", "3"]
                                 + ([] if vad else ["--no-vad"]) + list(extra))
    pipe = BatchedInferencePipeline(model=model) if batch_size > 1 else None
    failed = transcribe.run(args, model, pipe)
    if stub is not None:
        stub.append(model.model)
    return failed, words, model.model.calls


def test_batched_transcripts_land_on_the_right_files(tmp_path):
    audio = tmp_path / "audio"
    mapping = make_dataset(audio, FREQS)

    failed, words, calls = run(tmp_path, audio, "batched.csv", 4, FREQS)

    assert failed == 0
    assert max(calls) == 4, "clips should be decoded several at a time"
    result = read_csv(tmp_path / "batched.csv")
    assert result == {rel: words[f] for rel, f in mapping.items()}


def test_batched_matches_sequential(tmp_path):
    audio = tmp_path / "audio"
    make_dataset(audio, FREQS)

    run(tmp_path, audio, "batched.csv", 8, FREQS)
    _, _, calls = run(tmp_path, audio, "sequential.csv", 1, FREQS)

    assert set(calls) == {1}
    assert read_csv(tmp_path / "batched.csv") == read_csv(tmp_path / "sequential.csv")


def test_long_files_and_unreliable_clips_use_the_single_path(tmp_path, capsys):
    freqs = FREQS[:6]
    seconds = [5, 45, 5, 5, 12, 3]  # one file longer than Whisper's 30 s window
    audio = tmp_path / "audio"
    mapping = make_dataset(audio, freqs, seconds)

    failed, words, _ = run(tmp_path, audio, "out.csv", 4, freqs, bad={freqs[3]})

    assert failed == 0
    result = read_csv(tmp_path / "out.csv")
    for rel, f in mapping.items():
        assert set(result[rel].split()) == {words[f]}
    assert "1 re-checked one by one" in capsys.readouterr().out


def test_silence_broken_files_and_resume(tmp_path, capsys):
    audio = tmp_path / "audio"
    write_wav(audio / "silent.wav", np.zeros(SAMPLE_RATE * 3, dtype=np.float32))
    write_wav(audio / "empty.wav", np.zeros(0, dtype=np.float32))
    (audio / "broken.wav").write_bytes(b"not audio")

    failed, _, _ = run(tmp_path, audio, "out.csv", 4, FREQS[:2], vad=True)
    assert failed == 1
    assert read_csv(tmp_path / "out.csv") == {"silent.wav": "", "empty.wav": ""}

    # Second run only retries the broken file and writes no duplicates
    failed, _, calls = run(tmp_path, audio, "out.csv", 4, FREQS[:2], vad=True)
    assert failed == 1
    assert calls == []
    assert "2 already done, 1 to go" in capsys.readouterr().out
    assert len(read_csv(tmp_path / "out.csv")) == 2


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("extra", [(), ("--no-fallback",)])
def test_repetition_loops_are_cut_short(tmp_path, capsys, batch_size, extra):
    freqs = FREQS[:5]
    seconds = [10, 4, 20, 6, 8]
    audio = tmp_path / "audio"
    mapping = make_dataset(audio, freqs, seconds)
    looping = freqs[2]  # the 20 s clip

    stubs = []
    failed, words, _ = run(tmp_path, audio, "out.csv", batch_size, freqs, loop={looping}, extra=extra, stub=stubs)

    assert failed == 0
    result = read_csv(tmp_path / "out.csv")
    for rel, f in mapping.items():
        n = len(result[rel].split())
        if f == looping:
            assert 1 < n <= 12 * 20 + transcribe.TOKEN_HEADROOM
        else:
            assert result[rel] == words[f]
    assert max(stubs[0].max_lengths) <= 448
    assert min(stubs[0].max_lengths) < 448, "short clips should get a token cap"
    if batch_size == 1:
        assert "1 transcripts were cut short" in capsys.readouterr().out


def test_token_cap_can_be_disabled_and_skips_long_files():
    args = transcribe.parse_args(["x", "--max-tokens-per-second", "0"])
    assert transcribe.token_cap(10, args) is None
    args = transcribe.parse_args(["x"])
    assert transcribe.token_cap(2, args) == 12 * 2 + transcribe.TOKEN_HEADROOM
    assert transcribe.token_cap(29.9, args) <= transcribe.MAX_TOKENS
    assert transcribe.token_cap(45, args) is None


def test_prefetch_keeps_order_and_bounds_memory():
    import threading

    in_flight, peak, lock = 0, 0, threading.Lock()

    def work(i):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        return i

    out = []
    for i in transcribe.prefetch(range(100), work, workers=4, window=5):
        out.append(i)
        with lock:
            in_flight -= 1
    assert out == list(range(100))
    assert peak <= 6


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
