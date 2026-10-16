import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import repass  # noqa: E402
from fake_model import make_model, tone  # noqa: E402
from test_transcribe import FREQS, write_wav  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=("file", "duration_s", "text"))
        w.writeheader()
        w.writerows(rows)


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def setup(tmp_path):
    """Five 20 s files: two with healthy transcripts, three suspiciously short."""
    freqs = FREQS[:5]
    audio = tmp_path / "audio"
    for i, f in enumerate(freqs):
        write_wav(audio / f"c{i}.wav", tone(f, 20))
    good = "x" * 300  # 15 chars/s
    rows = [
        {"file": "c0.wav", "duration_s": "20.00", "text": good},
        {"file": "c1.wav", "duration_s": "20.00", "text": ""},  # empty: should be replaced
        {"file": "c2.wav", "duration_s": "20.00", "text": good},
        {"file": "c3.wav", "duration_s": "20.00", "text": "ab"},  # short: should be replaced
        {"file": "c4.wav", "duration_s": "20.00", "text": "abcdefghijk"},  # short, but longer than the new text
    ]
    csv_path = tmp_path / "t.csv"
    write_csv(csv_path, rows)
    return freqs, audio, csv_path, rows


def run(audio, csv_path, freqs, *extra, loop=frozenset()):
    model, words = make_model(freqs, loop_freqs=loop)
    args = repass.parse_args([str(audio), "-c", str(csv_path), "-l", "ro", "--min-cps", "6", *extra])
    return repass.run(args, model), words, model.model


def test_only_short_rows_are_retranscribed_and_improved(tmp_path):
    freqs, audio, csv_path, rows = setup(tmp_path)

    failed, words, stub = run(audio, csv_path, freqs)

    assert failed == 0
    out = read(csv_path)
    assert [r["file"] for r in out] == [r["file"] for r in rows], "row order must be kept"
    assert out[0]["text"] == rows[0]["text"] and out[2]["text"] == rows[2]["text"]
    assert out[1]["text"] == words[freqs[1]]
    assert out[3]["text"] == words[freqs[3]]
    assert out[4]["text"] == "abcdefghijk", "a shorter new transcript must not replace the old one"
    assert len(stub.calls) == 3, "only the three short rows should be transcribed"
    assert (tmp_path / "t_before_repass.csv").read_bytes()  # backup kept
    assert read(tmp_path / "t_before_repass.csv") == rows


def test_resume_skips_finished_files(tmp_path, capsys):
    freqs, audio, csv_path, _ = setup(tmp_path)
    run(audio, csv_path, freqs)
    first = read(csv_path)

    failed, _, stub = run(audio, csv_path, freqs)

    assert failed == 0
    # Rows that were improved are no longer short; c4 is still short but already logged
    assert stub.calls == []
    assert read(csv_path) == first
    assert "already re-transcribed" in capsys.readouterr().out


def test_dry_run_lists_candidates_without_transcribing(tmp_path):
    freqs, audio, csv_path, rows = setup(tmp_path)

    args = repass.parse_args([str(audio), "-c", str(csv_path), "--dry-run"])
    assert repass.run(args, model=None) == 0

    listing = read(tmp_path / "t_repass_candidates.csv")
    assert [r["file"] for r in listing] == ["c1.wav", "c3.wav", "c4.wav"]
    assert read(csv_path) == rows


def test_repetition_loops_never_win(tmp_path):
    freqs, audio, csv_path, rows = setup(tmp_path)

    run(audio, csv_path, freqs, loop={freqs[3]})

    out = read(csv_path)
    assert out[3]["text"] == "ab", "a looping transcript must not replace the old text"
    assert out[1]["text"] != ""


def test_short_files_are_ignored(tmp_path):
    rows = [{"file": "a.wav", "duration_s": "3.00", "text": ""},
            {"file": "b.wav", "duration_s": "12.00", "text": ""}]
    assert [r["file"] for r in repass.find_candidates(rows, 6, 5)] == ["b.wav"]
