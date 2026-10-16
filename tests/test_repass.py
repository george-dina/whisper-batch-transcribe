import csv
import re
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


def run(audio, csv_path, freqs, *extra, loop=frozenset(), quiet=frozenset()):
    model, words = make_model(freqs, loop_freqs=loop, quiet_freqs=quiet)
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


def test_no_skip_recovers_windows_whisper_thought_were_silent(tmp_path):
    freqs, audio, csv_path, rows = setup(tmp_path)
    quiet = {freqs[1]}

    run(audio, csv_path, freqs, quiet=quiet)
    assert read(csv_path)[1]["text"] == "", "by default the quiet window is skipped"

    _, words, stub = run(audio, csv_path, freqs, "--no-skip", "--fresh", quiet=quiet)
    out = read(csv_path)
    assert out[1]["text"] == words[freqs[1]]
    assert out[3]["text"] == words[freqs[3]], "earlier improvements must be kept"
    assert read(tmp_path / "t_before_repass.csv") == rows, "the backup must stay the original"


def test_short_files_are_ignored(tmp_path):
    rows = [{"file": "a.wav", "duration_s": "3.00", "text": ""},
            {"file": "b.wav", "duration_s": "12.00", "text": ""}]
    assert [r["file"] for r in repass.find_candidates(rows, 6, 5)] == ["b.wav"]


LIKE = r"\blike\b"


def setup_like(tmp_path):
    freqs = FREQS[:4]
    audio = tmp_path / "audio"
    for i, f in enumerate(freqs):
        write_wav(audio / f"c{i}.wav", tone(f, 20))
    rows = [
        # hallucination in the middle of real text: the real text must survive
        {"file": "c0.wav", "duration_s": "20.00", "text": "Bună ziua, doamna doctor. Nu uitați să dați like! Mulțumesc frumos."},
        # nothing but the hallucination: the fresh transcript should win
        {"file": "c1.wav", "duration_s": "20.00", "text": "Nu uitați să dați LIKE și subscribe."},
        # no match: must not be touched or transcribed
        {"file": "c2.wav", "duration_s": "20.00", "text": "Alo, bună ziua, vă sun de la Totem. " * 8},
        # "like" inside another word is not a match
        {"file": "c3.wav", "duration_s": "20.00", "text": "Ne vedem la Likerty, la revedere. " * 8},
    ]
    csv_path = tmp_path / "t.csv"
    write_csv(csv_path, rows)
    return freqs, audio, csv_path, rows


def test_match_mode_removes_the_phrase_and_keeps_the_best_text(tmp_path, capsys):
    freqs, audio, csv_path, rows = setup_like(tmp_path)

    failed, words, stub = run(audio, csv_path, freqs, "--match", LIKE)

    assert failed == 0
    assert len(stub.calls) == 2, "only the two matching files should be transcribed"
    out = read(csv_path)
    assert out[0]["text"] == "Bună ziua, doamna doctor. Mulțumesc frumos."
    assert out[1]["text"] == words[freqs[1]]
    assert out[2]["text"] == rows[2]["text"] and out[3]["text"] == rows[3]["text"]
    assert not any(repass.compile_pattern(LIKE).search(r["text"]) for r in out)
    assert read(tmp_path / "t_before_repass.csv") == rows
    assert "2 of 2 files cleaned: 1 with a fresh transcript, 1 by removing" in capsys.readouterr().out


def test_match_dry_run_shows_snippets(tmp_path, capsys):
    freqs, audio, csv_path, rows = setup_like(tmp_path)

    args = repass.parse_args([str(audio), "-c", str(csv_path), "--match", LIKE, "--dry-run"])
    assert repass.run(args, model=None) == 0

    listing = read(tmp_path / "t_repass_candidates.csv")
    assert [r["file"] for r in listing] == ["c0.wav", "c1.wav"]
    assert "like" in listing[0]["snippet"].lower()
    assert "2 of 4 files match" in capsys.readouterr().out
    assert read(csv_path) == rows


def test_later_passes_never_bring_back_removed_text(tmp_path):
    freqs, audio, csv_path, rows = setup_like(tmp_path)
    run(audio, csv_path, freqs, "--min-cps", "100")  # length pass first: touches every row
    run(audio, csv_path, freqs, "--match", LIKE)
    cleaned = read(csv_path)

    run(audio, csv_path, freqs, "--min-cps", "100")  # resumed length pass

    assert read(csv_path) == cleaned


def test_builtin_hallucination_list(tmp_path, capsys):
    freqs = FREQS[:5]
    audio = tmp_path / "audio"
    for i, f in enumerate(freqs):
        write_wav(audio / f"c{i}.wav", tone(f, 20))
    real = "Bună ziua, doamna doctor, vă sun în legătură cu programarea. "
    rows = [
        {"file": "c0.wav", "duration_s": "20.00", "text": real + "Mulțumesc pentru vizionare!"},
        {"file": "c1.wav", "duration_s": "20.00", "text": real + "Subtitrarea realizată de comunitatea Amara.org."},
        {"file": "c2.wav", "duration_s": "20.00", "text": real + "Nu uitați să vă abonați la canalul nostru."},
        {"file": "c3.wav", "duration_s": "20.00", "text": real + "Vă mulțumesc frumos. La revedere."},  # legitimate
        {"file": "c4.wav", "duration_s": "20.00", "text": real + "Puteți accesa www.otm.com.ro, secțiunea Confidențialitate."},  # legitimate
    ]
    csv_path = tmp_path / "t.csv"
    write_csv(csv_path, rows)

    failed, _, stub = run(audio, csv_path, freqs, "--hallucinations")

    assert failed == 0
    assert len(stub.calls) == 3
    out = read(csv_path)
    for r in out[:3]:
        assert r["text"] == real.strip()
    assert out[3] == rows[3] and out[4] == rows[4]
    report = capsys.readouterr().out
    assert "3 of 5 files match known Whisper phrases" in report
    assert "pentru vizionare: 1 files" in report


def test_hallucination_patterns_skip_normal_call_phrases():
    normal = [
        "Vă mulțumesc frumos, o zi bună.",
        "Puteți să ne scrieți un e-mail oricând.",
        "Am văzut programul pe site-ul nostru.",
        "Ne place colaborarea cu dumneavoastră.",
        "Abonamentul se reînnoiește lunar.",
    ]
    pattern = re.compile("|".join(f"(?:{p})" for p in repass.HALLUCINATIONS.values()), re.IGNORECASE)
    assert not [t for t in normal if pattern.search(t)]
