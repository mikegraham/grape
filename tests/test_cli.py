"""Tests for CLI argument parsing, formatting, and error handling."""

from pathlib import Path

import pytest
from PIL import Image

from grape.cli import _format_results, main, parse_keywords


def run_main(args, monkeypatch):
    """Call main() in-process so coverage is tracked.

    Returns (stdout, stderr, exit_code).
    """
    import io

    monkeypatch.setattr("sys.argv", ["grape", *args])
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)
    try:
        main()
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    return out.getvalue(), err.getvalue(), code


# --- parse_keywords ---

def test_parse_keywords_splits_on_comma():
    assert parse_keywords("cat,dog") == ["cat", "dog"]


def test_parse_keywords_strips_whitespace():
    assert parse_keywords(" cat , dog ") == ["cat", "dog"]


def test_parse_keywords_preserves_phrases():
    assert parse_keywords("golden retriever,tennis ball") == [
        "golden retriever",
        "tennis ball",
    ]


def test_parse_keywords_single():
    assert parse_keywords("single") == ["single"]


def test_parse_keywords_empty():
    assert parse_keywords(",,,") == []


# --- _format_results ---

def test_format_results_shows_score_and_path():
    results = [
        {"score": 0.85, "path": Path("/a/dog.jpg"), "scores": {"dog": 0.85}},
        {"score": 0.42, "path": Path("/a/cat.jpg"), "scores": {"dog": 0.42}},
    ]
    output = _format_results(results, verbose=False)
    lines = output.strip().split("\n")
    assert len(lines) == 2
    assert "0.850" in lines[0]
    assert "dog.jpg" in lines[0]
    assert "0.420" in lines[1]


def test_format_results_verbose_shows_breakdown():
    results = [
        {
            "score": 0.6,
            "path": Path("/a/img.jpg"),
            "scores": {"dog": 0.8, "cat": 0.4},
        },
    ]
    output = _format_results(results, verbose=True)
    assert "dog: 0.800" in output
    assert "cat: 0.400" in output


# --- argument validation ---

def test_bad_model_format(monkeypatch):
    _, err, code = run_main(["--model", "noslash", "dog", "x.jpg"], monkeypatch)
    assert code != 0
    assert "model_name/pretrained" in err


def test_empty_keywords_errors(monkeypatch):
    _, err, code = run_main([",,,", "x.jpg"], monkeypatch)
    assert code != 0


# --- error paths ---

def test_nonexistent_path_errors(monkeypatch):
    _, err, code = run_main(
        ["-q", "dog", "/tmp/grape_nonexistent_path_xyz"],
        monkeypatch,
    )
    assert code != 0
    assert "No such file" in err


def test_dir_without_r_warns(tmp_path, monkeypatch):
    """Passing a directory without -r warns and finds no images."""
    Image.new("RGB", (1, 1)).save(tmp_path / "img.jpg", format="JPEG")
    _, err, code = run_main(["-q", "dog", str(tmp_path)], monkeypatch)
    assert "Is a directory" in err


def test_no_images_exits(tmp_path, monkeypatch):
    """Empty directory with -r exits with an error."""
    _, err, code = run_main(["-q", "-r", "dog", str(tmp_path)], monkeypatch)
    assert code != 0
    assert "no images found" in err


# --- end-to-end with real model (slow) ---

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.slow
def test_e2e_single_file(monkeypatch):
    """Score a single file through the full CLI pipeline."""
    out, _, code = run_main(["-q", "dog", str(FIXTURES / "dog.jpg")], monkeypatch)
    assert code == 0
    assert "dog.jpg" in out


@pytest.mark.slow
def test_e2e_recursive_dir(monkeypatch):
    """Score a directory recursively."""
    out, _, code = run_main(["-q", "-r", "dog", str(FIXTURES)], monkeypatch)
    assert code == 0
    assert "dog.jpg" in out


@pytest.mark.slow
def test_e2e_scores_mode(monkeypatch):
    """--scores prints 'score  path' lines."""
    out, _, code = run_main(
        ["-q", "-s", "dog", str(FIXTURES / "dog.jpg")], monkeypatch,
    )
    assert code == 0
    parts = out.strip().split()
    assert len(parts) >= 2
    float(parts[0])  # score should be a float


@pytest.mark.slow
def test_e2e_count_mode(monkeypatch):
    """--count prints the number of matching images."""
    out, _, code = run_main(
        ["-q", "-c", "-r", "dog", str(FIXTURES)], monkeypatch,
    )
    assert code == 0
    assert int(out.strip()) > 0


@pytest.mark.slow
def test_e2e_top_n(monkeypatch):
    """--top 1 limits output to a single result."""
    out, _, code = run_main(
        ["-q", "--top", "1", "-r", "dog", str(FIXTURES)], monkeypatch,
    )
    assert code == 0
    assert len(out.strip().split("\n")) == 1


@pytest.mark.slow
def test_e2e_threshold_filters(monkeypatch):
    """--threshold 1.0 filters out everything."""
    out, err, code = run_main(
        ["-q", "-t", "1.0", "dog", str(FIXTURES / "dog.jpg")], monkeypatch,
    )
    assert code == 0
    assert "no images above threshold" in err


@pytest.mark.slow
def test_e2e_threshold_count(monkeypatch):
    """--threshold 1.0 with --count prints 0."""
    out, _, code = run_main(
        ["-q", "-c", "-t", "1.0", "dog", str(FIXTURES / "dog.jpg")], monkeypatch,
    )
    assert code == 0
    assert out.strip() == "0"


@pytest.mark.slow
def test_e2e_verbose(monkeypatch):
    """--verbose shows per-keyword breakdown."""
    out, _, code = run_main(
        ["-q", "-v", "dog,cat", str(FIXTURES / "dog.jpg")], monkeypatch,
    )
    assert code == 0
    assert "dog:" in out
    assert "cat:" in out


@pytest.mark.slow
def test_e2e_status_message(monkeypatch):
    """Without -q, prints image count and keywords to stderr."""
    _, err, code = run_main(
        ["-r", "dog", str(FIXTURES)], monkeypatch,
    )
    assert code == 0
    assert "image" in err
    assert "dog" in err


@pytest.mark.slow
def test_e2e_with_cache(tmp_path, monkeypatch):
    """--cache creates a DB and caches image embeddings."""
    db = tmp_path / "test.db"
    out, _, code = run_main(
        ["-q", "--cache", str(db), "dog", str(FIXTURES / "dog.jpg")],
        monkeypatch,
    )
    assert code == 0
    assert "dog.jpg" in out
    assert db.exists()
