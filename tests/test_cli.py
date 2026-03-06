"""Tests for CLI argument parsing, formatting, and error handling."""

import shlex
from pathlib import Path
from urllib.parse import unquote, urlparse

import dask
import numpy as np
import pytest
from PIL import Image

from grape.cli import (
    DEFAULT_PROMPT_ENSEMBLE,
    _apply_excluded_keywords,
    _format_html,
    _format_results,
    _ScannedImage,
    _show_in_webview,
    main,
    parse_keywords,
)
from grape.search import ScoredImage


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
        ScoredImage(path=Path("/a/dog.jpg"), scores={"dog": 0.85}, score=0.85),
        ScoredImage(path=Path("/a/cat.jpg"), scores={"dog": 0.42}, score=0.42),
    ]
    output = _format_results(results, verbose=False)
    lines = output.strip().split("\n")
    assert len(lines) == 2
    assert "0.850" in lines[0]
    assert "dog.jpg" in lines[0]
    assert "0.420" in lines[1]


def test_format_results_verbose_shows_breakdown():
    results = [
        ScoredImage(
            path=Path("/a/img.jpg"),
            scores={"dog": 0.8, "cat": 0.4}, score=0.6,
        ),
    ]
    output = _format_results(results, verbose=True)
    assert "dog: 0.800" in output
    assert "cat: 0.400" in output


def test_apply_excluded_keywords_adjusts_score_and_labels():
    results = [
        ScoredImage(path=Path("/a/img.jpg"), scores={"dog": 0.9, "cat": 0.2}),
    ]
    _apply_excluded_keywords(results, ["dog"], ["cat"])
    assert results[0].score == pytest.approx(0.7)
    assert results[0].scores["dog"] == pytest.approx(0.9)
    assert results[0].scores["not:cat"] == pytest.approx(0.2)


def test_apply_excluded_keywords_with_empty_include():
    results = [
        ScoredImage(path=Path("/a/img.jpg"), scores={"cat": 0.2}),
    ]
    _apply_excluded_keywords(results, [], ["cat"])
    assert results[0].score == pytest.approx(-0.2)
    assert results[0].scores["not:cat"] == pytest.approx(0.2)


def test_format_html_embeds_images(tmp_path):
    image_path = tmp_path / "my photo.jpg"
    Image.new("RGB", (1, 1)).save(image_path, format="JPEG")
    results = [
        ScoredImage(path=image_path, scores={"dog": 0.75}, score=0.75),
    ]

    html_doc = _format_html(results, ["dog"])
    assert "max-width: 100%" in html_doc
    assert "<img " in html_doc
    assert image_path.resolve().as_uri() in html_doc
    assert str(image_path.resolve()) in html_doc
    assert "0.750" in html_doc
    assert html_doc.index("0.750") < html_doc.index("<img ")


def test_format_html_verbose_shows_breakdown(tmp_path):
    image_path = tmp_path / "my photo.jpg"
    Image.new("RGB", (1, 1)).save(image_path, format="JPEG")
    results = [
        ScoredImage(path=image_path, scores={"dog": 0.75, "cat": 0.25}, score=0.75),
    ]

    html_doc = _format_html(results, ["dog", "cat"])
    assert "score: 0.750" in html_doc
    assert "dog: 0.750" in html_doc
    assert "cat: 0.250" in html_doc


def test_format_html_shows_original_path_text(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    image_path = Path("relative name.jpg")
    Image.new("RGB", (1, 1)).save(image_path, format="JPEG")
    results = [
        ScoredImage(path=image_path, scores={"dog": 0.75}, score=0.75),
    ]

    html_doc = _format_html(results, ["dog"])
    assert "<p class=\"meta path\">relative name.jpg</p>" in html_doc
    assert str((tmp_path / image_path).as_uri()) in html_doc


def test_show_in_webview_calls_create_window_and_start(monkeypatch):
    calls: list[tuple] = []
    loaded_html: list[str] = []
    class FakeWebview:
        def __init__(self):
            self.settings = {"OPEN_DEVTOOLS_IN_DEBUG": True}

        def create_window(self, *args, **kwargs):
            calls.append(("create", args, kwargs))
            html_path = Path(unquote(urlparse(kwargs["url"]).path))
            loaded_html.append(html_path.read_text(encoding="utf-8"))

        def start(self, **kwargs):
            calls.append(("start", kwargs))

    fake_webview = FakeWebview()
    monkeypatch.setattr("grape.cli._get_webview", lambda: fake_webview)
    _show_in_webview("<html>ok</html>")
    assert calls[0][0] == "create"
    assert calls[0][1] == ("grape results",)
    assert "url" in calls[0][2]
    assert calls[0][2]["width"] == 1280
    assert calls[0][2]["height"] == 900
    assert calls[0][2]["maximized"] is True
    assert calls[0][2]["resizable"] is True
    assert calls[0][2]["text_select"] is True
    assert calls[0][2]["zoomable"] is True
    assert calls[0][2]["min_size"] == (480, 320)
    assert loaded_html == ["<html>ok</html>"]
    assert calls[1] == ("start", {"debug": True})
    assert fake_webview.settings["OPEN_DEVTOOLS_IN_DEBUG"] is False


# --- argument validation ---

def test_bad_model_format(monkeypatch):
    _, err, code = run_main(["--model", "noslash", "dog", "x.jpg"], monkeypatch)
    assert code != 0
    assert "model_name/pretrained" in err


def test_empty_keywords_errors(tmp_path, monkeypatch):
    image_path = tmp_path / "empty keywords.jpg"
    image_path.write_bytes(b"unused")
    _, err, code = run_main(["-q", ",,,", str(image_path)], monkeypatch)
    assert code != 0
    assert "keyword" in err.lower()


def test_empty_exclude_keywords_errors(tmp_path, monkeypatch):
    image_path = tmp_path / "empty exclude.jpg"
    image_path.write_bytes(b"unused")
    _stub_pipeline(monkeypatch)
    out, _, code = run_main(
        ["-q", "-s", "-x", ",,,", "dog", str(image_path)],
        monkeypatch,
    )
    assert code == 0
    assert "0.750" in out


def test_both_keyword_lists_empty_is_error(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    image_path.write_bytes(b"unused")
    _, err, code = run_main(
        ["-q", "-x", ",,,", ",,,", str(image_path)],
        monkeypatch,
    )
    assert code != 0
    assert "keyword" in err.lower()


def test_count_short_flag_is_rejected(monkeypatch):
    _, err, code = run_main(["-c", "dog", "x.jpg"], monkeypatch)
    assert code != 0
    assert "unrecognized arguments: -c" in err


def test_count_long_flag_is_rejected(monkeypatch):
    _, err, code = run_main(["--count", "dog", "x.jpg"], monkeypatch)
    assert code != 0
    assert "unrecognized arguments: --count" in err


def test_view_rejects_print0(monkeypatch):
    _, err, code = run_main(["--view", "-print0", "dog", "x.jpg"], monkeypatch)
    assert code != 0
    assert "not allowed with argument" in err
    assert "--view" in err
    assert "-print0" in err


# --- error paths ---

def test_nonexistent_path_errors(monkeypatch):
    _stub_pipeline(monkeypatch)
    _, err, code = run_main(
        ["-q", "dog", "/tmp/grape_nonexistent_path_xyz"],
        monkeypatch,
    )
    assert code != 0
    assert "No such file" in err


def test_dir_without_r_warns(tmp_path, monkeypatch):
    """Passing a directory without -r warns and finds no images."""
    Image.new("RGB", (1, 1)).save(tmp_path / "img.jpg", format="JPEG")
    _stub_pipeline(monkeypatch)
    _, err, code = run_main(["-q", "dog", str(tmp_path)], monkeypatch)
    assert "Is a directory" in err


def test_no_images_exits(tmp_path, monkeypatch):
    """Empty directory with -r exits with an error."""
    _stub_pipeline(monkeypatch)
    _, err, code = run_main(["-q", "-r", "dog", str(tmp_path)], monkeypatch)
    assert code != 0
    assert "no images found" in err


def _stub_pipeline(monkeypatch, score=0.75):
    """Stub the delayed pipeline tasks so tests don't load the real model.

    Replaces _import_model_module, _load_model, _encode_keywords, and
    _resolve_model_id with plain @dask.delayed stubs that return
    lightweight fake objects. _score_all is replaced with a version that
    scores each scanned image using the fixed score value.
    """
    import grape.cli as cli_mod

    @dask.delayed
    def _fake_import_model_module(model_name):
        return object()

    @dask.delayed
    def _fake_load_model(model_module, model_name, pretrained, quiet):
        return object()

    @dask.delayed
    def _fake_encode_keywords(model, score_keywords, prompt_templates):
        return object()

    @dask.delayed
    def _fake_resolve_and_index_cache(model_module, model_name, pretrained, cache):
        return (None, None)

    @dask.delayed
    def _fake_prepare_cached_embeddings(scan_result, cache_context):
        items, scan_done = scan_result
        return (None, [], items, scan_done)

    @dask.delayed
    def _fake_score_all(
        prepared, model, score_keywords, text_emb,
        cache, quiet,
    ):
        _image_emb, _cached_items, uncached_items, scan_done = prepared
        results = [
            ScoredImage(path=item.path, score=score,
                        scores={kw: score for kw in score_keywords})
            for item in uncached_items
        ]
        return results, scan_done

    monkeypatch.setattr(cli_mod, "_import_model_module", _fake_import_model_module)
    monkeypatch.setattr(cli_mod, "_load_model", _fake_load_model)
    monkeypatch.setattr(cli_mod, "_encode_keywords", _fake_encode_keywords)
    monkeypatch.setattr(
        cli_mod, "_resolve_and_index_cache", _fake_resolve_and_index_cache,
    )
    monkeypatch.setattr(
        cli_mod, "_prepare_cached_embeddings", _fake_prepare_cached_embeddings,
    )
    monkeypatch.setattr(cli_mod, "_score_all", _fake_score_all)


def test_default_output_shell_quotes_paths(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    image_path.write_bytes(b"unused")
    _stub_pipeline(monkeypatch)
    out, _, code = run_main(["-q", "dog", str(image_path)], monkeypatch)
    assert code == 0
    assert out == f"{shlex.quote(str(image_path))}\n"


def test_print0_outputs_raw_nul_terminated_paths(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    image_path.write_bytes(b"unused")
    _stub_pipeline(monkeypatch)
    out, _, code = run_main(["-q", "-print0", "dog", str(image_path)], monkeypatch)
    assert code == 0
    assert out == f"{image_path}\0"


def test_scores_output_shell_quotes_paths(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    image_path.write_bytes(b"unused")
    _stub_pipeline(monkeypatch)
    out, _, code = run_main(["-q", "-s", "dog", str(image_path)], monkeypatch)
    assert code == 0
    assert shlex.quote(str(image_path)) in out


def test_exclude_keywords_adjusts_output_score(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    image_path.write_bytes(b"unused")

    # Custom stub that returns different scores per keyword
    import grape.cli as cli_mod

    @dask.delayed
    def _fake_score_all(
        prepared, model, score_keywords, text_emb,
        cache, quiet,
    ):
        _image_emb, _cached_items, uncached_items, scan_done = prepared
        results = [
            ScoredImage(path=item.path,
                        scores={"dog": 0.9, "cat": 0.2})
            for item in uncached_items
        ]
        return results, scan_done

    _stub_pipeline(monkeypatch)
    monkeypatch.setattr(cli_mod, "_score_all", _fake_score_all)

    out, _, code = run_main(
        ["-q", "-s", "-x", "cat", "dog", str(image_path)],
        monkeypatch,
    )
    assert code == 0
    assert "0.700" in out
    assert shlex.quote(str(image_path)) in out


def test_exclude_alias_anti_is_rejected(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    image_path.write_bytes(b"unused")
    _, err, code = run_main(
        ["-q", "-s", "--anti", "cat", "dog", str(image_path)],
        monkeypatch,
    )
    assert code != 0
    assert "unrecognized arguments: --anti" in err


def test_exclude_verbose_shows_not_keyword(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    image_path.write_bytes(b"unused")

    import grape.cli as cli_mod

    @dask.delayed
    def _fake_score_all(
        prepared, model, score_keywords, text_emb,
        cache, quiet,
    ):
        _image_emb, _cached_items, uncached_items, scan_done = prepared
        results = [
            ScoredImage(path=item.path,
                        scores={"dog": 0.9, "cat": 0.2})
            for item in uncached_items
        ]
        return results, scan_done

    _stub_pipeline(monkeypatch)
    monkeypatch.setattr(cli_mod, "_score_all", _fake_score_all)

    out, _, code = run_main(
        ["-q", "-v", "-x", "cat", "dog", str(image_path)],
        monkeypatch,
    )
    assert code == 0
    assert "not:cat: 0.200" in out


def test_ensemble_prompts_uses_default_template_set(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    image_path.write_bytes(b"unused")
    captured: dict[str, object] = {}

    import grape.cli as cli_mod

    @dask.delayed
    def _fake_encode_keywords(model, score_keywords, prompt_templates):
        captured["prompt_templates"] = prompt_templates
        return object()

    _stub_pipeline(monkeypatch)
    monkeypatch.setattr(cli_mod, "_encode_keywords", _fake_encode_keywords)

    out, _, code = run_main(
        ["-q", "dog", str(image_path)],
        monkeypatch,
    )
    assert code == 0
    assert out == f"{shlex.quote(str(image_path))}\n"
    assert captured["prompt_templates"] == DEFAULT_PROMPT_ENSEMBLE
    assert len(DEFAULT_PROMPT_ENSEMBLE) == 3


def test_ensemble_prompts_custom_templates_override_default(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    image_path.write_bytes(b"unused")
    captured: dict[str, object] = {}

    import grape.cli as cli_mod

    @dask.delayed
    def _fake_encode_keywords(model, score_keywords, prompt_templates):
        captured["prompt_templates"] = prompt_templates
        return object()

    _stub_pipeline(monkeypatch)
    monkeypatch.setattr(cli_mod, "_encode_keywords", _fake_encode_keywords)

    out, _, code = run_main(
        [
            "-q",
            "--ensemble-prompts",
            "one {},two {}",
            "dog",
            str(image_path),
        ],
        monkeypatch,
    )
    assert code == 0
    assert out == f"{shlex.quote(str(image_path))}\n"
    assert captured["prompt_templates"] == ["one {}", "two {}"]


def test_ensemble_prompts_template_without_placeholder_errors(monkeypatch):
    _, err, code = run_main(
        ["--ensemble-prompts", "bad template", "dog", "x.jpg"],
        monkeypatch,
    )
    assert code != 0
    assert "must include '{}'" in err


def test_view_calls_webview_with_html(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    image_path.write_bytes(b"unused")
    _stub_pipeline(monkeypatch)
    captured: list[str] = []
    monkeypatch.setattr("grape.cli._show_in_webview", captured.append)
    out, _, code = run_main(["-q", "--view", "dog", str(image_path)], monkeypatch)
    assert code == 0
    assert out == ""
    assert len(captured) == 1
    html_doc = captured[0]
    assert "<img " in html_doc
    assert image_path.resolve().as_uri() in html_doc
    assert str(image_path.resolve()) in html_doc
    assert html_doc.index("0.750") < html_doc.index("<img ")


def test_score_all_uses_in_memory_cache_index():
    from grape.cli import (
        _prepare_cached_embeddings,
        _ScanDone,
        _score_all,
    )

    class _NoDbCache:
        def get_many_for_paths(self, *_args, **_kwargs):
            raise AssertionError("DB batch lookup should not be called")

    items = [
        _ScannedImage(
            path=Path("/tmp/a.jpg"),
            path_key="/tmp/a.jpg",
            file_stat="stat-a",
        )
    ]
    cached_index = {
        ("/tmp/a.jpg", "stat-a"): np.array([1.0, 0.0], dtype=np.float32)
    }
    text_emb = np.array([[1.0, 0.0]], dtype=np.float32)
    scan_result = (items, _ScanDone(image_count=1))
    cache_context = ("model-id", cached_index)

    prepared = _prepare_cached_embeddings(
        scan_result, cache_context,
    ).compute()

    results, _done = _score_all(
        prepared, object(), ["dog"], text_emb,
        _NoDbCache(), True,
    ).compute()

    assert len(results) == 1
    assert results[0].path == Path("/tmp/a.jpg")
    assert results[0].scores["dog"] == pytest.approx(1.0)
    assert results[0].score == pytest.approx(1.0)


def test_scan_files_includes_cache_metadata(tmp_path):
    """_scan_files returns items with cache metadata for plain files."""
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"test")

    from grape.cli import _scan_files

    # _scan_files is @dask.delayed, so .compute() to get the result
    items, done = _scan_files(
        [str(image_path)], False, None,
    ).compute()

    assert len(items) == 1
    item = items[0]
    assert isinstance(item, _ScannedImage)
    assert item.path == image_path
    assert item.path_key == str(image_path.resolve())
    assert item.file_stat is not None
    assert item.file_stat.startswith("[")
    assert item.file_stat.endswith("]")
    assert done.image_count == 1
    assert done.error_message is None


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
