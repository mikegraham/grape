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


def test_parse_keywords_custom_separator():
    assert parse_keywords("cat;dog", separator=";") == ["cat", "dog"]


def test_parse_keywords_empty_separator_no_split():
    """Empty separator treats the whole string as one keyword."""
    assert parse_keywords("red, white, and blue", separator="") == [
        "red, white, and blue",
    ]


def test_parse_keywords_empty_separator_blank_input():
    assert parse_keywords("  ", separator="") == []


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


def test_format_html_structure(tmp_path):
    """Rendered HTML parses correctly and has the right structural bones."""
    from html.parser import HTMLParser

    images = []
    for i in range(3):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (1, 1)).save(p, format="JPEG")
        images.append(p)
    results = [
        ScoredImage(path=p, scores={"dog": 0.5 + i * 0.1}, score=0.5 + i * 0.1)
        for i, p in enumerate(images)
    ]

    html_doc = _format_html(results, ["dog"])

    # Parses without error and collect tag counts.
    tags: dict[str, int] = {}

    class Counter(HTMLParser):
        def handle_starttag(self, tag, attrs):
            tags[tag] = tags.get(tag, 0) + 1
        def handle_decl(self, decl):
            tags["!doctype"] = tags.get("!doctype", 0) + 1

    Counter().feed(html_doc)

    assert tags.get("!doctype", 0) == 1, "missing doctype"
    assert tags.get("html", 0) == 1
    assert tags.get("head", 0) == 1
    assert tags.get("body", 0) == 1
    assert tags.get("style", 0) == 1
    assert tags.get("img", 0) == len(results), "one <img> per result"
    assert tags.get("meta", 0) >= 1, "missing <meta> tags"


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
    _, err, code = run_main(["--model", "noslash", "-k", "dog", "x.jpg"], monkeypatch)
    assert code != 0
    assert "model_name/pretrained" in err


def test_empty_keywords_errors(tmp_path, monkeypatch):
    image_path = tmp_path / "empty keywords.jpg"
    Image.new("RGB", (1, 1)).save(image_path)
    _, err, code = run_main(["-q", "-k", ",,,", str(image_path)], monkeypatch)
    assert code != 0
    assert "keyword" in err.lower() or "--like" in err.lower()


def test_empty_exclude_keywords_errors(tmp_path, monkeypatch):
    image_path = tmp_path / "empty exclude.jpg"
    Image.new("RGB", (1, 1)).save(image_path)
    _stub_pipeline(monkeypatch)
    out, _, code = run_main(
        ["-q", "-s", "-x", ",,,", "-k", "dog", str(image_path)],
        monkeypatch,
    )
    assert code == 0
    assert "0.750" in out


def test_exclude_only_without_include_or_like_is_error(tmp_path, monkeypatch):
    """Exclude-only queries are nonsensical (ranks by least similar)."""
    image_path = tmp_path / "my photo.jpg"
    Image.new("RGB", (1, 1)).save(image_path)
    _, err, code = run_main(
        ["-q", "-x", "cat", str(image_path)],
        monkeypatch,
    )
    assert code != 0
    assert "keyword" in err.lower() or "--like" in err.lower()


def test_both_keyword_lists_empty_is_error(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    Image.new("RGB", (1, 1)).save(image_path)
    _, err, code = run_main(
        ["-q", "-x", ",,,", "-k", ",,,", str(image_path)],
        monkeypatch,
    )
    assert code != 0
    assert "keyword" in err.lower()


def test_count_short_flag_is_rejected(monkeypatch):
    _, err, code = run_main(["-c", "-k", "dog", "x.jpg"], monkeypatch)
    assert code != 0
    assert "unrecognized arguments: -c" in err


def test_count_long_flag_is_rejected(monkeypatch):
    _, err, code = run_main(["--count", "-k", "dog", "x.jpg"], monkeypatch)
    assert code != 0
    assert "unrecognized arguments: --count" in err


def test_view_rejects_print0(monkeypatch):
    _, err, code = run_main(["--view", "-print0", "-k", "dog", "x.jpg"], monkeypatch)
    assert code != 0
    assert "not allowed with argument" in err
    assert "--view" in err
    assert "-print0" in err


# --- error paths ---

def test_nonexistent_path_errors(monkeypatch):
    _stub_pipeline(monkeypatch)
    _, err, code = run_main(
        ["-q", "-k", "dog", "/tmp/grape_nonexistent_path_xyz"],
        monkeypatch,
    )
    assert code != 0
    assert "No such file" in err


def test_dir_without_r_warns(tmp_path, monkeypatch):
    """Passing a directory without -r warns and finds no images."""
    Image.new("RGB", (1, 1)).save(tmp_path / "img.jpg", format="JPEG")
    _stub_pipeline(monkeypatch)
    _, err, code = run_main(["-q", "-k", "dog", str(tmp_path)], monkeypatch)
    assert "Is a directory" in err


def test_no_images_exits(tmp_path, monkeypatch):
    """Empty directory with -r exits with an error."""
    _stub_pipeline(monkeypatch)
    _, err, code = run_main(["-q", "-R", "-k", "dog", str(tmp_path)], monkeypatch)
    assert code != 0
    assert "no images found" in err


def _stub_pipeline(monkeypatch, score=0.75):
    """Stub the delayed pipeline tasks so tests don't load the real model.

    Replaces _load_model, _encode_keywords, _resolve_and_index_cache,
    and _score_all with plain @dask.delayed stubs that return
    lightweight fake objects so tests don't load the real model.
    """
    import grape.cli as cli_mod

    @dask.delayed
    def _fake_load_model(model_name, pretrained, quiet):
        return object()

    @dask.delayed
    def _fake_encode_keywords(
        model, score_keywords, prompt_templates, cache_context, cache,
    ):
        return object()

    @dask.delayed
    def _fake_resolve_and_index_cache(model_name, pretrained, cache):
        return (None, None)

    @dask.delayed
    def _fake_prepare_cached_embeddings(scan_result, cache_context):
        items, scan_done = scan_result
        return (None, [], items, scan_done)

    @dask.delayed
    def _fake_score_all(
        prepared, model, score_keywords, like_paths, text_emb,
        cache, quiet,
    ):
        _image_emb, _cached_items, uncached_items, scan_done = prepared
        results = [
            ScoredImage(path=item.path, score=score,
                        scores={kw: score for kw in score_keywords})
            for item in uncached_items
        ]
        return results, scan_done

    @dask.delayed
    def _fake_encode_like_images(model, like_paths, cache_context, cache):
        return np.ones((len(like_paths), 2), dtype=np.float32)

    @dask.delayed
    def _fake_combine_query_embeddings(text_emb, like_emb):
        return object()

    monkeypatch.setattr(cli_mod, "_load_model", _fake_load_model)
    monkeypatch.setattr(cli_mod, "_encode_keywords", _fake_encode_keywords)
    monkeypatch.setattr(
        cli_mod, "_encode_like_images", _fake_encode_like_images,
    )
    monkeypatch.setattr(
        cli_mod, "_combine_query_embeddings", _fake_combine_query_embeddings,
    )
    monkeypatch.setattr(
        cli_mod, "_resolve_and_index_cache", _fake_resolve_and_index_cache,
    )
    monkeypatch.setattr(
        cli_mod, "_prepare_cached_embeddings", _fake_prepare_cached_embeddings,
    )
    monkeypatch.setattr(cli_mod, "_score_all", _fake_score_all)


def test_default_output_shell_quotes_paths(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    Image.new("RGB", (1, 1)).save(image_path)
    _stub_pipeline(monkeypatch)
    out, _, code = run_main(["-q", "-k", "dog", str(image_path)], monkeypatch)
    assert code == 0
    assert out == f"{shlex.quote(str(image_path))}\n"


def test_print0_outputs_raw_nul_terminated_paths(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    Image.new("RGB", (1, 1)).save(image_path)
    _stub_pipeline(monkeypatch)
    out, _, code = run_main(
        ["-q", "-print0", "-k", "dog", str(image_path)], monkeypatch,
    )
    assert code == 0
    assert out == f"{image_path}\0"


def test_scores_output_shell_quotes_paths(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    Image.new("RGB", (1, 1)).save(image_path)
    _stub_pipeline(monkeypatch)
    out, _, code = run_main(["-q", "-s", "-k", "dog", str(image_path)], monkeypatch)
    assert code == 0
    assert shlex.quote(str(image_path)) in out


def test_exclude_keywords_adjusts_output_score(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    Image.new("RGB", (1, 1)).save(image_path)

    # Custom stub that returns different scores per keyword
    import grape.cli as cli_mod

    @dask.delayed
    def _fake_score_all(
        prepared, model, score_keywords, like_paths, text_emb,
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
        ["-q", "-s", "-x", "cat", "-k", "dog", str(image_path)],
        monkeypatch,
    )
    assert code == 0
    assert "0.700" in out
    assert shlex.quote(str(image_path)) in out


def test_exclude_alias_anti_is_rejected(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    Image.new("RGB", (1, 1)).save(image_path)
    _, err, code = run_main(
        ["-q", "-s", "--anti", "cat", "-k", "dog", str(image_path)],
        monkeypatch,
    )
    assert code != 0
    assert "unrecognized arguments: --anti" in err


def test_exclude_verbose_shows_not_keyword(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    Image.new("RGB", (1, 1)).save(image_path)

    import grape.cli as cli_mod

    @dask.delayed
    def _fake_score_all(
        prepared, model, score_keywords, like_paths, text_emb,
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
        ["-q", "-v", "-x", "cat", "-k", "dog", str(image_path)],
        monkeypatch,
    )
    assert code == 0
    assert "not:cat: 0.200" in out


def test_ensemble_prompts_uses_default_template_set(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    Image.new("RGB", (1, 1)).save(image_path)
    captured: dict[str, object] = {}

    import grape.cli as cli_mod

    @dask.delayed
    def _fake_encode_keywords(
        model, score_keywords, prompt_templates, cache_context, cache,
    ):
        captured["prompt_templates"] = prompt_templates
        return object()

    _stub_pipeline(monkeypatch)
    monkeypatch.setattr(cli_mod, "_encode_keywords", _fake_encode_keywords)

    out, _, code = run_main(
        ["-q", "-k", "dog", str(image_path)],
        monkeypatch,
    )
    assert code == 0
    assert out == f"{shlex.quote(str(image_path))}\n"
    assert captured["prompt_templates"] == DEFAULT_PROMPT_ENSEMBLE
    assert len(DEFAULT_PROMPT_ENSEMBLE) == 3


def test_ensemble_prompts_custom_templates_override_default(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    Image.new("RGB", (1, 1)).save(image_path)
    captured: dict[str, object] = {}

    import grape.cli as cli_mod

    @dask.delayed
    def _fake_encode_keywords(
        model, score_keywords, prompt_templates, cache_context, cache,
    ):
        captured["prompt_templates"] = prompt_templates
        return object()

    _stub_pipeline(monkeypatch)
    monkeypatch.setattr(cli_mod, "_encode_keywords", _fake_encode_keywords)

    out, _, code = run_main(
        [
            "-q",
            "--ensemble-prompts",
            "one {},two {}",
            "-k", "dog",
            str(image_path),
        ],
        monkeypatch,
    )
    assert code == 0
    assert out == f"{shlex.quote(str(image_path))}\n"
    assert captured["prompt_templates"] == ["one {}", "two {}"]


def test_ensemble_prompts_template_without_placeholder_errors(monkeypatch):
    _, err, code = run_main(
        ["--ensemble-prompts", "bad template", "-k", "dog", "x.jpg"],
        monkeypatch,
    )
    assert code != 0
    assert "must include '{}'" in err


def test_view_calls_webview_with_html(tmp_path, monkeypatch):
    image_path = tmp_path / "my photo.jpg"
    Image.new("RGB", (1, 1)).save(image_path)
    _stub_pipeline(monkeypatch)
    captured: list[str] = []
    monkeypatch.setattr("grape.cli._show_in_webview", captured.append)
    out, _, code = run_main(["-q", "--view", "-k", "dog", str(image_path)], monkeypatch)
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
        prepared, object(), ["dog"], [], text_emb,
        _NoDbCache(), True,
    ).compute()

    assert len(results) == 1
    assert results[0].path == Path("/tmp/a.jpg")
    assert results[0].scores["dog"] == pytest.approx(1.0)
    assert results[0].score == pytest.approx(1.0)


def test_score_all_duplicate_like_paths_keep_separate_scores():
    """--like /a/ref.jpg --like /b/ref.jpg must preserve both scores.

    Previously, like scores were stored in the same dict as keyword scores
    keyed by basename -- so two like images with the same filename would
    overwrite each other.  Now like_scores is a separate list of
    (path, similarity) tuples, avoiding collisions entirely.
    """
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
    # One text keyword + two like images (same basename, different dirs).
    text_keywords = ["dog"]
    like_paths = ["/x/ref.jpg", "/y/ref.jpg"]
    query_emb = np.array(
        [
            [1.0, 0.0],   # "dog"
            [0.0, 1.0],   # like /x/ref.jpg
            [1.0, 1.0],   # like /y/ref.jpg
        ],
        dtype=np.float32,
    )
    scan_result = (items, _ScanDone(image_count=1))
    cache_context = ("model-id", cached_index)

    prepared = _prepare_cached_embeddings(
        scan_result, cache_context,
    ).compute()

    results, _done = _score_all(
        prepared, object(), text_keywords, like_paths, query_emb,
        _NoDbCache(), True,
    ).compute()

    assert len(results) == 1
    r = results[0]
    # Text keyword score is in the dict.
    assert "dog" in r.scores
    # Both like images have their own entry in like_scores.
    assert len(r.like_scores) == 2
    assert r.like_scores[0][0] == "/x/ref.jpg"
    assert r.like_scores[1][0] == "/y/ref.jpg"
    # The two like scores are different (different query vectors).
    assert r.like_scores[0][1] != pytest.approx(r.like_scores[1][1])



def test_scan_files_includes_cache_metadata(tmp_path):
    """_scan_files returns items with cache metadata for plain files."""
    image_path = tmp_path / "a.jpg"
    Image.new("RGB", (1, 1)).save(image_path)

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


def test_scan_files_rejects_non_images_passed_directly(tmp_path):
    """Non-image files passed as direct paths are filtered during scan.

    Regression: previously _scan_files accepted any file passed directly
    without checking _is_image, so videos/broken files showed up as
    "uncached" and triggered an unnecessary model load.
    """
    from grape.cli import _scan_files

    jpg = tmp_path / "real.jpg"
    Image.new("RGB", (1, 1)).save(jpg)
    mp4 = tmp_path / "video.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x1cftypisom")
    txt = tmp_path / "readme.txt"
    txt.write_text("not an image")

    items, done = _scan_files(
        [str(jpg), str(mp4), str(txt)], False, None,
    ).compute()

    assert done.image_count == 1
    assert items[0].path == jpg


# --- _format_query_summary ---

def test_format_query_summary_keywords_only():
    from grape.cli import _format_query_summary
    assert _format_query_summary(["dog", "cat"], []) == "dog, cat"


def test_format_query_summary_with_like():
    from grape.cli import _format_query_summary
    result = _format_query_summary(["dog"], [], like_names=["ref.jpg"])
    assert result == "dog; like: ref.jpg"


def test_format_query_summary_with_exclude():
    from grape.cli import _format_query_summary
    result = _format_query_summary(["dog"], ["cat"])
    assert result == "dog; excluding: cat"


def test_format_query_summary_all_parts():
    from grape.cli import _format_query_summary
    result = _format_query_summary(
        ["dog"], ["cat"], like_names=["ref.jpg"],
    )
    assert result == "dog; like: ref.jpg; excluding: cat"


def test_format_query_summary_no_keywords():
    from grape.cli import _format_query_summary
    assert _format_query_summary([], []) == "(no keywords)"


# --- _format_results with like_scores ---

def test_format_results_verbose_shows_like_scores():
    results = [
        ScoredImage(
            path=Path("/a/img.jpg"),
            scores={"dog": 0.8},
            like_scores=[("/x/ref.jpg", 0.95)],
            score=0.875,
        ),
    ]
    output = _format_results(results, verbose=True)
    assert "dog: 0.800" in output
    assert "like:ref.jpg: 0.950" in output


# --- _score_all error handling ---

def test_score_all_skips_syntax_error():
    """SyntaxError during image encoding is caught and recorded as not-image."""
    from grape.cli import (
        _prepare_cached_embeddings,
        _ScanDone,
        _score_all,
    )

    class _RaisingModel:
        def model_id(self):
            return "test-model"
        def encode_image(self, path):
            raise SyntaxError("Not a PNG file")

    class _TrackingCache:
        def __init__(self):
            self.not_images = []
        def get(self, path, model_id):
            return None
        def put(self, path, model_id, emb):
            pass
        def put_not_image(self, path, *, path_key=None, file_stat=None):
            self.not_images.append(str(path))
        def get_many_for_paths(self, *a, **kw):
            raise AssertionError("should not be called")

    items = [
        _ScannedImage(
            path=Path("/tmp/bad.png"),
            path_key="/tmp/bad.png",
            file_stat="stat-bad",
        ),
    ]
    text_emb = np.array([[1.0, 0.0]], dtype=np.float32)
    scan_result = (items, _ScanDone(image_count=1))
    cache_context = ("model-id", {})
    tracking = _TrackingCache()

    prepared = _prepare_cached_embeddings(
        scan_result, cache_context,
    ).compute()

    results, _done = _score_all(
        prepared, _RaisingModel(), ["dog"], [], text_emb,
        tracking, True,
    ).compute()

    assert len(results) == 0
    assert "/tmp/bad.png" in tracking.not_images


def test_score_all_skips_oserror_no_errno():
    """OSError with errno=None (PIL format error) is caught; real errors propagate."""
    from grape.cli import (
        _prepare_cached_embeddings,
        _ScanDone,
        _score_all,
    )

    class _RaisingModel:
        def model_id(self):
            return "test-model"
        def encode_image(self, path):
            raise OSError("cannot identify image file")

    items = [
        _ScannedImage(
            path=Path("/tmp/bad.jpg"),
            path_key="/tmp/bad.jpg",
            file_stat="stat-bad",
        ),
    ]
    text_emb = np.array([[1.0, 0.0]], dtype=np.float32)
    scan_result = (items, _ScanDone(image_count=1))
    cache_context = ("model-id", {})

    prepared = _prepare_cached_embeddings(
        scan_result, cache_context,
    ).compute()

    results, _done = _score_all(
        prepared, _RaisingModel(), ["dog"], [], text_emb,
        None, True,
    ).compute()

    assert len(results) == 0


def test_score_all_propagates_real_oserror():
    """OSError with an errno (e.g. ENOENT) must not be swallowed."""
    import errno

    from grape.cli import (
        _prepare_cached_embeddings,
        _ScanDone,
        _score_all,
    )

    class _RaisingModel:
        def model_id(self):
            return "test-model"
        def encode_image(self, path):
            raise FileNotFoundError(
                errno.ENOENT, "No such file", str(path),
            )

    items = [
        _ScannedImage(
            path=Path("/tmp/gone.jpg"),
            path_key="/tmp/gone.jpg",
            file_stat="stat-gone",
        ),
    ]
    text_emb = np.array([[1.0, 0.0]], dtype=np.float32)
    scan_result = (items, _ScanDone(image_count=1))
    cache_context = ("model-id", {})

    prepared = _prepare_cached_embeddings(
        scan_result, cache_context,
    ).compute()

    with pytest.raises(FileNotFoundError):
        _score_all(
            prepared, _RaisingModel(), ["dog"], [], text_emb,
            None, True,
        ).compute()


# --- _scan_files with cached not-images ---

def test_scan_files_skips_direct_file_cached_as_not_image(tmp_path):
    """Direct file arg known as not-image via cache is skipped."""
    from grape.cache import EmbeddingCache
    from grape.cli import _scan_files

    jpg = tmp_path / "real.jpg"
    Image.new("RGB", (1, 1)).save(jpg)
    bad = tmp_path / "bad.dat"
    bad.write_bytes(b"not an image")

    cache = EmbeddingCache(tmp_path / "test.db")
    cache.put_not_image(bad)

    items, done = _scan_files(
        [str(jpg), str(bad)], False, cache,
    ).compute()

    assert done.image_count == 1
    assert items[0].path == jpg
    cache.close()


# --- _filter_and_sort ---

def test_filter_and_sort_threshold(capsys):
    """--threshold filters results below the cutoff."""
    from grape.cli import _filter_and_sort, _ScanDone
    results = [
        ScoredImage(path=Path("/a.jpg"), scores={"dog": 0.8}, score=0.8),
        ScoredImage(path=Path("/b.jpg"), scores={"dog": 0.3}, score=0.3),
        ScoredImage(path=Path("/c.jpg"), scores={"dog": 0.5}, score=0.5),
    ]
    out = _filter_and_sort(
        (results, _ScanDone(image_count=3)),
        ["dog"], [], [], threshold=0.4, top=None, quiet=True,
    )
    assert [r.score for r in out] == [0.8, 0.5]


def test_filter_and_sort_top_n(capsys):
    """--top limits to N highest-scoring results."""
    from grape.cli import _filter_and_sort, _ScanDone
    results = [
        ScoredImage(path=Path(f"/{i}.jpg"), scores={"dog": s}, score=s)
        for i, s in enumerate([0.3, 0.8, 0.5, 0.7])
    ]
    out = _filter_and_sort(
        (results, _ScanDone(image_count=4)),
        ["dog"], [], [], threshold=None, top=2, quiet=True,
    )
    assert len(out) == 2
    assert out[0].score == 0.8
    assert out[1].score == 0.7


def test_filter_and_sort_status_message(capsys):
    """Without quiet, prints image count and query to stderr."""
    from grape.cli import _filter_and_sort, _ScanDone
    results = [
        ScoredImage(path=Path("/a.jpg"), scores={"sunset": 0.5}, score=0.5),
    ]
    _filter_and_sort(
        (results, _ScanDone(image_count=1)),
        ["sunset"], [], [], threshold=None, top=None, quiet=False,
    )
    err = capsys.readouterr().err
    assert "1 image" in err
    assert "sunset" in err


# --- end-to-end with real model (slow) ---

FIXTURES = Path(__file__).parent / "fixtures"
E2E_MODEL = "ViT-B-32/laion2b_s34b_b79k"


@pytest.mark.slow
def test_e2e_single_file(monkeypatch):
    """Score a single file through the full CLI pipeline."""
    out, _, code = run_main(
        [
            "-q",
            "--no-cache",
            "--model",
            E2E_MODEL,
            "-k",
            "dog",
            str(FIXTURES / "dog.jpg"),
        ],
        monkeypatch,
    )
    assert code == 0
    assert "dog.jpg" in out



@pytest.mark.slow
def test_e2e_with_cache(tmp_path, monkeypatch):
    """--cache creates a DB and caches image embeddings."""
    db = tmp_path / "test.db"
    out, _, code = run_main(
        [
            "-q",
            "--model",
            E2E_MODEL,
            "--cache",
            str(db),
            "-k",
            "dog",
            str(FIXTURES / "dog.jpg"),
        ],
        monkeypatch,
    )
    assert code == 0
    assert "dog.jpg" in out
    assert db.exists()


@pytest.mark.slow
def test_e2e_like(monkeypatch):
    """--like finds images similar to a reference."""
    out, _, code = run_main(
        [
            "-q", "--no-cache",
            "--model", E2E_MODEL,
            "--scores",
            "--like", str(FIXTURES / "dog.jpg"),
            str(FIXTURES / "cat.jpg"),
            str(FIXTURES / "dog.jpg"),
        ],
        monkeypatch,
    )
    assert code == 0
    lines = out.strip().splitlines()
    assert len(lines) == 2
    # Self-match (dog vs dog) should score highest and appear first.
    assert "dog.jpg" in lines[0]
