# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `--view`: image dimensions (e.g. `1920x1080`) shown next to the filename.
- `--view`: vertical scroll-snap (proximity mode) so Space / PgDn lands on each result; small wheel scrolls don't snap.
- `-v` / `--verbose` now also prints each newly-encoded file's path to stderr.
- `ImageRecord.from_display_path()` classmethod that resolves realpath + stat once at construction so callers can't drift the cache key by computing it inconsistently.

### Changed
- CLI output preserves the user's input path style verbatim: a relative path stays relative (with a leading `./` if given), an absolute path stays absolute. Matches `find` / `rg` / `du` behaviour. The cache key is always the realpath; display and key are now cleanly separated.
- `--view` layout: long paths use single-line ellipsis with hover tooltip (was multi-line wrap); a single tall image is capped at viewport height; score line uses `opacity: 0.6` so it adapts to dark mode (was a fixed `#555`).
- `find_images` and `iter_image_records` yield display strings instead of `pathlib.Path` objects; `ScoredImage.path` and `ImageRecord.path` are now `str`. Library users that relied on receiving `Path` will need `Path(p)` at the call site.

### Fixed
- Cache-key parity between stdin file inputs (`find ... | grape -`) and `grape -R`: both now write to the same cache row even when symlinks are involved (previously they fragmented).
- Stdin paths containing NUL bytes are silently dropped instead of crashing with `ValueError` from `os.stat` (most often hit when piping `find -print0` without `xargs -0`).

### Internal
- `EmbeddingCache` path-argument types narrowed from `str | os.PathLike[str]` to `str`.
- The stat-key builder is now `stat_key_from_stat` (renamed from the private `_stat_key_from_stat`) with a docstring explaining the (size, mtime, ino, dev, ctime) choice. The format string lives at one site; previously three byte-identical copies were maintained in cache, search, and cli.
- `_ScannedImage` (in cli) was a structural duplicate of `ImageRecord` (in search); dropped in favour of `ImageRecord` everywhere.
- `_get_embedding` now threads `path_key`/`file_stat` through to the cache, avoiding redundant `realpath`+`stat` syscalls and a small TOCTOU window between scan and encode.
- `_show_in_webview` documents why `debug=True` is required (pywebview 6.x couples right-click context menu to debug mode in every backend; OPEN_DEVTOOLS_IN_DEBUG=False keeps the inspector hidden).

## [0.1.0] - 2026-04-25

Initial release.
