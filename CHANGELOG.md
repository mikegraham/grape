# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `find_images` and `iter_image_records` now yield display strings (the caller's input style, e.g. `./foo.jpg` is preserved like `find`/`rg`/`du` do) instead of normalized `pathlib.Path` objects. The cache key is still always the realpath; the two are cleanly separated. Library users that relied on receiving `Path` instances will need `Path(p)` at the call site.
- `ScoredImage.path` and `ImageRecord.path` are now `str` (display path); cache keys live on `ImageRecord.path_key`.

### Added
- `ImageRecord.from_display_path()` classmethod that resolves realpath + stat once at construction so callers can't drift the cache key by computing it inconsistently.

### Internal
- `EmbeddingCache` path-argument types narrowed from `str | os.PathLike[str]` to `str`.
- The stat-key builder is now `stat_key_from_stat` (renamed from the private `_stat_key_from_stat`) with a docstring explaining the (size, mtime, ino, dev, ctime) choice.
- `_get_embedding` now threads `path_key`/`file_stat` through to the cache, avoiding redundant `realpath`+`stat` syscalls and a small TOCTOU window between scan and encode.

## [0.1.0] - 2026-04-25

Initial release.
