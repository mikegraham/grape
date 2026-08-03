# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Cached embeddings now survive reboots and remounts on anonymous-device filesystems (ecryptfs, overlay, NFS, btrfs, ZFS...). The cache keyed on the filesystem device id, which the kernel renumbers at mount time, so the entire library re-encoded after every reboot. Existing cache rows are reused as-is; no rebuild needed.

## [0.2.0] - 2026-05-12

### Added
- Animated GIF/WEBP/APNG inputs are scored across multiple frames, not just the first. **Cache rows for animated images from 0.1.0 are first-frame-only and won't auto-update.**
- `--view` shows image dimensions next to the filename.
- `--view` snaps Space/PgDn to each result.
- `-v` / `--verbose` prints each newly-encoded file's path to stderr.

### Changed
- CLI output preserves the user's input path style (relative stays relative including `./`; absolute stays absolute).
- Warm-cache invocations are ~20% faster.
- `find_images`, `iter_image_records`, `ScoredImage.path`, `ImageRecord.path` return `str` instead of `pathlib.Path`.

### Fixed
- `find ... | grape -` and `grape -R` write to the same cache row when symlinks are involved (previously fragmented).
- Stdin paths containing NUL bytes are dropped instead of crashing.
- `--cache PATH` is taken as-is; only the default cache auto-creates its parent. Fixes a crash on `--cache grape.db` (bare filename).

## [0.1.0] - 2026-04-25

Initial release.
