# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Runs that load the model finish about 1 s sooner, by skipping interpreter teardown on exit.
- Default model is now `ViT-L-16-SigLIP2-256/webli` (was `EVA02-L-14/merged2b_s4b_b131k`): +2.7 zero-shot ImageNet, better retrieval, same encode speed. **First run re-encodes your library.** Download is ~3.5 GB (was ~900 MB).

### Fixed
- Models with a Hugging Face tokenizer (SigLIP, SigLIP 2) now load offline once cached, and no longer contact the Hub on every run.
- No more empty `Encoding: 0it` progress bar when every image is already cached.

## [0.3.0] - 2026-08-03

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
