# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Internal
- `EmbeddingCache` path-argument types narrowed from `str | os.PathLike[str]` to `str`.
- The stat-key builder is now `stat_key_from_stat` (renamed from the private `_stat_key_from_stat`) with a docstring explaining the (size, mtime, ino, dev, ctime) choice.

## [0.1.0] - 2026-04-25

Initial release.
