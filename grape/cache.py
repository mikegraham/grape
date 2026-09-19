"""SQLite-backed embedding cache.

Avoids redundant CLIP encoding by caching embeddings keyed on
(absolute_path, model_id) with file-stat-based invalidation.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_CREATE_EMBEDDINGS = """\
CREATE TABLE IF NOT EXISTS embeddings (
    path       TEXT NOT NULL,
    file_stat  TEXT NOT NULL,
    model      TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    cached_at  TEXT NOT NULL
               DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (path, model)
)
"""

_CREATE_NOT_IMAGES = """\
CREATE TABLE IF NOT EXISTS not_images (
    path       TEXT NOT NULL PRIMARY KEY,
    file_stat  TEXT NOT NULL
)
"""

_CREATE_MODEL_IDS = """\
CREATE TABLE IF NOT EXISTS model_ids (
    model_name TEXT NOT NULL,
    pretrained TEXT NOT NULL,
    model_id   TEXT NOT NULL,
    PRIMARY KEY (model_name, pretrained)
)
"""

_CREATE_TEXT_EMBEDDINGS = """\
CREATE TABLE IF NOT EXISTS text_embeddings (
    model      TEXT NOT NULL,
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    cached_at  TEXT NOT NULL
               DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (model, text)
)
"""

# Metadata-only scans (image_paths) would otherwise walk the table
# itself, whose pages are ~97% embedding blob -- and at >=1024 dims every
# row spills to an overflow page. This index covers those queries so they
# read a small b-tree instead. Column order matters: model first so
# ``WHERE model = ?`` can seek.
_CREATE_COVER_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_embeddings_cover
    ON embeddings (model, path, file_stat)
"""

_INSERT = (
    "INSERT OR REPLACE INTO embeddings"
    " (path, file_stat, model, embedding)"
    " VALUES (?, ?, ?, ?)"
)


def _stat_key(path: str) -> str:
    """JSON array of stat fields used for cache invalidation."""
    st = os.stat(path)
    return stat_key_from_stat(st)


def stat_key_from_stat(st: os.stat_result) -> str:
    """Cache invalidation token: (size, mtime, ino, dev, ctime).

    Conservative on purpose: a change to size, mtime, ino, or ctime
    invalidates the entry, maximizing the chance a hit really is the
    same file content. Format must stay byte-identical to
    ``json.dumps([...])``.

    The dev slot is deliberately pinned to 0, not ``st.st_dev``. On
    anonymous-device filesystems (ecryptfs, overlay, NFS, btrfs...) the
    kernel assigns the major-0 minor at mount time, so it renumbers
    across reboots/remounts even though the file never changed --
    including it invalidated the whole library on every reboot. ino
    still guards against the path pointing at a different file on the
    same volume. The slot is kept as a constant rather than dropped so
    the field positions stay stable and ``_canonical_stat`` can
    neutralize the real dev stored in pre-0.3.0 rows.
    """
    return (
        f"[{st.st_size}, {st.st_mtime_ns},"
        f" {st.st_ino}, 0, {st.st_ctime_ns}]"
    )


def _canonical_stat(token: str) -> str:
    """Return *token* with the dev field (index 3) zeroed for comparison.

    Rows cached before dev was pinned to 0 stored the real st_dev there.
    Canonicalizing both sides at compare time lets those rows keep
    matching the current file without rewriting the DB or re-encoding.
    A token that is not the expected 5-element integer array is returned
    unchanged, so opaque sentinels (e.g. test values like "stat-a")
    still compare exactly.
    """
    # Fast path: every token written since dev was pinned to 0 is already
    # canonical, so skip the json round-trip (~2us/row, paid on every row
    # of every index scan). Tokens grape writes are returned unchanged,
    # which is what the round-trip below would produce anyway.
    #
    # A token grape did not write may skip normalization here (json.dumps
    # would render "1.5e-9" as "1.5e-09"). That is safe in the only
    # direction that matters: this is used to compare a stored token
    # against a freshly computed one, and returning a token verbatim
    # cannot make two distinct stats compare equal. Worst case is a
    # re-encode, never a stale hit. Guarded by
    # test_canonical_stat_never_merges_distinct_stats.
    if token.startswith("[") and token.endswith("]"):
        parts = token[1:-1].split(", ")
        if len(parts) == 5 and parts[3] == "0":
            return token
    try:
        fields = json.loads(token)
    except (ValueError, TypeError):
        return token
    if not isinstance(fields, list) or len(fields) != 5:
        return token
    fields[3] = 0
    return json.dumps(fields)


class EmbeddingIndex(NamedTuple):
    """Every cached image embedding for one model, as a single matrix."""

    # (path, canonical file_stat) -> row of ``matrix``
    rows: dict[tuple[str, str], int]
    matrix: NDArray[np.float32]


class EmbeddingCache:
    """Read-through cache for CLIP image embeddings stored in SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            with self._lock:
                self._conn.execute("PRAGMA journal_mode=WAL")
                # Avoid "database is locked" when multiple grape processes
                # hit the same cache file concurrently.
                self._conn.execute("PRAGMA busy_timeout=5000")
                # Memory-map the db instead of read()ing it. Embedding
                # blobs make this file large (~4 KB/row), and mmap cut a
                # full scan from 263ms to 113ms at 50k rows.
                self._conn.execute("PRAGMA mmap_size=1073741824")
                # Validate b-tree cell bounds on every page read, so a
                # damaged file is caught instead of followed into
                # unrelated memory. Measured cost on a full scan: nil.
                self._conn.execute("PRAGMA cell_size_check=ON")
                self._conn.execute(_CREATE_EMBEDDINGS)
                self._conn.execute(_CREATE_NOT_IMAGES)
                self._conn.execute(_CREATE_TEXT_EMBEDDINGS)
                self._conn.execute(_CREATE_MODEL_IDS)
                self._conn.execute(_CREATE_COVER_INDEX)
                self._conn.commit()
        except sqlite3.DatabaseError:
            self._conn.close()
            raise

    def get(
        self,
        path: str,
        model_id: str,
        *,
        path_key: str | None = None,
        file_stat: str | None = None,
    ) -> NDArray[np.float32] | None:
        """Return the cached embedding, or ``None`` on miss/stale."""
        import numpy as np

        resolved = path_key or os.path.realpath(path)
        with self._lock:
            row = self._conn.execute(
                "SELECT file_stat, embedding FROM embeddings"
                " WHERE path = ? AND model = ?",
                (resolved, model_id),
            ).fetchone()
        if row is None:
            return None
        stored_stat, blob = row
        stat_key = file_stat or _stat_key(resolved)
        if _canonical_stat(stored_stat) != _canonical_stat(stat_key):
            return None
        arr: NDArray[np.float32] = (
            np.frombuffer(blob, dtype=np.float32)
            .reshape(1, -1)
            .copy()
        )
        return arr

    def get_many_for_paths(
        self,
        model_id: str,
        path_stats: Mapping[str, str],
        *,
        chunk_size: int = 1024,
    ) -> dict[str, NDArray[np.float32]]:
        """Return cached embeddings for many paths in batched queries.

        ``path_stats`` maps absolute path keys to expected stat keys.
        Only rows whose ``file_stat`` matches the provided value are
        returned.
        """
        import numpy as np

        if not path_stats:
            return {}

        paths = list(path_stats.keys())
        out: dict[str, NDArray[np.float32]] = {}
        for start in range(0, len(paths), chunk_size):
            chunk = paths[start:start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            sql = (
                "SELECT path, file_stat, embedding FROM embeddings"
                " WHERE model = ? AND path IN ({})"
            ).format(placeholders)
            params = [model_id, *chunk]
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
            for row_path, row_stat, blob in rows:
                expected_stat = path_stats.get(row_path)
                if expected_stat is None or (
                    _canonical_stat(row_stat) != _canonical_stat(expected_stat)
                ):
                    continue
                out[row_path] = np.frombuffer(blob, dtype=np.float32).copy()
        return out

    def embedding_index_for_model(self, model_id: str) -> EmbeddingIndex:
        """Load every cached embedding for *model_id* into one matrix."""
        import numpy as np

        rows: dict[tuple[str, str], int] = {}
        with self._lock:
            (capacity,) = self._conn.execute(
                "SELECT count(*) FROM embeddings WHERE model = ?", (model_id,),
            ).fetchone()
            cursor = self._conn.execute(
                "SELECT path, file_stat, embedding FROM embeddings"
                " WHERE model = ?",
                (model_id,),
            )
            first = cursor.fetchone()
            if first is None:
                return EmbeddingIndex({}, np.empty((0, 0), dtype=np.float32))
            # Copy blobs straight into one matrix: an array per row plus a
            # vstack cost ~420ms at 20k rows, this ~70ms.
            size = len(first[2])
            matrix = np.empty((capacity, size // 4), dtype=np.float32)
            buf = matrix.data.cast("B")
            row = 0
            for path, file_stat, blob in chain((first,), cursor):
                # Another process inserted rows since the count.
                if row == capacity:
                    break
                buf[row * size:(row + 1) * size] = blob
                rows[(path, _canonical_stat(file_stat))] = row
                row += 1
        return EmbeddingIndex(rows, matrix[:row])

    def has_any_embedding(
        self,
        path: str,
        *,
        path_key: str | None = None,
        file_stat: str | None = None,
    ) -> bool:
        """Return ``True`` if any cached embedding matches current file stat."""
        resolved = path_key or os.path.realpath(path)
        stat_key = _canonical_stat(file_stat or _stat_key(resolved))
        # Can't filter file_stat in SQL: older rows store a different
        # dev, so match on canonical form in Python instead.
        with self._lock:
            rows = self._conn.execute(
                "SELECT file_stat FROM embeddings WHERE path = ?",
                (resolved,),
            ).fetchall()
        return any(_canonical_stat(s) == stat_key for (s,) in rows)

    def image_paths(self) -> set[str]:
        """Return paths that have an embedding under any model."""
        with self._lock:
            rows = self._conn.execute(
                # No DISTINCT: the set comprehension below already dedups,
                # and DISTINCT forces a temp b-tree that stops SQLite from
                # using the covering index (48ms -> 9ms at 20k rows).
                "SELECT path FROM embeddings"
            ).fetchall()
        return {path for (path,) in rows}

    def not_image_index(self) -> set[tuple[str, str]]:
        """Return ``(path, file_stat)`` pairs known to be non-images."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, file_stat FROM not_images"
            ).fetchall()
        return {(path, _canonical_stat(file_stat)) for path, file_stat in rows}

    def put(
        self,
        path: str,
        model_id: str,
        embedding: NDArray[np.float32],
        *,
        path_key: str | None = None,
        file_stat: str | None = None,
    ) -> None:
        """Insert or replace the cached embedding for *(path, model)*."""
        resolved = path_key or os.path.realpath(path)
        stat_key = file_stat or _stat_key(resolved)
        with self._lock:
            self._conn.execute(
                _INSERT,
                (resolved, stat_key,
                 model_id, embedding.tobytes()),
            )
            self._conn.commit()

    def put_many(
        self,
        model_id: str,
        rows: Sequence[
            tuple[
                str,
                NDArray[np.float32],
                str | None,
                str | None,
            ]
        ],
    ) -> None:
        """Insert or replace multiple embeddings in one transaction."""
        payload: list[tuple[str, str, str, bytes]] = []
        for path, embedding, path_key, file_stat in rows:
            resolved = path_key or os.path.realpath(path)
            stat_key = file_stat or _stat_key(resolved)
            payload.append((resolved, stat_key, model_id, embedding.tobytes()))
        if not payload:
            return
        with self._lock:
            self._conn.executemany(_INSERT, payload)
            self._conn.commit()

    def is_not_image(
        self,
        path: str,
        *,
        path_key: str | None = None,
        file_stat: str | None = None,
    ) -> bool:
        """Return ``True`` if *path* was previously recorded as not an image."""
        resolved = path_key or os.path.realpath(path)
        with self._lock:
            row = self._conn.execute(
                "SELECT file_stat FROM not_images WHERE path = ?",
                (resolved,),
            ).fetchone()
        if row is None:
            return False
        stored_stat: str = row[0]
        stat_key = file_stat or _stat_key(resolved)
        return _canonical_stat(stored_stat) == _canonical_stat(stat_key)

    def put_not_image(
        self,
        path: str,
        *,
        path_key: str | None = None,
        file_stat: str | None = None,
    ) -> None:
        """Record that *path* is not a valid image."""
        resolved = path_key or os.path.realpath(path)
        stat_key = file_stat or _stat_key(resolved)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO not_images (path, file_stat)"
                " VALUES (?, ?)",
                (resolved, stat_key),
            )
            self._conn.commit()

    def get_text_embeddings(
        self,
        model_id: str,
        texts: Sequence[str],
    ) -> dict[str, NDArray[np.float32]]:
        """Return cached text embeddings for the given prompt strings.

        Returns a dict mapping text -> embedding for cache hits only.
        """
        import numpy as np

        if not texts:
            return {}
        placeholders = ",".join("?" for _ in texts)
        sql = (
            "SELECT text, embedding FROM text_embeddings"
            " WHERE model = ?"
            " AND text IN ({})"
        ).format(placeholders)
        params = [model_id, *texts]
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return {
            text: np.frombuffer(blob, dtype=np.float32).reshape(1, -1).copy()
            for text, blob in rows
        }

    def put_text_embeddings(
        self,
        model_id: str,
        text_embeddings: Sequence[tuple[str, NDArray[np.float32]]],
    ) -> None:
        """Store text embeddings keyed by the actual prompt string."""
        if not text_embeddings:
            return
        sql = (
            "INSERT OR REPLACE INTO text_embeddings"
            " (model, text, embedding)"
            " VALUES (?, ?, ?)"
        )
        payload = [
            (model_id, text, emb.tobytes())
            for text, emb in text_embeddings
        ]
        with self._lock:
            self._conn.executemany(sql, payload)
            self._conn.commit()

    def get_model_id(
        self,
        model_name: str,
        pretrained: str,
    ) -> str | None:
        """Return cached model_id, or None on first use."""
        with self._lock:
            row = self._conn.execute(
                "SELECT model_id FROM model_ids"
                " WHERE model_name = ? AND pretrained = ?",
                (model_name, pretrained),
            ).fetchone()
        return row[0] if row else None

    def put_model_id(
        self,
        model_name: str,
        pretrained: str,
        model_id: str,
    ) -> None:
        """Cache the model_id for (model_name, pretrained)."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO model_ids"
                " (model_name, pretrained, model_id)"
                " VALUES (?, ?, ?)",
                (model_name, pretrained, model_id),
            )
            self._conn.commit()

    def rename_model_id(self, old_id: str, new_id: str) -> None:
        """Rename all cache entries from old_id to new_id.

        Used when the model_id format changes (e.g. bare hf_hub path gains
        a @commit suffix) so existing embeddings don't need to be recomputed.
        """
        with self._lock:
            self._conn.execute(
                "UPDATE embeddings SET model = ? WHERE model = ?",
                (new_id, old_id),
            )
            self._conn.execute(
                "UPDATE text_embeddings SET model = ? WHERE model = ?",
                (new_id, old_id),
            )
            self._conn.execute(
                "UPDATE model_ids SET model_id = ? WHERE model_id = ?",
                (new_id, old_id),
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()
