"""SQLite-backed embedding cache.

Avoids redundant CLIP encoding by caching embeddings keyed on
(absolute_path, model_id) with file-stat-based invalidation.
"""

import os
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

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

_INSERT = (
    "INSERT OR REPLACE INTO embeddings"
    " (path, file_stat, model, embedding)"
    " VALUES (?, ?, ?, ?)"
)


def _stat_key(path: str) -> str:
    """JSON array of stat fields used for cache invalidation."""
    st = os.stat(path)
    return _stat_key_from_stat(st)


def _stat_key_from_stat(st: os.stat_result) -> str:
    """JSON array of stat fields used for cache invalidation."""
    # Keep exact spacing compatible with json.dumps(list) to preserve
    # cache-key stability while reducing JSON encoder overhead.
    return (
        f"[{st.st_size}, {st.st_mtime_ns},"
        f" {st.st_ino}, {st.st_dev}, {st.st_ctime_ns}]"
    )


class EmbeddingCache:
    """Read-through cache for CLIP image embeddings stored in SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_CREATE_EMBEDDINGS)
            self._conn.execute(_CREATE_NOT_IMAGES)
            self._conn.commit()

    def get(
        self,
        path: Path,
        model_id: str,
        *,
        path_key: str | None = None,
        file_stat: str | None = None,
    ) -> NDArray[np.float32] | None:
        """Return the cached embedding, or ``None`` on miss/stale."""
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
        if stored_stat != stat_key:
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
                if expected_stat is None or row_stat != expected_stat:
                    continue
                out[row_path] = np.frombuffer(blob, dtype=np.float32).copy()
        return out

    def embedding_index_for_model(
        self,
        model_id: str,
    ) -> dict[tuple[str, str], NDArray[np.float32]]:
        """Return in-memory ``(path, file_stat) -> embedding`` for a model."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, file_stat, embedding FROM embeddings WHERE model = ?",
                (model_id,),
            ).fetchall()
        return {
            (path, file_stat): np.frombuffer(blob, dtype=np.float32).copy()
            for path, file_stat, blob in rows
        }

    def has_any_embedding(
        self,
        path: Path,
        *,
        path_key: str | None = None,
        file_stat: str | None = None,
    ) -> bool:
        """Return ``True`` if any cached embedding matches current file stat."""
        resolved = path_key or os.path.realpath(path)
        stat_key = file_stat or _stat_key(resolved)
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM embeddings"
                " WHERE path = ? AND file_stat = ?"
                " LIMIT 1",
                (resolved, stat_key),
            ).fetchone()
        return row is not None

    def image_hit_index(self) -> set[tuple[str, str]]:
        """Return ``(path, file_stat)`` pairs known to have embeddings."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT path, file_stat FROM embeddings"
            ).fetchall()
        return {(path, file_stat) for path, file_stat in rows}

    def not_image_index(self) -> set[tuple[str, str]]:
        """Return ``(path, file_stat)`` pairs known to be non-images."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, file_stat FROM not_images"
            ).fetchall()
        return {(path, file_stat) for path, file_stat in rows}

    def put(
        self,
        path: Path,
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
        rows: Sequence[tuple[Path, NDArray[np.float32], str | None, str | None]],
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
        path: Path,
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
        return stored_stat == stat_key

    def put_not_image(
        self,
        path: Path,
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

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()
