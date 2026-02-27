"""SQLite-backed embedding cache.

Avoids redundant CLIP encoding by caching embeddings keyed on
(absolute_path, model_id) with file-stat-based invalidation.
"""

import json
import os
import sqlite3
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
    return json.dumps([
        st.st_size, st.st_mtime_ns,
        st.st_ino, st.st_dev, st.st_ctime_ns,
    ])


class EmbeddingCache:
    """Read-through cache for CLIP image embeddings stored in SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_EMBEDDINGS)
        self._conn.execute(_CREATE_NOT_IMAGES)
        self._conn.commit()

    def get(
        self, path: Path, model_id: str,
    ) -> NDArray[np.float32] | None:
        """Return the cached embedding, or ``None`` on miss/stale."""
        resolved = os.path.realpath(path)
        row = self._conn.execute(
            "SELECT file_stat, embedding FROM embeddings"
            " WHERE path = ? AND model = ?",
            (resolved, model_id),
        ).fetchone()
        if row is None:
            return None
        stored_stat, blob = row
        if stored_stat != _stat_key(resolved):
            return None
        arr: NDArray[np.float32] = (
            np.frombuffer(blob, dtype=np.float32)
            .reshape(1, -1)
            .copy()
        )
        return arr

    def put(
        self, path: Path, model_id: str, embedding: NDArray[np.float32],
    ) -> None:
        """Insert or replace the cached embedding for *(path, model)*."""
        resolved = os.path.realpath(path)
        self._conn.execute(
            _INSERT,
            (resolved, _stat_key(resolved),
             model_id, embedding.tobytes()),
        )
        self._conn.commit()

    def is_not_image(self, path: Path) -> bool:
        """Return ``True`` if *path* was previously recorded as not an image."""
        resolved = os.path.realpath(path)
        row = self._conn.execute(
            "SELECT file_stat FROM not_images WHERE path = ?",
            (resolved,),
        ).fetchone()
        if row is None:
            return False
        stored_stat: str = row[0]
        return stored_stat == _stat_key(resolved)

    def put_not_image(self, path: Path) -> None:
        """Record that *path* is not a valid image."""
        resolved = os.path.realpath(path)
        self._conn.execute(
            "INSERT OR REPLACE INTO not_images (path, file_stat)"
            " VALUES (?, ?)",
            (resolved, _stat_key(resolved)),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
