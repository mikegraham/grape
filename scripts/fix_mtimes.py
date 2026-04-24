"""Update file_stat entries in a grape cache DB to match current disk mtimes.

Useful after files have been touched/copied/moved -- the cache keys
become stale and every image looks "uncached".

Usage: .venv/bin/python grape_fix_mtimes.py grape.db
"""
import os
import sqlite3
import sys


def _stat_key(path: str) -> str | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (
        f"[{st.st_size}, {st.st_mtime_ns},"
        f" {st.st_ino}, {st.st_dev}, {st.st_ctime_ns}]"
    )


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <cache.db>", file=sys.stderr)
        sys.exit(1)

    db_path = sys.argv[1]
    conn = sqlite3.connect(db_path)

    updated = 0
    missing = 0

    for table in ("embeddings", "not_images"):
        rows = conn.execute(
            f"SELECT rowid, path, file_stat FROM {table}"  # noqa: S608
        ).fetchall()

        for rowid, path, old_stat in rows:
            new_stat = _stat_key(path)
            if new_stat is None:
                missing += 1
                continue
            if new_stat != old_stat:
                conn.execute(
                    f"UPDATE {table} SET file_stat = ? WHERE rowid = ?",  # noqa: S608
                    (new_stat, rowid),
                )
                updated += 1

    conn.commit()
    conn.close()
    print(f"updated {updated} rows, {missing} files missing")


if __name__ == "__main__":
    main()
