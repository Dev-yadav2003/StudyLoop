"""A throwaway PostgreSQL cluster for integration tests.

Resolution order:
  1. TEST_DATABASE_URL  — use an existing server (CI service container). Never Supabase prod.
  2. local binaries     — `initdb` a fresh cluster in a temp dir on a free port, start it,
                          and delete everything afterwards. PG_BIN overrides the search.
  3. neither            — integration tests are skipped.

Durability is switched off (fsync etc.): the data is disposable and tests run faster.
"""

from __future__ import annotations

import glob
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def find_pg_bin() -> Path | None:
    if env := os.environ.get("PG_BIN"):
        return Path(env)
    if found := shutil.which("pg_ctl"):
        return Path(found).parent
    if sys.platform == "win32":
        candidates = sorted(glob.glob(r"C:\Program Files\PostgreSQL\*\bin"), reverse=True)
        for candidate in candidates:
            if (Path(candidate) / "initdb.exe").is_file():
                return Path(candidate)
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def ephemeral_postgres(bin_dir: Path) -> Iterator[str]:
    root = Path(tempfile.mkdtemp(prefix="lecnotes-pg-"))
    data, log = root / "data", root / "server.log"
    port = _free_port()
    options = (
        f"-p {port} -h 127.0.0.1 -c fsync=off -c full_page_writes=off -c synchronous_commit=off"
    )

    def pg(tool: str, *args: str) -> None:
        # Never capture output: `pg_ctl start` spawns the server, which inherits our pipes,
        # so waiting for EOF would hang for the server's whole lifetime (seen on Windows).
        done = subprocess.run(
            [str(bin_dir / tool), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )
        if done.returncode != 0:
            detail = log.read_text(errors="replace")[-2000:] if log.exists() else ""
            raise RuntimeError(f"{tool} {args[-1]} failed ({done.returncode})\n{detail}")

    try:
        pg("initdb", "-D", str(data), "-U", "postgres", "-A", "trust", "-E", "UTF8",
           "--no-locale", "--no-sync")  # fmt: skip
        pg("pg_ctl", "-D", str(data), "-o", options, "-l", str(log), "-w", "start")
        try:
            yield f"postgresql+psycopg://postgres@127.0.0.1:{port}/postgres"
        finally:
            pg("pg_ctl", "-D", str(data), "-m", "immediate", "-w", "stop")
    finally:
        shutil.rmtree(root, ignore_errors=True)
