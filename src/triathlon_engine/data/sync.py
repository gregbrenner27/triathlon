"""Thin wrappers around the GarminDB CLI (GRE-5).

GarminDB owns the download/import pipeline; these helpers just standardise how
the rest of the project (and Greg) invoke it. Credentials and tokens live in
~/.GarminDb / GarminDB's own cache — never in this repo. The SQLite DBs land in
~/HealthData/DBs by default.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# GarminDB's default output location (config: directories.base_dir).
HEALTH_DATA_DIR = Path.home() / "HealthData"
DB_DIR = HEALTH_DATA_DIR / "DBs"

_FULL_SYNC = ["garmindb_cli.py", "--all", "--download", "--import", "--analyze"]
_INCREMENTAL_SYNC = [*_FULL_SYNC, "--latest"]


def full_sync() -> int:
    """Download the entire Garmin history and (re)build the SQLite DBs.

    First run prompts for Garmin Connect MFA — run interactively from a
    terminal, not from automation.
    """
    return subprocess.run(_FULL_SYNC).returncode


def incremental_sync() -> int:
    """Pull only activities/monitoring newer than what's already on disk."""
    return subprocess.run(_INCREMENTAL_SYNC).returncode


def db_paths() -> list[Path]:
    """Return the GarminDB SQLite files currently on disk."""
    return sorted(DB_DIR.glob("*.db")) if DB_DIR.exists() else []


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "latest"
    sys.exit(full_sync() if mode == "full" else incremental_sync())
