#!/usr/bin/env python3
"""Sync the canonical repo-root README.md into rapidsegment/README.md.

PyPI renders the package README (pyproject: readme = "README.md"), while GitHub
renders the root README. Keep the ROOT README as the single source of truth and
run this before building/publishing so the two stay identical.

    python sync_readme.py
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "README.md"
DST = ROOT / "rapidsegment" / "README.md"

if not SRC.exists():
    raise SystemExit(f"Source README not found: {SRC}")

if DST.exists():
    if SRC.resolve() == DST.resolve():
        print("README is already a single file (no copy needed).")
        raise SystemExit(0)

shutil.copyfile(SRC, DST)
print(f"Synced {SRC} -> {DST}")
