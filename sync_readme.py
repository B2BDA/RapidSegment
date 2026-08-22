#!/usr/bin/env python3
"""Sync the canonical repo-root README.md into rapidsegment/README.md.

PyPI renders the package README (pyproject: readme = "README.md"), while GitHub
renders the root README. Keep the ROOT README as the single source of truth and
run this before building/publishing so the two stay in sync.

The hero banner image is intentionally skipped in the PyPI copy (GitHub keeps it).
Everything else -- including all emojis -- is copied exactly.

    python sync_readme.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "README.md"
DST = ROOT / "rapidsegment" / "README.md"

if not SRC.exists():
    raise SystemExit(f"Source README not found: {SRC}")

if DST.exists() and SRC.resolve() == DST.resolve():
    print("README is already a single file (no copy needed).")
    raise SystemExit(0)

text = SRC.read_text(encoding="utf-8")

# Skip the hero banner <p align="center">...</p> block (matched by its unique
# image URL so only this exact banner is removed, nothing else).
BANNER_SRC = "f2584720-246b-4bda-b5a2-1b843bbec474"
banner_re = re.compile(
    r'<p align="center">.*?' + re.escape(BANNER_SRC) + r'.*?</p>',
    re.DOTALL,
)
text = banner_re.sub("", text)

DST.write_text(text, encoding="utf-8")
print(f"Synced {SRC} -> {DST} (hero banner skipped; emojis preserved)")
