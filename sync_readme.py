#!/usr/bin/env python3
"""Sync the canonical repo-root README.md into rapidsegment/README.md.

PyPI renders the package README (pyproject: readme = "README.md"), while GitHub
renders the root README. Keep the ROOT README as the single source of truth and
run this before building/publishing so the two stay in sync.

The hero "banner" image is intentionally stripped from the PyPI copy (GitHub
keeps it). Edit README.md (root); everything except the banner is synced.

    python sync_readme.py
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "README.md"
DST = ROOT / "rapidsegment" / "README.md"

if not SRC.exists():
    raise SystemExit(f"Source README not found: {SRC}")

if DST.exists() and SRC.resolve() == DST.resolve():
    print("README is already a single file (no copy needed).")
    raise SystemExit(0)

text = SRC.read_text(encoding="utf-8")

# Strip the hero banner <p align="center">...<img.../></p> for the PyPI copy.
banner_re = re.compile(r'<p align="center">\s*<img[^>]*>\s*</p>', re.IGNORECASE)
text = banner_re.sub("", text)
# Collapse any resulting runs of 3+ newlines back to a single blank line.
text = re.sub(r"\n{3,}", "\n\n", text)

DST.write_text(text, encoding="utf-8")
print(f"Synced {SRC} -> {DST} (banner stripped from PyPI copy)")
