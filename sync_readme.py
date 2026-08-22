#!/usr/bin/env python3
"""Sync the canonical repo-root README.md into rapidsegment/README.md.

PyPI renders the package README (pyproject: readme = "README.md"), while GitHub
renders the root README. Keep the ROOT README as the single source of truth and
run this before building/publishing so the prose/sections stay in sync.

The hero banner is NOT synced: the PyPI README keeps ITS OWN banner image, even
though the root README has a different one. Everything else -- including all
emojis -- is copied exactly from the root README, with the package README's
existing <p align="center"> banner preserved in place.

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

root_text = SRC.read_text(encoding="utf-8")

banner_re = re.compile(r'<p align="center">.*?</p>', re.DOTALL)

# The package README keeps its own banner; only replace the banner if the
# package already has one (otherwise fall back to the root banner on first run).
pkg_banner = ""
if DST.exists():
    pkg_text = DST.read_text(encoding="utf-8")
    m = banner_re.search(pkg_text)
    if m:
        pkg_banner = m.group(0)

if pkg_banner:
    new_text = banner_re.sub(pkg_banner, root_text, count=1)
else:
    new_text = root_text

DST.write_text(new_text, encoding="utf-8")
print(f"Synced {SRC} -> {DST} (package banner preserved; emojis preserved)")
