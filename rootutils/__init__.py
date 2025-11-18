"""Minimal local stub of rootutils for tests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_root(*, indicator: str = ".project-root") -> Path:
    """Walk upward from cwd to find the first directory containing ``indicator``.

    This mirrors the subset of functionality required by the official rootutils.
    """

    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / indicator).exists():
            return parent
    raise RuntimeError(f"Failed to locate project root using indicator {indicator!r}")


def setup_root(
    file: str,
    *,
    indicator: str = ".project-root",
    pythonpath: bool = False,
) -> Path:
    """Return the root path and optionally append it to PYTHONPATH.

    Only the subset used inside this repository is implemented.
    """

    root = find_root(indicator=indicator)
    if pythonpath:
        import sys

        root_str = str(root)
        if root_str not in sys.path:
            sys.path.append(root_str)
    return root

