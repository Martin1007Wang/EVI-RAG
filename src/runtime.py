from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def find_project_root(
    start: str | Path | None = None,
    *,
    indicator: str = ".project-root",
) -> Path:
    search_roots: list[Path] = []
    if start is not None:
        start_path = Path(start).resolve()
        search_roots.append(start_path.parent if start_path.is_file() else start_path)
    search_roots.append(Path.cwd().resolve())
    search_roots.append(Path(__file__).resolve().parents[1])

    seen: set[Path] = set()
    for root in search_roots:
        for candidate in (root, *root.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / indicator).exists():
                return candidate

    return Path(__file__).resolve().parents[1]


def load_project_env(start: str | Path | None = None) -> Path:
    project_root = find_project_root(start)
    load_dotenv(project_root / ".env")
    return project_root

