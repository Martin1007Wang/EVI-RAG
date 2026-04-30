# EVI-RAG Agent Guide

This file is for coding agents working in `/mnt/wangjingxiong/EVI-RAG`.
It is intentionally repo-specific and should be preferred over generic Python advice.
If repo behavior and this file disagree, follow the source files and update this document.

## Scope and rule files

- Repository type: Python ML/research codebase built around Hydra, Lightning, PyTorch, and pytest.
- Main runtime entrypoints live in `src/train.py`, `src/evaluate.py`, `src/preprocess.py`, and `src/datasets/build_edge_retrieval_labels.py`.
- Config composition is centralized in `configs/`.
- Tests live in `tests/` and `tests/answer_reachability/`.
- No existing top-level `AGENTS.md`, `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` were found when this file was created.
- There is no Ruff, mypy, or pyright configuration in the repo today.

## Environment and setup

- Preferred local Python from `environment.yaml` is Python 3.10.
- Default conda environment for this repo is `pog`; when running Python commands, assume `conda activate pog` unless the user says otherwise.
- CI runs tests on Python 3.8, 3.9, and 3.10, so avoid using syntax newer than Python 3.8 without checking compatibility.
- `pyupgrade --py38-plus` is part of pre-commit, so Python 3.8+ idioms are acceptable.
- CI bootstrap: `python -m pip install --upgrade pip` then `pip install -r requirements.txt`.
- Conda bootstrap: `conda env create -f environment.yaml` then `conda activate myenv`.
- Editable install is optional: `pip install -e .` adds `train_command`, `evaluate_command`, and `preprocess_command` from `setup.py`.
- The repo contains a local `rootutils` stub for tests; do not replace it casually.
- `.env` exists at repo root; treat it as sensitive and do not commit or rewrite it unless explicitly asked.

## Build, lint, format, and test commands

- There is no dedicated package build command or PEP 517 build backend configured.
- For most work, treat install + tests + Hydra entrypoints as the relevant execution surface.
- Discover Make targets with `make help`; cleanup commands are `make clean` and `make clean-logs`.
- Run repo-wide formatting/lint with `make format` (`pre-commit run -a`).
- Run hooks on changed files with `pre-commit run --files path/to/file.py`.
- Run a single hook with `pre-commit run black --files path/to/file.py`, `pre-commit run isort --files ...`, or `pre-commit run flake8 --files ...`.
- Fast tests: `make test` or `pytest -k "not slow"`.
- Full tests: `make test-full` or `pytest -v`.
- Coverage: `pytest --cov src`.
- Single file: `pytest tests/test_question_context_preprocessing.py`.
- Single test: `pytest tests/test_question_context_preprocessing.py::test_write_questions_with_token_context_roundtrip`.
- One trajectory test: `pytest tests/answer_reachability/test_search_exactness.py::test_search_exact_top_order`.
- Keyword subset: `pytest tests/answer_reachability -k normalization`.
- Slow-only runs: `pytest -m slow`.
- Pytest defaults already enable `--strict-markers`, `--doctest-modules`, `--durations=0`, and `log_cli=True`.
- Because doctests are enabled, broken examples in docstrings can fail test runs.

## CI expectations

- PR code-quality checks run `pre-commit` on changed files only.
- Main-branch code-quality checks run `pre-commit` across the repo.
- Test CI runs on Ubuntu, macOS, and Windows.
- Test CI covers Python 3.8, 3.9, and 3.10.
- Coverage CI uses `pytest --cov src`.

## Hydra and runtime commands

- When training or evaluation needs GPU, first attach the existing tmux session with `tmux attach -t train` and run the command inside that session.
- Do not launch GPU training/eval jobs from a fresh shell when the `train` tmux session is available; use that session as the default execution context.
- Train via Hydra: `python src/train.py experiment=train/webqsp_baseline`.
- Evaluate a checkpoint: `python src/evaluate.py experiment=eval/webqsp ckpt.path=/path/to/model.ckpt`.
- Rebuild retrieval preprocess outputs: `python src/preprocess.py dataset=webqsp`.
- Build edge retriever labels: `python src/datasets/build_edge_retrieval_labels.py dataset=webqsp-sub`.
- `make train` runs `python src/train.py` with the default config.
- Training defaults to `dataset=webqsp`; override the dataset group explicitly when needed.
- Evaluation uses `configs/evaluate.yaml` and the shared `trainer` group.
- Hydra config mutation should use `open_dict`, not raw attribute writes on frozen/structured configs.
- `print_config` now lives at the root task config (`train.yaml`, `evaluate.yaml`, `preprocess.yaml`).

## Formatting and import rules

- Formatting is enforced by Black with a 99 character line length.
- Import sorting is enforced by isort with `--profile black`.
- Keep imports grouped as standard library, third-party, then local `src` imports.
- Preserve blank lines between import groups and let isort settle exact ordering.
- `flake8` ignores `E402`, and that is intentional for entrypoints.
- In entry scripts, call `rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)` before importing `src.*` modules.
- Do not "fix" entrypoint import order by moving local imports above `rootutils.setup_root(...)`.
- `flake8` also ignores `E203`, `E501`, `F401`, and `F841`; still prefer clean imports and no dead code.
- Markdown, YAML, shell, notebooks, spelling, and security checks also run through pre-commit.

## Typing and data modeling

- Newer modules usually start with `from __future__ import annotations`; prefer that in new files.
- Prefer built-in generics in new code, for example `list[str]`, `dict[str, Any]`, and `Path | None`.
- Match the surrounding file if it still uses `typing.List`, `typing.Dict`, or `Optional`.
- Use `TYPE_CHECKING` blocks for type-only imports when a runtime import would be heavy or optional.
- Optional third-party imports are often wrapped in `_require_*` helpers or guarded `try/except ModuleNotFoundError` blocks.
- Frozen dataclasses are common for configs and immutable runtime records.
- Validate dataclass inputs in `__post_init__` with explicit constraints.
- Tensor-heavy structures often encode shape contracts in type names and runtime validation.
- When mutating Hydra configs, convert or resolve carefully; `OmegaConf.to_container(...)` is common at module boundaries.
- Avoid silently passing through invalid configs; this repo prefers explicit validation.

## Naming conventions and structure

- Use `snake_case` for functions, variables, and test names.
- Use `PascalCase` for classes.
- Use `UPPER_SNAKE_CASE` for module constants.
- Prefix private helpers with `_`; this repo has many `_resolve_*`, `_normalize_*`, `_validate_*`, and `_require_*` helpers.
- Config dataclasses usually end in `Config`.
- Runtime orchestration functions are thin and often named `train`, `evaluate`, or `main`; keep business logic in helpers, utility modules, or model/data modules.
- Use `__all__` where a module intentionally exposes a curated public surface.
- Metric names commonly use slash-separated namespaces like `train/loss`, `val/sub/...`, and `llm/subgraphrag/full/hit@1`.

## Error handling, logging, and side effects

- Fail fast with specific exception types.
- Use `ValueError` for invalid config values, unsupported modes, or malformed user inputs.
- Use `TypeError` for contract violations on object shape or type.
- Use `RuntimeError` for invalid runtime state, missing trainable params, or impossible execution conditions.
- Use `FileNotFoundError` for required data/config artifacts that are missing.
- Error messages should be actionable, usually include the bad value, and often include a fix hint or example Hydra command.
- Avoid swallowing exceptions.
- Broad exception handling is mostly limited to top-level wrappers that log and re-raise.
- Prefer repo helpers like `get_logger`, `RankedLogger`, `log_event`, and `log_metric` from `src.utils.logging_utils`.
- Prefer structured log events with stable field names over ad hoc `print(...)` calls.

## Testing conventions

- Tests are written with plain pytest functions, not unittest classes.
- Small deterministic builders in `tests/answer_reachability/conftest.py` are the preferred style for tensor fixtures.
- Direct tests of underscore-prefixed helpers are acceptable when those helpers are part of a stable internal contract.
- Use `pytest.raises(..., match=...)` for failure cases; exact error text matters in this repo.
- Use `pytest.approx(...)` for floating-point invariants.
- Keep test data minimal and explicit; many tests build tiny manual graphs or JSON payloads inline.
- When adding a helper with important validation logic, add a focused unit test near similar files instead of only integration coverage.
- Prefer targeted test runs while iterating, then run `make test` or `pytest` before finishing.

## Repo-specific content style

- Preserve mixed English and Chinese comments/messages when editing nearby code; that mix already exists in configs and model modules.
- Keep comments brief and high-signal.
- Favor architecture comments that explain why a constraint exists, not line-by-line narration.
- Keep data/config files as the source of truth for paths and defaults; Python should validate, adapt, and orchestrate.
- When working near Hydra configs, preserve the `defaults:` ordering because override order matters.
- When editing docs or Markdown, expect `mdformat` to normalize numbering and tables.

## Terminology memo

- Treat the math view and the implementation view as two aligned descriptions of the same state, not as competing definitions.
- Theoretical/MDP language may describe the state as a full semantic subgraph with node semantics/features; implementation code factorizes that state into static graph context plus a lightweight dynamic trace for memory efficiency.
- When explaining `State`, make it clear that stored `edge_ids` are the dynamic handle used to reconstruct the full Markov state on demand from the prepared batch.
- Use two naming layers consistently: `question_*` for question-linked entities before/at data materialization, and `anchor_*` for the in-graph grounded start nodes used by rollout/search.
- Avoid reviving legacy synonyms such as `topic`, bare `start`, or `q_entity`/`seed_entity_ids` in new code unless you are intentionally handling backward-compatibility.
- The checked-in `docs/` directory is intentionally obsolete; do not recreate stale repo Markdown docs when updating semantics. Put durable repo guidance in `AGENTS.md` comments/memos instead.

## Practical agent checklist

- Read the relevant Hydra config before changing training or evaluation behavior.
- Check `Makefile`, `pyproject.toml`, and `.pre-commit-config.yaml` before inventing commands.
- Prefer unit tests under `tests/` over expensive end-to-end training runs.
- Be careful with data paths in `configs/paths/default.yaml`; they point to absolute `/mnt/data/...` locations.
- Do not assume CPU eval is valid just because Lightning supports it; this repo intentionally blocks most CPU eval flows.
- If you add a new workflow rule file later, update this `AGENTS.md` so future agents see it immediately.
