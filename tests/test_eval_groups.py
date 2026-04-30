from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.eval.groups import flatten_metric_groups


def test_flatten_metric_groups_with_prefix() -> None:
    groups = {
        "sample": {"expected_target_f1": 0.5},
        "best_of_k": {"max_recall_at_1": 1.0},
    }

    assert flatten_metric_groups(groups, prefix="val") == {
        "val/sample/expected_target_f1": 0.5,
        "val/best_of_k/max_recall_at_1": 1.0,
    }


def test_flatten_metric_groups_without_prefix_or_group() -> None:
    groups = {
        "": {"metric": 2},
    }

    assert flatten_metric_groups(groups) == {"metric": 2.0}
