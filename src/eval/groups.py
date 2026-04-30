from __future__ import annotations

from collections.abc import Mapping

MetricDict = dict[str, float]
MetricGroups = dict[str, MetricDict]


def flatten_metric_groups(
    groups: Mapping[str, Mapping[str, float]],
    *,
    prefix: str | None = None,
    sep: str = "/",
) -> dict[str, float]:
    flat: dict[str, float] = {}
    for group_name, metrics in groups.items():
        for name, value in metrics.items():
            key_parts: list[str] = []
            if prefix:
                key_parts.append(prefix)
            if group_name:
                key_parts.append(group_name)
            key_parts.append(name)
            flat[sep.join(key_parts)] = float(value)
    return flat


__all__ = ["MetricDict", "MetricGroups", "flatten_metric_groups"]
