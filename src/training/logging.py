from __future__ import annotations

from collections.abc import Mapping

from lightning import LightningModule

from src.utils.scalars import Scalar, detach_scalar


def scalarize(
    values: Mapping[str, Scalar],
    *,
    detach: bool = True,
) -> dict[str, Scalar]:
    out: dict[str, Scalar] = {}

    for name, value in values.items():
        out[name] = detach_scalar(value, name=name) if detach else value

    return out


def prefix_keys(
    prefix: str,
    values: Mapping[str, Scalar],
) -> dict[str, Scalar]:
    if not prefix:
        return dict(values)
    return {f"{prefix}/{name}": value for name, value in values.items()}


def merge_metrics(
    *parts: Mapping[str, Scalar],
) -> dict[str, Scalar]:
    merged: dict[str, Scalar] = {}

    for part in parts:
        overlap = merged.keys() & part.keys()
        if overlap:
            raise KeyError(f"Duplicate metric keys are not allowed: {sorted(overlap)}.")
        merged.update(part)

    return merged


def log_scalars(
    module: LightningModule,
    values: Mapping[str, Scalar],
    *,
    prefix: str,
    batch_size: int,
    prog_bar_keys: set[str] | None = None,
    on_step: bool = False,
    on_epoch: bool = True,
    sync_dist: bool = True,
) -> None:
    prog_bar_keys = prog_bar_keys or set()
    safe_values = scalarize(values)

    plain = {name: value for name, value in safe_values.items() if name not in prog_bar_keys}
    prog = {name: value for name, value in safe_values.items() if name in prog_bar_keys}

    if plain:
        module.log_dict(
            prefix_keys(prefix, plain),
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=False,
            batch_size=int(batch_size),
            sync_dist=sync_dist,
        )

    for name, value in prog.items():
        module.log(
            f"{prefix}/{name}" if prefix else name,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            batch_size=int(batch_size),
            sync_dist=sync_dist,
        )
