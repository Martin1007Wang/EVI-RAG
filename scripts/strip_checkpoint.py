#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch


def _load_checkpoint(path: Path) -> Dict[str, Any] | Dict[str, torch.Tensor]:
    if "weights_only" in torch.load.__code__.co_varnames:
        # 显式使用完整反序列化；仅用于本地可信 checkpoint 的一次性清洗。
        return torch.load(str(path), map_location="cpu", weights_only=False)
    return torch.load(str(path), map_location="cpu")


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Expected state_dict mapping, got {type(state)!r}")
    return state  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip Lightning checkpoint into a pure state_dict-only file (trusted checkpoints only)."
    )
    parser.add_argument("--in", dest="in_path", required=True, help="Input checkpoint path")
    parser.add_argument("--out", dest="out_path", required=True, help="Output path for stripped state_dict")
    args = parser.parse_args()

    in_path = Path(args.in_path).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = _load_checkpoint(in_path)
    state = _extract_state_dict(ckpt)
    torch.save(state, str(out_path))
    print(f"Saved stripped state_dict to: {out_path}")


if __name__ == "__main__":
    main()
