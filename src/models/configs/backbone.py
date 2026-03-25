from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneConfig:
    """Backbone configuration for graph/question encoding."""

    embedding_dim: int = 1024
    hidden_dim: int = 512

    use_adapter: bool = True
    adapter_dim: int = 128
    adapter_dropout: float = 0.1


__all__ = ["BackboneConfig"]
