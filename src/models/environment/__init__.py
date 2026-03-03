# src/models/environment/__init__.py
"""
[系统模块] 图环境状态管理
负责图结构解析和状态制备
"""

from .builder import GraphEnvironmentBuilder
from .context import CsrAdjacency, GraphEnvContext
from .ops import (
    FlowDirection,
    build_node_membership_mask,
    has_super_source_layout,
    infer_super_source_absolute_indices,
    resolve_super_source_absolute_indices,
)
from .state import DynamicAgentState


__all__ = [
    "GraphEnvironmentBuilder",
    "CsrAdjacency",
    "GraphEnvContext",
    "DynamicAgentState",
    "build_node_membership_mask",
    "FlowDirection",
    "has_super_source_layout",
    "infer_super_source_absolute_indices",
    "resolve_super_source_absolute_indices",
]
