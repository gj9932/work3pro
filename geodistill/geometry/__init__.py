"""Geometry helpers for LPGA / relation adapter (paper §3.7-3.10).

Public surface:
- ``decode_ray``      — δ_hat -> sub-token pixel (u_bar, v_bar)
- ``back_project``    — (u_bar, v_bar, d_hat, K, cam_to_ego) -> ego coord
- ``Universal3DPE``   — sin/cos 3D positional encoder used by the injection
- ``RelationGraphLabels`` etc. live in cross_view_label.py (Chunk 3)
"""

from .sub_token_ray import decode_ray, back_project
from .pe_3d import Universal3DPE

__all__ = ["decode_ray", "back_project", "Universal3DPE"]
