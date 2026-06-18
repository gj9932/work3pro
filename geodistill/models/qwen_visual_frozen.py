"""Frozen Qwen2.5-VL native vision path.

This module wraps the HuggingFace Qwen2.5-VL model so the geometry adapter can
read native ``image_embeds`` (output of the spatial merger) without modifying the
vision encoder or the merger.

Forward returns the merged-token feature map (``H_img``) and the per-token meta
required by downstream LPGA components:

    H_img      ∈ R^{N × hidden_size}   (hidden_size = 3584 for Qwen2.5-VL-7B)
    grid_thw   ∈ N × 3                 (per-image (T, H, W) describing token layout)
    cam_offset ∈ N_cam + 1             (boundary into ``H_img`` per camera)
    token_uv   ∈ R^{N × 2}             (image-space center of each merged token, in pixels)
    token_box  ∈ R^{N × 4}             (image-space (u0, v0, u1, v1) of each merged token)

References:
- ``paper/work3pro_cvpr2027_draft5_zh_qwen.md`` §3.1, §3.2.
- ``plan/Task_work3_1pro.md`` M0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
    import torch.nn as nn
except ImportError:  # let import fail loudly only when actually used
    torch = None
    nn = None  # type: ignore

# transformers is required at runtime; we avoid importing it at module load so
# the package can be imported in environments where the legacy work3 deps live
# alongside (torch 1.12).  Real users instantiate ``QwenVisualFrozen`` from the
# geodistill env defined by ``requirements_geodistill.txt``.


@dataclass
class QwenVisualOutput:
    """Container for the frozen Qwen vision-path forward result."""

    image_embeds: "torch.Tensor"        # (N, hidden_size)
    grid_thw: "torch.Tensor"            # (N_img, 3)
    cam_offsets: "torch.Tensor"         # (N_cam + 1,) cumulative split into image_embeds
    token_uv: "torch.Tensor"            # (N, 2) pixel-space token center
    token_box: "torch.Tensor"           # (N, 4) pixel-space (u0, v0, u1, v1)
    token_camera_id: "torch.Tensor"     # (N,) long, camera index per token


class QwenVisualFrozen(nn.Module if nn is not None else object):
    """Wrap ``Qwen2_5_VLForConditionalGeneration`` with all visual params frozen.

    The wrapper does not run the LLM: it only invokes the vision tower and the
    merger.  The merger's output is what the LPGA consumes; everything below is
    geometric metadata derived from ``image_grid_thw``.
    """

    SPATIAL_MERGE: int = 2  # Qwen2.5-VL spatial merger collapses 2x2 patch tokens
    PATCH_SIZE: int = 14    # Qwen2.5-VL ViT patch

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        device_map: str | dict | None = None,
        attn_implementation: str | None = None,
    ) -> None:
        if torch is None or nn is None:
            raise ImportError("torch is required for QwenVisualFrozen")
        super().__init__()

        from transformers import AutoConfig, AutoProcessor  # noqa: WPS433
        from transformers import Qwen2_5_VLForConditionalGeneration

        dtype = _resolve_dtype(torch_dtype)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        load_kwargs: dict = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": dtype,
        }
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        # We load the full model (the LLM head is unused but cheap to host).  In
        # practice training scripts will wrap this with LoRA over the LLM.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        # 冻结整个 vision tower 与 merger,LLM 在 Stage B 才解锁 LoRA。
        self._freeze_vision_path()

        merger_dim = getattr(self.config, "hidden_size", None) or self.config.text_config.hidden_size
        self.hidden_size: int = int(merger_dim)
        # Qwen 处理器把 image factor 设置为 patch_size * spatial_merge_size。
        image_processor = self.processor.image_processor
        self.patch_size = int(getattr(image_processor, "patch_size", self.PATCH_SIZE))
        self.spatial_merge = int(getattr(image_processor, "merge_size", self.SPATIAL_MERGE))

    # ---------------------------------------------------------------- freeze
    def _freeze_vision_path(self) -> None:
        visual = self.model.visual  # Qwen2_5_VisionTransformer
        for p in visual.parameters():
            p.requires_grad_(False)
        visual.eval()

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def assert_vision_frozen(self) -> None:
        bad = [n for n, p in self.model.visual.named_parameters() if p.requires_grad]
        if bad:
            raise RuntimeError(f"Qwen vision path has trainable params: {bad[:4]}...")

    # ----------------------------------------------------------------- core
    def encode_images(
        self,
        pixel_values: "torch.Tensor",
        image_grid_thw: "torch.Tensor",
        camera_widths: Sequence[int] | None = None,
        camera_heights: Sequence[int] | None = None,
    ) -> QwenVisualOutput:
        """Run the frozen visual path.

        Args:
            pixel_values: Tensor as produced by ``Qwen2VLImageProcessor`` (already
                packed with grid info; shape depends on processor version).
            image_grid_thw: (N_img, 3) ``(t, h, w)`` per camera.
            camera_widths / camera_heights: optional original-image sizes; used to
                map merged-token centers back to pixel space.  When omitted, we
                fall back to ``h * patch_size``.
        """
        if torch is None:
            raise ImportError("torch is required to call encode_images")

        device = next(self.model.visual.parameters()).device
        pixel_values = pixel_values.to(device=device, dtype=self.model.visual.dtype)
        image_grid_thw = image_grid_thw.to(device=device, dtype=torch.long)

        with torch.no_grad():
            image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        # image_embeds: (N_total_merged_tokens, hidden_size)
        meta = self._build_token_meta(image_grid_thw, camera_widths, camera_heights, device)
        return QwenVisualOutput(
            image_embeds=image_embeds,
            grid_thw=image_grid_thw,
            cam_offsets=meta["cam_offsets"],
            token_uv=meta["token_uv"],
            token_box=meta["token_box"],
            token_camera_id=meta["token_camera_id"],
        )

    # -------------------------------------------------------------- helpers
    def _build_token_meta(
        self,
        grid_thw: "torch.Tensor",
        camera_widths: Sequence[int] | None,
        camera_heights: Sequence[int] | None,
        device: "torch.device",
    ) -> dict[str, "torch.Tensor"]:
        n_img = int(grid_thw.shape[0])
        if camera_widths is not None and len(camera_widths) != n_img:
            raise ValueError("camera_widths must align with image_grid_thw")
        if camera_heights is not None and len(camera_heights) != n_img:
            raise ValueError("camera_heights must align with image_grid_thw")

        merge = self.spatial_merge
        patch = self.patch_size
        merged_stride = merge * patch  # pixel size per merged token edge

        cam_offsets = [0]
        token_uv: list[torch.Tensor] = []
        token_box: list[torch.Tensor] = []
        token_camera_id: list[torch.Tensor] = []

        for cam_idx in range(n_img):
            t, h, w = (int(v) for v in grid_thw[cam_idx].tolist())
            if t != 1:
                raise NotImplementedError(
                    "GeoDistill currently assumes per-camera single-frame inputs (t=1)"
                )
            merged_h = h // merge
            merged_w = w // merge
            n_tokens = merged_h * merged_w
            cam_offsets.append(cam_offsets[-1] + n_tokens)

            # 像素空间下每个 merged token 的中心与 box。若调用方提供原图尺寸,
            # 我们等比缩放到原图坐标;否则使用 patch * merge 像素的内部坐标。
            inner_w = merged_w * merged_stride
            inner_h = merged_h * merged_stride
            scale_x = (camera_widths[cam_idx] / inner_w) if camera_widths is not None else 1.0
            scale_y = (camera_heights[cam_idx] / inner_h) if camera_heights is not None else 1.0

            ys = torch.arange(merged_h, device=device, dtype=torch.float32)
            xs = torch.arange(merged_w, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

            u0 = grid_x * merged_stride * scale_x
            v0 = grid_y * merged_stride * scale_y
            u1 = (grid_x + 1) * merged_stride * scale_x
            v1 = (grid_y + 1) * merged_stride * scale_y
            uc = (u0 + u1) * 0.5
            vc = (v0 + v1) * 0.5

            token_uv.append(torch.stack((uc.reshape(-1), vc.reshape(-1)), dim=-1))
            token_box.append(
                torch.stack(
                    (u0.reshape(-1), v0.reshape(-1), u1.reshape(-1), v1.reshape(-1)),
                    dim=-1,
                )
            )
            token_camera_id.append(torch.full((n_tokens,), cam_idx, device=device, dtype=torch.long))

        return {
            "cam_offsets": torch.as_tensor(cam_offsets, device=device, dtype=torch.long),
            "token_uv": torch.cat(token_uv, dim=0),
            "token_box": torch.cat(token_box, dim=0),
            "token_camera_id": torch.cat(token_camera_id, dim=0),
        }


def _resolve_dtype(name: str):
    if torch is None:
        raise ImportError("torch is required to resolve dtype")
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float16", "half", "fp16"):
        return torch.float16
    if name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name!r}")
