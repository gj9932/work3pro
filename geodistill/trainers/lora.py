"""Lazy LoRA wiring for the Qwen2.5-VL LLM (paper §3.12).

Local CPU env has no transformers; so this module is intentionally lazy:
``apply_lora`` is the only entry point and raises a clear error if the runtime
cannot satisfy it. On the GPU box, the call composes ``peft.LoraConfig`` with
the LLM half of ``Qwen2_5_VLForConditionalGeneration`` (the visual path stays
frozen — see :mod:`geodistill.models.qwen_visual_frozen`).
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = ["LoRAConfig", "apply_lora"]


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def apply_lora(qwen_for_conditional_generation, cfg: LoRAConfig = LoRAConfig()):
    """Wrap the LLM submodule of a Qwen2.5-VL model with LoRA.

    Returns the (potentially mutated in-place) outer Qwen module so callers can
    keep using the same handle. Raises if peft / transformers is unavailable.
    """
    if cfg.rank != 16:
        # Paper §4.2 fixes rank-16; tolerate other ranks but warn loudly.
        import warnings
        warnings.warn(
            f"LoRA rank={cfg.rank} differs from paper-fixed rank=16; "
            f"keep rank=16 for table 1-6 main rows."
        )
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "peft is required to apply LoRA on the Qwen LLM. "
            "Install requirements_geodistill.txt on the GPU box."
        ) from exc

    lora_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
    )
    # Qwen2.5-VL stores the LLM under ``model``. We apply LoRA there so the
    # vision tower / merger remain untouched (paper §3.13 freeze table).
    if hasattr(qwen_for_conditional_generation, "model"):
        llm = qwen_for_conditional_generation.model
        qwen_for_conditional_generation.model = get_peft_model(llm, lora_cfg)
    else:
        # Fallback: apply to the whole module (caller should have frozen the
        # vision path explicitly).
        qwen_for_conditional_generation = get_peft_model(qwen_for_conditional_generation, lora_cfg)
    return qwen_for_conditional_generation
