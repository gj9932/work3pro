"""Lazy Qwen2.5-VL LLM glue (paper §3.12).

Provides:
- ``load_qwen_for_training(model_id)`` → returns the full
  ``Qwen2_5_VLForConditionalGeneration`` with vision path frozen.
- ``apply_lora_to_llm(qwen, rank, ...)`` → wraps the LLM with rank-16 LoRA
  via :mod:`geodistill.trainers.lora` (peft).
- ``inject_visual_embeds(qwen, h_geo, ...)`` → splices ``H_geo`` into the LLM
  forward in place of the original ``image_embeds``.
- ``compute_lm_loss(qwen, batch_text, h_geo)`` → autoregressive cross-entropy
  on the language tokens of a QA / planning batch.
- ``extract_pos_hidden(qwen, output, pos_token_id)`` → pull the hidden states
  at every ``<POS>`` position for the coord decoder.

Everything is GPU-only; importing this module on CPU is fine, *calling* the
functions raises ``ImportError`` with the standard hint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


__all__ = [
    "QwenLMConfig",
    "load_qwen_for_training",
    "apply_lora_to_llm",
    "inject_visual_embeds",
    "compute_lm_loss",
    "extract_pos_hidden",
    "RUNTIME_HINT",
]


RUNTIME_HINT = (
    "Qwen LLM glue requires transformers + peft + the Qwen2.5-VL-7B weights. "
    "Run on the GPU box. Locally use --source synthetic and λ_LM=0 to validate "
    "the geometry path without the LLM."
)


@dataclass(frozen=True)
class QwenLMConfig:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    pos_token: str = "<POS>"


def load_qwen_for_training(cfg: QwenLMConfig):
    """Load Qwen2.5-VL-7B with the vision path frozen and the LLM ready for LoRA."""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
    except Exception as exc:                                                       # noqa: BLE001
        raise ImportError(RUNTIME_HINT) from exc

    if cfg.torch_dtype == "bfloat16":
        import torch
        dtype = torch.bfloat16
    elif cfg.torch_dtype in ("fp16", "float16", "half"):
        import torch
        dtype = torch.float16
    else:
        import torch
        dtype = torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_id, trust_remote_code=cfg.trust_remote_code, torch_dtype=dtype,
    )
    # freeze visual + merger explicitly (paper §3.13)
    model.visual.eval()
    for p in model.visual.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if cfg.pos_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([cfg.pos_token], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
    pos_id = tokenizer.convert_tokens_to_ids(cfg.pos_token)
    return model, tokenizer, pos_id


def apply_lora_to_llm(model, rank: int = 16, alpha: int = 32, dropout: float = 0.05,
                      target_modules=("q_proj", "k_proj", "v_proj", "o_proj")):
    from geodistill.trainers.lora import apply_lora, LoRAConfig
    return apply_lora(model, LoRAConfig(rank=rank, alpha=alpha, dropout=dropout,
                                          target_modules=tuple(target_modules)))


def inject_visual_embeds(model, h_geo, image_grid_thw, input_ids, attention_mask=None,
                          labels=None, image_token_id: int | None = None):
    """Run the LLM forward with ``H_geo`` as the visual token embedding.

    Qwen2.5-VL replaces ``<|image_pad|>`` placeholders in ``input_ids`` with the
    image embedding sequence at runtime. The standard model.forward() accepts
    ``inputs_embeds`` overrides; we precompute the input embeddings, splice
    H_geo into the image-pad slots, and call the LLM half directly.

    This implementation matches the public Qwen2.5-VL inference helper
    (``model.get_input_embeddings()`` + scatter at image placeholders); on the
    GPU box you may want to swap this for the official
    ``Qwen2_5_VLForConditionalGeneration.forward`` with ``pixel_values`` so
    the merger runs on the same device — see Qwen2.5-VL HF docs.
    """
    try:
        import torch
    except Exception as exc:                                                       # noqa: BLE001
        raise ImportError(RUNTIME_HINT) from exc

    if image_token_id is None:
        # Qwen2.5-VL default; verify against your tokenizer if you change ckpts.
        image_token_id = getattr(model.config, "image_token_id", 151655)

    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask = (input_ids == image_token_id)
    if int(image_mask.sum()) != int(h_geo.shape[0]):
        raise ValueError(
            f"H_geo has {h_geo.shape[0]} tokens but {int(image_mask.sum())} image-pad placeholders; "
            f"mismatched image_grid_thw or template."
        )
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[image_mask] = h_geo.to(inputs_embeds.dtype)

    return model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
        return_dict=True,
    )


def compute_lm_loss(model, h_geo, image_grid_thw, input_ids, attention_mask=None, labels=None):
    """Wrap inject_visual_embeds and return the LM cross-entropy."""
    out = inject_visual_embeds(
        model, h_geo, image_grid_thw, input_ids,
        attention_mask=attention_mask, labels=labels,
    )
    if out.loss is None:
        raise ValueError("compute_lm_loss expects ``labels`` so HF returns a loss")
    return out.loss, out


def extract_pos_hidden(out, input_ids, pos_token_id: int):
    """Return (B, T_pos, hidden) hidden states at every <POS> position.

    ``out`` is the HF Output object from ``inject_visual_embeds``; we use the
    last hidden state.
    """
    hs = out.hidden_states[-1]                                  # (B, T, hidden)
    mask = (input_ids == pos_token_id)
    if int(mask.sum()) == 0:
        return hs.new_zeros((hs.shape[0], 0, hs.shape[-1]))
    # Per-batch list — pad to max length so the coord decoder can run a single MLP
    max_pos = int(mask.sum(dim=1).max())
    out_seq = hs.new_zeros((hs.shape[0], max_pos, hs.shape[-1]))
    for b in range(hs.shape[0]):
        idx = mask[b].nonzero(as_tuple=True)[0]
        if idx.numel():
            out_seq[b, :idx.numel()] = hs[b, idx]
    return out_seq
