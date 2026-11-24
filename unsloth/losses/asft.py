# SPDX-License-Identifier: Apache-2.0
"""Token-level ASFT (arXiv:2509.23753v3), using model-returned logits."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint


@dataclass
class ASFTStreamingConfig:
    enabled: bool = False
    ref_strategy: str = "none"
    ref_microbatch_size: Optional[int] = None
    seq_chunk_size: Optional[int] = None
    kl_token_chunk_size: Optional[int] = 256
    force_fp32_kl: bool = True

    def validate(self):
        if self.ref_strategy not in ("none", "batch_micro", "seq_kv_cache"):
            raise ValueError("Unknown ASFT reference strategy")
        if self.enabled and self.ref_strategy == "seq_kv_cache":
            raise NotImplementedError("KV-cache streaming is not validated on current Unsloth")
        for value in (self.ref_microbatch_size, self.seq_chunk_size, self.kl_token_chunk_size):
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError("ASFT chunk sizes must be positive integers")
        if not self.force_fp32_kl:
            raise ValueError("ASFT requires FP32-or-higher loss arithmetic")


def validate_objective(mode, kl_weight, kl_direction):
    if mode not in ("sft", "dft", "sft+kl", "asft"):
        raise ValueError(f"Unknown ASFT mode: {mode}")
    if not math.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError("kl_weight must be finite and nonnegative")
    if kl_direction not in ("forward", "reverse"):
        raise ValueError("kl_direction must be forward or reverse")


def needs_reference(mode, kl_weight):
    return mode in ("asft", "sft+kl") and kl_weight != 0


def token_loss(policy, targets, reference=None, *, mode="asft", kl_weight=0.03, kl_direction="forward"):
    """Sum over supervised positions. Reference and DFT confidence are detached.

    Inputs are already-effective logits; do not apply model softcaps/scaling twice.
    Caller selects valid causal positions before invoking this function.
    """
    validate_objective(mode, kl_weight, kl_direction)
    dtype = torch.float64 if policy.dtype == torch.float64 else torch.float32
    logp = policy.to(dtype).log_softmax(-1)
    ce = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    loss = ce * (-ce.detach()).exp() if mode in ("dft", "asft") else ce
    if needs_reference(mode, kl_weight):
        if reference is None or reference.shape != policy.shape:
            raise ValueError("ASFT needs matching reference logits")
        logq = reference.detach().to(device=policy.device, dtype=dtype).log_softmax(-1)
        if kl_direction == "forward":
            kl = (logq.exp() * (logq - logp)).sum(-1)
        else:
            kl = (logp.exp() * (logp - logq)).sum(-1)
        loss = loss + kl_weight * kl
    return loss.sum()


def build_shift_labels(labels, packed_seq_lengths=None):
    if labels.ndim != 2 or not labels.shape[0] or not labels.shape[1]:
        raise ValueError("ASFT expects nonempty [batch, sequence] labels")
    shifted = torch.full_like(labels, -100)
    shifted[:, :-1] = labels[:, 1:]
    if packed_seq_lengths is not None:
        raise NotImplementedError("Packed ASFT requires validated boundary and denominator handling")
    return shifted


@contextmanager
def reference_context(model, disable_adapter=False):
    # Preserve heterogeneous training modes, not just the top-level flag.
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad(), (model.disable_adapter() if disable_adapter else nullcontext()):
            yield
    finally:
        for module, training in modes:
            module.training = training


def get_reference_forward_callable(model, reference_policy="disable_adapter", original_model=None, return_outputs=False):
    if reference_policy == "disable_adapter" and hasattr(model, "disable_adapter"):
        reference, disable = model, True
    elif original_model is not None and reference_policy in ("disable_adapter", "frozen_copy"):
        reference, disable = original_model, False
    else:
        raise ValueError("Provide an adapter-disabled or fixed reference model")

    def forward(**inputs):
        with reference_context(reference, disable):
            outputs = reference(**inputs)
        return outputs if return_outputs else outputs.logits

    return forward


def _check_logits(logits, labels):
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[:2] != labels.shape:
        raise RuntimeError("ASFT requires full [batch, sequence, vocab] logits; disable fused LM-head loss")


def _slice_batch(inputs, start, stop, batch_size):
    return {
        key: value[start:stop] if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch_size else value
        for key, value in inputs.items()
    }


def compute_asft_loss(
    model, inputs, *, asft_mode="asft", kl_weight=0.03,
    reference_policy="disable_adapter", streaming_config=None, original_model=None,
    return_outputs=False, kl_direction="forward", num_items_in_batch=None,
    reference_model=None,
):
    """Causal token-sum objective divided by the optimizer-window token count.

    Only text, unpacked causal LM batches are supported. The reference is evaluated
    before the policy so toggling adapters cannot disturb a live policy graph.
    """
    validate_objective(asft_mode, kl_weight, kl_direction)
    cfg = streaming_config or ASFTStreamingConfig()
    cfg.validate()
    labels = inputs["labels"]
    shifted = build_shift_labels(labels, inputs.get("packed_seq_lengths"))
    mask = shifted.ne(-100)
    if "attention_mask" in inputs:
        attention = inputs["attention_mask"]
        if attention.ndim != 2 or attention.shape != labels.shape:
            raise ValueError("ASFT requires a 2D padding mask")
        if ((labels.ne(-100)) & attention.eq(0)).any():
            raise ValueError("Padding labels must be masked to -100")
        if (mask[:, :-1] & attention[:, :-1].eq(0)).any():
            raise ValueError("Targets after a padding boundary must be masked")
    positions = inputs.get("position_ids")
    if positions is not None and positions.ndim == 2 and (mask[:, :-1] & positions[:, 1:].eq(0)).any():
        raise ValueError("Packed boundaries must be masked; packing is not yet supported")
    if num_items_in_batch is None:
        num_items_in_batch = inputs.get("num_items_in_batch")
    if num_items_in_batch is not None and labels[:, 0].ne(-100).any():
        raise ValueError("Mask first label to -100 for exact accumulated token normalization")
    forward_inputs = {k: v for k, v in inputs.items() if k not in ("labels", "num_items_in_batch", "shift_labels")}
    forward_inputs.update(use_cache=False, return_dict=True)
    batch_size = labels.shape[0]
    reference_rows = []
    if needs_reference(asft_mode, kl_weight) and mask.any():
        base = reference_model if reference_model is not None else model
        ref_forward = get_reference_forward_callable(base, reference_policy, original_model)
        micro = (cfg.ref_microbatch_size or max(1, batch_size // 2)) if cfg.enabled and cfg.ref_strategy == "batch_micro" else batch_size
        for start in range(0, batch_size, micro):
            stop = min(batch_size, start + micro)
            ref = ref_forward(**_slice_batch(forward_inputs, start, stop, batch_size))
            _check_logits(ref, labels[start:stop])
            reference_rows.extend(ref[i, mask[start + i].to(ref.device)].detach() for i in range(stop - start))
            del ref
    outputs = model(**forward_inputs)
    logits = outputs.logits
    _check_logits(logits, labels)
    mask, shifted = mask.to(logits.device), shifted.to(logits.device)
    denominator = mask.sum() if num_items_in_batch is None else torch.as_tensor(num_items_in_batch, device=logits.device)
    total = logits.reshape(-1)[:0].sum(dtype=torch.float32 if logits.dtype != torch.float64 else torch.float64)
    for i in range(batch_size):
        indices = mask[i].nonzero(as_tuple=True)[0]
        chunk = cfg.kl_token_chunk_size or max(1, indices.numel())
        for start in range(0, indices.numel(), chunk):
            ix = indices[start:start + chunk]
            targets = shifted[i, ix]
            ref = reference_rows[i][start:start + chunk] if reference_rows else None
            def loss_chunk(row, index, target, anchor):
                return token_loss(row[index], target, anchor, mode=asft_mode, kl_weight=kl_weight, kl_direction=kl_direction)
            if torch.is_grad_enabled() and logits.requires_grad and cfg.kl_token_chunk_size is not None:
                total = total + checkpoint(loss_chunk, logits[i], ix, targets, ref, use_reentrant=False)
            else:
                total = total + loss_chunk(logits[i], ix, targets, ref)
    loss = total / denominator.clamp_min(1)
    return (loss, outputs) if return_outputs else loss
