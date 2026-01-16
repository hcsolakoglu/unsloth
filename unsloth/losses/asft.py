# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import contextlib
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from unsloth.kernels.cross_entropy_loss import Fast_CrossEntropyLoss
from unsloth.utils.packing import mask_packed_sequence_boundaries

IGNORE_INDEX = -100


@dataclass
class ASFTStreamingConfig:
    enabled: bool = False
    ref_strategy: Literal["none", "batch_micro", "seq_kv_cache"] = "none"
    ref_microbatch_size: Optional[int] = None
    seq_chunk_size: Optional[int] = None
    kl_token_chunk_size: Optional[int] = None
    force_fp32_kl: bool = True


def _logit_transforms_from_config(model) -> Tuple[float, float]:
    config = getattr(model, "config", None)
    logit_softcapping = 0.0
    logit_scaling = 0.0
    if config is not None:
        logit_softcapping = float(getattr(config, "final_logit_softcapping", 0) or 0)
        logit_scaling = float(getattr(config, "logit_scale", 0) or 0)
        model_type = getattr(config, "model_type", None)
        if model_type == "granite":
            logit_scaling = 1 / float(getattr(config, "logits_scaling", 1) or 1)
        elif model_type == "falcon_h1":
            logit_scaling = float(getattr(config, "lm_head_multiplier", logit_scaling))
        elif hasattr(config, "lm_head_multiplier") and logit_scaling == 0.0:
            logit_scaling = float(getattr(config, "lm_head_multiplier"))
    return logit_softcapping, logit_scaling


def effective_logits(logits: torch.Tensor, model) -> torch.Tensor:
    logit_softcapping, logit_scaling = _logit_transforms_from_config(model)
    logits_eff = logits.float()
    if logit_scaling != 0:
        logits_eff = logits_eff * logit_scaling
    if logit_softcapping != 0:
        logits_eff = logit_softcapping * torch.tanh(logits_eff / logit_softcapping)
    return logits_eff


def fast_cross_entropy_loss_per_token(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logit_softcapping: float,
    logit_scaling: float,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, vocab = logits.shape
    flat_logits = logits.view(batch * seq_len, vocab)
    flat_labels = labels.view(-1)
    safe_labels = torch.clamp(flat_labels, 0, vocab - 1)

    if flat_logits.is_cuda:
        ce_flat = Fast_CrossEntropyLoss.apply(
            flat_logits, safe_labels, logit_softcapping, logit_scaling
        )
    else:
        ce_flat = F.cross_entropy(
            flat_logits, safe_labels, reduction="none", ignore_index=ignore_index
        )

    valid_mask = flat_labels != ignore_index
    ce = ce_flat.view(batch, seq_len)
    ce = ce.masked_fill(~valid_mask.view(batch, seq_len), 0)
    return ce, valid_mask.view(batch, seq_len)


def build_shift_labels(labels: torch.Tensor) -> torch.Tensor:
    shift_labels = torch.empty_like(labels)
    shift_labels[..., :-1] = labels[..., 1:]
    shift_labels[..., -1] = IGNORE_INDEX
    return shift_labels


def _clone_reference(model):
    reference = copy.deepcopy(model)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    return reference


def get_reference_forward_callable(
    model,
    reference_policy: Literal["disable_adapter", "frozen_copy"] = "disable_adapter",
    original_model=None,
) -> Callable[[Dict], torch.Tensor]:
    if reference_policy == "disable_adapter" and hasattr(model, "disable_adapter"):
        def ref_forward(**kwargs):
            was_training = model.training
            with torch.inference_mode(), model.disable_adapter():
                model_was_training = model.training
                if model_was_training:
                    model.eval()
                outputs = model(**kwargs)
            if was_training:
                model.train()
            return outputs

        return ref_forward

    reference_model = _clone_reference(original_model or model)

    def ref_forward(**kwargs):
        with torch.inference_mode():
            return reference_model(**kwargs)

    return ref_forward


def _extract_logits(outputs) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise ValueError("Model outputs do not contain logits.")


def _compute_token_kl(
    cur_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    model,
    *,
    force_fp32: bool,
    kl_token_chunk_size: Optional[int],
) -> torch.Tensor:
    cur_eff = effective_logits(cur_logits, model)
    ref_eff = effective_logits(ref_logits, model)
    if force_fp32:
        cur_eff = cur_eff.float()
        ref_eff = ref_eff.float()

    cur_logp = F.log_softmax(cur_eff, dim=-1)
    ref_p = torch.softmax(ref_eff, dim=-1)
    kl = F.kl_div(cur_logp, ref_p, reduction="none").sum(-1)
    kl = kl.masked_fill(~valid_mask, 0)

    if kl_token_chunk_size and kl_token_chunk_size > 0:
        flat = kl.view(-1)
        mask_flat = valid_mask.view(-1)
        out = torch.zeros_like(flat)
        valid_indices = torch.nonzero(mask_flat, as_tuple=False).flatten()
        for start in range(0, valid_indices.numel(), kl_token_chunk_size):
            idx = valid_indices[start : start + kl_token_chunk_size]
            out[idx] = flat[idx]
        kl = out.view_as(kl)
    return kl


def _slice_batch_inputs(batch_inputs: Dict, start: int, end: int) -> Dict:
    sliced: Dict = {}
    batch_dim = None
    for value in batch_inputs.values():
        if torch.is_tensor(value) and value.dim() > 0:
            batch_dim = value.shape[0]
            break
    for key, value in batch_inputs.items():
        if torch.is_tensor(value) and value.dim() > 0 and batch_dim is not None and value.shape[0] == batch_dim:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def _compute_seq_chunk_reference_logits(
    batch_inputs: Dict,
    ref_forward: Callable,
    seq_chunk_size: Optional[int],
) -> Optional[torch.Tensor]:
    if seq_chunk_size is None or seq_chunk_size <= 0:
        return None
    input_ids = batch_inputs.get("input_ids")
    if input_ids is None:
        return None
    attention_mask = batch_inputs.get("attention_mask")
    position_ids = batch_inputs.get("position_ids")
    past_key_values = None
    ref_logits_parts = []
    try:
        total_seq = input_ids.size(1)
        for start in range(0, total_seq, seq_chunk_size):
            end = min(total_seq, start + seq_chunk_size)
            chunk_inputs = dict(batch_inputs)
            chunk_inputs["input_ids"] = input_ids[:, start:end]
            if attention_mask is not None:
                chunk_inputs["attention_mask"] = attention_mask[:, :end]
            if position_ids is not None:
                chunk_inputs["position_ids"] = position_ids[:, start:end]
            if past_key_values is not None:
                chunk_inputs["past_key_values"] = past_key_values
            chunk_inputs["use_cache"] = True
            outputs = ref_forward(**chunk_inputs)
            if not hasattr(outputs, "past_key_values"):
                return None
            past_key_values = outputs.past_key_values
            ref_logits_parts.append(_extract_logits(outputs))
        return torch.cat(ref_logits_parts, dim=1)
    except Exception:
        return None


def compute_asft_loss(
    model,
    inputs: Dict,
    *,
    asft_mode: Literal["sft", "dft", "sft+kl", "asft"] = "asft",
    kl_weight: float = 0.0,
    reference_policy: Literal["disable_adapter", "frozen_copy"] = "disable_adapter",
    streaming: Optional[ASFTStreamingConfig] = None,
    return_outputs: bool = False,
):
    streaming = streaming or ASFTStreamingConfig()
    forward_inputs = {
        k: v for k, v in inputs.items() if k not in {"labels", "num_items_in_batch"}
    }
    forward_inputs.pop("labels", None)

    outputs = model(**forward_inputs)
    logits = _extract_logits(outputs)
    labels = inputs["labels"]

    shift_labels = build_shift_labels(labels)
    packed_seq_lengths = inputs.get("packed_seq_lengths", None)
    if packed_seq_lengths is not None:
        mask_packed_sequence_boundaries(shift_labels, packed_seq_lengths)
    valid_mask = shift_labels != IGNORE_INDEX

    n_items = inputs.get("num_items_in_batch", None)
    if n_items is None:
        n_items = valid_mask.sum()
    n_items_tensor = torch.as_tensor(
        n_items, device=logits.device, dtype=logits.dtype
    )

    logit_softcapping, logit_scaling = _logit_transforms_from_config(model)
    ce, valid_mask = fast_cross_entropy_loss_per_token(
        logits, shift_labels, logit_softcapping, logit_scaling, IGNORE_INDEX
    )

    token_loss = ce
    if asft_mode in ("dft", "asft"):
        logits_eff = effective_logits(logits, model)
        probs = torch.softmax(logits_eff, dim=-1)
        safe_labels = torch.clamp(shift_labels, 0, logits.shape[-1] - 1)
        weights = torch.gather(probs, -1, safe_labels.unsqueeze(-1)).squeeze(-1).detach()
        dft = ce * weights
        token_loss = dft

    kl_component = None
    requires_kl = asft_mode in ("sft+kl", "asft")
    if requires_kl and kl_weight != 0:
        ref_forward = get_reference_forward_callable(
            model, reference_policy=reference_policy
        )
        ref_logits = None
        if streaming.enabled and streaming.ref_strategy == "batch_micro":
            batch = logits.shape[0]
            token_kl = torch.zeros_like(valid_mask, dtype=logits.dtype)
            micro = streaming.ref_microbatch_size or 1
            if packed_seq_lengths is not None:
                micro = batch  # fallback to full batch to avoid slicing packed lengths
            for start in range(0, batch, micro):
                end = min(batch, start + micro)
                sliced_inputs = _slice_batch_inputs(inputs, start, end)
                sliced_inputs = {k: v for k, v in sliced_inputs.items() if k != "labels"}
                ref_out = ref_forward(**sliced_inputs)
                ref_logits_mb = _extract_logits(ref_out)
                cur_logits_mb = logits[start:end]
                valid_mask_mb = valid_mask[start:end]
                token_kl[start:end] = _compute_token_kl(
                    cur_logits_mb,
                    ref_logits_mb,
                    valid_mask_mb,
                    model,
                    force_fp32=streaming.force_fp32_kl,
                    kl_token_chunk_size=streaming.kl_token_chunk_size,
                )
                del ref_logits_mb
            kl_component = token_kl
        elif streaming.enabled and streaming.ref_strategy == "seq_kv_cache":
            ref_logits = _compute_seq_chunk_reference_logits(
                {k: v for k, v in inputs.items() if k != "labels"},
                ref_forward,
                streaming.seq_chunk_size,
            )

        if ref_logits is None:
            ref_out = ref_forward(**{k: v for k, v in inputs.items() if k != "labels"})
            ref_logits = _extract_logits(ref_out)

        if kl_component is None:
            kl_component = _compute_token_kl(
                logits,
                ref_logits,
                valid_mask,
                model,
                force_fp32=streaming.force_fp32_kl,
                kl_token_chunk_size=streaming.kl_token_chunk_size,
            )

    if requires_kl and kl_component is not None:
        if asft_mode == "sft+kl":
            token_loss = ce + kl_weight * kl_component
        else:
            token_loss = token_loss + kl_weight * kl_component

    valid_tokens = valid_mask.sum()
    if valid_tokens.item() == 0:
        loss = logits.sum() * 0.0
    else:
        loss = token_loss[valid_mask].sum() / n_items_tensor

    if return_outputs:
        return loss, outputs
    return loss
