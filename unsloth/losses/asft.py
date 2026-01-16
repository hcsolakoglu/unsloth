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

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from ..kernels.cross_entropy_loss import Fast_CrossEntropyLoss
from ..utils.packing import mask_packed_sequence_boundaries


@dataclass
class ASFTStreamingConfig:
    strategy: Optional[str] = None
    micro_batch_size: Optional[int] = None
    seq_chunk_size: Optional[int] = None


def effective_logits(
    logits: torch.Tensor,
    *,
    logit_softcapping: float = 0.0,
    logit_scaling: float = 0.0,
) -> torch.Tensor:
    if logit_scaling != 0.0:
        logits = logit_scaling * logits
    if logit_softcapping != 0.0:
        logits = (1.0 / logit_softcapping) * logits
        logits = torch.tanh(logits)
        logits = logit_softcapping * logits
    return logits


def fast_cross_entropy_loss_per_token(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    logit_softcapping: float = 0.0,
    logit_scaling: float = 0.0,
) -> torch.Tensor:
    batch, seq_len, vocab = logits.shape
    flat_logits = logits.reshape(batch * seq_len, vocab)
    flat_labels = labels.reshape(-1)
    if logits.is_cuda:
        losses = Fast_CrossEntropyLoss.apply(
            flat_logits,
            flat_labels,
            logit_softcapping,
            logit_scaling,
        )
    else:
        adjusted = effective_logits(
            logits,
            logit_softcapping = logit_softcapping,
            logit_scaling = logit_scaling,
        )
        losses = F.cross_entropy(
            adjusted.reshape(batch * seq_len, vocab),
            flat_labels,
            reduction = "none",
            ignore_index = -100,
        )
    return losses.view(batch, seq_len)


def build_shift_labels(
    labels: torch.Tensor,
    packed_seq_lengths: Optional[Any] = None,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    shift_labels = torch.empty_like(labels)
    shift_labels[..., :-1] = labels[..., 1:]
    shift_labels[..., -1] = ignore_index
    mask_packed_sequence_boundaries(
        shift_labels,
        packed_seq_lengths,
        ignore_index = ignore_index,
    )
    return shift_labels


def _get_logit_factors(model: Any) -> tuple[float, float]:
    config = getattr(model, "config", SimpleNamespace())
    logit_softcapping = float(getattr(config, "final_logit_softcapping", 0) or 0)
    logit_scaling = float(getattr(config, "logit_scale", 0) or 0)
    model_type = getattr(config, "model_type", None)
    if model_type == "granite":
        scaling_val = float(getattr(config, "logits_scaling", 1) or 1)
        logit_scaling = 1.0 / scaling_val if scaling_val != 0.0 else 0.0
    elif model_type == "falcon_h1":
        logit_scaling = float(getattr(config, "lm_head_multiplier", 0) or 0)
    return logit_softcapping, logit_scaling


def _extract_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    raise ValueError("ASFT expects model outputs to contain logits.")


def _build_model_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    model_inputs = dict(inputs)
    model_inputs["labels"] = None
    return model_inputs


def _microbatch_inputs(
    model_inputs: dict[str, Any],
    start: int,
    end: int,
) -> dict[str, Any]:
    sliced = {}
    for key, value in model_inputs.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            if value.shape[0] == model_inputs["input_ids"].shape[0]:
                sliced[key] = value[start:end]
                continue
        sliced[key] = value
    return sliced


def _reference_forward_batch_micro(
    reference_model: Any,
    model_inputs: dict[str, Any],
    micro_batch_size: int,
) -> torch.Tensor:
    outputs = []
    batch = model_inputs["input_ids"].shape[0]
    for start in range(0, batch, micro_batch_size):
        end = min(start + micro_batch_size, batch)
        micro_inputs = _microbatch_inputs(model_inputs, start, end)
        out = reference_model(**micro_inputs)
        outputs.append(_extract_logits(out))
    return torch.cat(outputs, dim = 0)


def _reference_forward_seq_kv_cache(
    reference_model: Any,
    model_inputs: dict[str, Any],
    seq_chunk_size: int,
) -> torch.Tensor:
    input_ids = model_inputs["input_ids"]
    seq_len = input_ids.shape[1]
    logits_chunks = []
    past_key_values = None
    for start in range(0, seq_len, seq_chunk_size):
        end = min(start + seq_chunk_size, seq_len)
        chunk_inputs = dict(model_inputs)
        chunk_inputs["input_ids"] = input_ids[:, start:end]
        if "attention_mask" in chunk_inputs:
            chunk_inputs["attention_mask"] = chunk_inputs["attention_mask"][:, :end]
        if "position_ids" in chunk_inputs:
            chunk_inputs["position_ids"] = chunk_inputs["position_ids"][:, start:end]
        chunk_inputs["past_key_values"] = past_key_values
        chunk_inputs["use_cache"] = True
        out = reference_model(**chunk_inputs)
        logits_chunks.append(_extract_logits(out))
        past_key_values = getattr(out, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError("ASFT seq_kv_cache requires past_key_values support.")
    return torch.cat(logits_chunks, dim = 1)


def get_reference_forward_callable(
    model: Any,
    reference_model: Optional[Any] = None,
    streaming_config: Optional[ASFTStreamingConfig] = None,
) -> Callable[[dict[str, Any]], torch.Tensor]:
    ref_model = (
        reference_model
        or getattr(model, "reference_model", None)
        or getattr(model, "ref_model", None)
        or model
    )
    strategy = streaming_config.strategy if streaming_config else None
    if strategy == "batch_micro":
        micro_batch_size = streaming_config.micro_batch_size or 1

        def forward(model_inputs: dict[str, Any]) -> torch.Tensor:
            return _reference_forward_batch_micro(
                ref_model,
                model_inputs,
                micro_batch_size,
            )

        return forward
    if strategy == "seq_kv_cache":
        seq_chunk_size = streaming_config.seq_chunk_size or 1

        def forward(model_inputs: dict[str, Any]) -> torch.Tensor:
            return _reference_forward_seq_kv_cache(
                ref_model,
                model_inputs,
                seq_chunk_size,
            )

        return forward

    def forward(model_inputs: dict[str, Any]) -> torch.Tensor:
        return _extract_logits(ref_model(**model_inputs))

    return forward


def compute_asft_loss(
    model: Any,
    inputs: dict[str, Any],
    *,
    reference_model: Optional[Any] = None,
    streaming_config: Optional[ASFTStreamingConfig] = None,
    sft_weight: float = 1.0,
    dft_weight: float = 1.0,
    dft_beta: float = 1.0,
    kl_weight: float = 1.0,
    return_outputs: bool = False,
    return_details: bool = False,
    num_items_in_batch: Optional[int] = None,
) -> Any:
    labels = inputs.get("labels")
    if labels is None:
        raise ValueError("ASFT requires labels in inputs.")
    packed_seq_lengths = inputs.get("packed_seq_lengths")
    shift_labels = build_shift_labels(labels, packed_seq_lengths)
    valid_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.clone()
    safe_labels[~valid_mask] = 0

    model_inputs = _build_model_inputs(inputs)
    outputs = model(**model_inputs)
    logits = _extract_logits(outputs)

    logit_softcapping, logit_scaling = _get_logit_factors(model)
    sft_loss_per_token = fast_cross_entropy_loss_per_token(
        logits,
        shift_labels,
        logit_softcapping = logit_softcapping,
        logit_scaling = logit_scaling,
    )

    ref_forward = get_reference_forward_callable(
        model,
        reference_model = reference_model,
        streaming_config = streaming_config,
    )
    with torch.inference_mode():
        ref_logits = ref_forward(model_inputs)

    ref_softcapping, ref_scaling = _get_logit_factors(reference_model or model)
    cur_logits = effective_logits(
        logits,
        logit_softcapping = logit_softcapping,
        logit_scaling = logit_scaling,
    )
    ref_logits = effective_logits(
        ref_logits,
        logit_softcapping = ref_softcapping,
        logit_scaling = ref_scaling,
    )
    cur_log_probs = torch.log_softmax(cur_logits, dim = -1)
    ref_log_probs = torch.log_softmax(ref_logits, dim = -1)
    label_logp_cur = cur_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    label_logp_ref = ref_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    dft_weights = torch.exp(dft_beta * (label_logp_ref - label_logp_cur)).detach()
    dft_loss_per_token = sft_loss_per_token * dft_weights
    ref_probs = ref_log_probs.exp()
    kl_per_token = torch.sum(ref_probs * (ref_log_probs - cur_log_probs), dim = -1)

    sft_loss_per_token = sft_loss_per_token * valid_mask
    dft_loss_per_token = dft_loss_per_token * valid_mask
    kl_per_token = kl_per_token * valid_mask
    loss_per_token = (
        sft_weight * sft_loss_per_token
        + dft_weight * dft_loss_per_token
        + kl_weight * kl_per_token
    )
    denom = (
        num_items_in_batch
        or inputs.get("num_items_in_batch")
        or inputs.get("n_items")
        or valid_mask.sum()
    )
    if isinstance(denom, torch.Tensor):
        denom = int(denom.item())
    else:
        denom = int(denom)
    denom = denom if denom > 0 else 1
    loss = loss_per_token.sum() / denom

    if return_details:
        details = {
            "sft_loss": sft_loss_per_token.sum() / denom,
            "dft_loss": dft_loss_per_token.sum() / denom,
            "kl_loss": kl_per_token.sum() / denom,
            "dft_weights": dft_weights,
            "valid_mask": valid_mask,
        }
    else:
        details = None

    if return_outputs:
        if return_details:
            return loss, outputs, details
        return loss, outputs
    if return_details:
        return loss, details
    return loss
