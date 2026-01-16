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

from types import SimpleNamespace

import pytest
import torch

from unsloth.losses.asft import (
    ASFTStreamingConfig,
    build_shift_labels,
    compute_asft_loss,
    get_reference_forward_callable,
)


class FixedLogitsModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype = torch.float32))
        self.config = SimpleNamespace(final_logit_softcapping = 0.0, logit_scale = 0.0, model_type = "llama")

    def forward(self, input_ids = None, labels = None, **kwargs):
        logits = self.fixed_logits.to(input_ids.device)
        return SimpleNamespace(logits = logits)


class ToyCacheModel(torch.nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias = False)
        self.config = SimpleNamespace(final_logit_softcapping = 0.0, logit_scale = 0.0, model_type = "llama")

    def forward(
        self,
        input_ids = None,
        labels = None,
        past_key_values = None,
        use_cache = False,
        **kwargs,
    ):
        hidden = self.embedding(input_ids)
        if past_key_values is not None:
            hidden = torch.cat([past_key_values[0], hidden], dim = 1)
        context = hidden.cumsum(dim = 1)
        logits = self.lm_head(context)[:, -input_ids.shape[1] :, :]
        next_past = (hidden.detach(),) if use_cache else None
        return SimpleNamespace(logits = logits, past_key_values = next_past)


def test_asft_math_components():
    cur_logits = [[[1.0, 0.5, -0.2, 0.0], [0.1, 0.2, 0.3, 0.4], [0.0, -0.1, 0.2, 0.1]]]
    ref_logits = [[[0.9, 0.1, -0.3, 0.2], [0.2, 0.1, 0.0, 0.4], [0.1, 0.0, 0.1, 0.2]]]
    model = FixedLogitsModel(cur_logits)
    reference_model = FixedLogitsModel(ref_logits)
    input_ids = torch.zeros((1, 3), dtype = torch.long)
    labels = torch.tensor([[0, 1, 2]])
    inputs = {"input_ids": input_ids, "labels": labels}

    loss, details = compute_asft_loss(
        model,
        inputs,
        reference_model = reference_model,
        sft_weight = 1.0,
        dft_weight = 1.0,
        dft_beta = 1.0,
        kl_weight = 1.0,
        return_details = True,
    )

    shift_labels = build_shift_labels(labels)
    valid_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.clone()
    safe_labels[~valid_mask] = 0
    cur_logits_t = torch.tensor(cur_logits)
    ref_logits_t = torch.tensor(ref_logits)
    sft_per = torch.nn.functional.cross_entropy(
        cur_logits_t.view(-1, cur_logits_t.shape[-1]),
        shift_labels.view(-1),
        reduction = "none",
        ignore_index = -100,
    ).view_as(shift_labels)
    cur_log_probs = torch.log_softmax(cur_logits_t, dim = -1)
    ref_log_probs = torch.log_softmax(ref_logits_t, dim = -1)
    label_logp_cur = cur_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    label_logp_ref = ref_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    dft_weights = torch.exp(label_logp_ref - label_logp_cur)
    dft_per = sft_per * dft_weights
    kl_per = torch.sum(ref_log_probs.exp() * (ref_log_probs - cur_log_probs), dim = -1)
    denom = valid_mask.sum()
    expected_sft = (sft_per * valid_mask).sum() / denom
    expected_dft = (dft_per * valid_mask).sum() / denom
    expected_kl = (kl_per * valid_mask).sum() / denom
    expected_total = expected_sft + expected_dft + expected_kl

    assert torch.allclose(details["sft_loss"], expected_sft)
    assert torch.allclose(details["dft_loss"], expected_dft)
    assert torch.allclose(details["kl_loss"], expected_kl)
    assert torch.allclose(loss, expected_total)


def test_build_shift_labels_masks_boundaries():
    labels = torch.arange(6, dtype = torch.long).view(1, 6)
    shift_labels = build_shift_labels(labels, packed_seq_lengths = [2, 3, 1])
    expected = torch.tensor([[1, -100, 3, 4, -100, -100]])
    assert torch.equal(shift_labels, expected)


def test_streaming_equivalence():
    torch.manual_seed(0)
    model = ToyCacheModel(vocab = 8, hidden = 4)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    labels = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    base_inputs = {"input_ids": input_ids, "labels": None}
    base_logits = model(input_ids = input_ids).logits

    micro_config = ASFTStreamingConfig(strategy = "batch_micro", micro_batch_size = 1)
    micro_forward = get_reference_forward_callable(
        model,
        reference_model = model,
        streaming_config = micro_config,
    )
    micro_logits = micro_forward(base_inputs)
    assert torch.allclose(base_logits, micro_logits)

    seq_config = ASFTStreamingConfig(strategy = "seq_kv_cache", seq_chunk_size = 2)
    seq_forward = get_reference_forward_callable(
        model,
        reference_model = model,
        streaming_config = seq_config,
    )
    seq_logits = seq_forward(base_inputs)
    assert torch.allclose(base_logits, seq_logits)

    full_loss, full_details = compute_asft_loss(
        model,
        {"input_ids": input_ids, "labels": labels},
        reference_model = model,
        return_details = True,
    )
    stream_loss, stream_details = compute_asft_loss(
        model,
        {"input_ids": input_ids, "labels": labels},
        reference_model = model,
        streaming_config = seq_config,
        return_details = True,
    )
    assert torch.allclose(full_loss, stream_loss)
    assert torch.allclose(full_details["sft_loss"], stream_details["sft_loss"])


def test_asft_disabled_uses_super():
    trl = pytest.importorskip("trl")
    from unsloth.trainer import UnslothTrainer

    sentinel = object()

    def fake_compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        return sentinel

    original = trl.SFTTrainer.compute_loss
    trl.SFTTrainer.compute_loss = fake_compute_loss
    try:
        trainer = UnslothTrainer.__new__(UnslothTrainer)
        trainer.args = SimpleNamespace(asft_enabled = False)
        result = UnslothTrainer.compute_loss(trainer, None, {})
        assert result is sentinel
    finally:
        trl.SFTTrainer.compute_loss = original
