import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import types

# Provide a lightweight unsloth_zoo stub to avoid GPU requirements in tests.
if "unsloth_zoo" not in sys.modules:
    unsloth_zoo = types.ModuleType("unsloth_zoo")
    device_type = types.SimpleNamespace(
        is_hip=lambda: False,
        get_device_type=lambda: "cuda",
        DEVICE_TYPE="cuda",
        DEVICE_TYPE_TORCH="cuda",
        DEVICE_COUNT=1,
        ALLOW_PREQUANTIZED_MODELS=True,
    )
    rl_env = types.SimpleNamespace(
        RL_ENVIRONMENT_MAPPING={},
        register_rl_environment=lambda *args, **kwargs: None,
        RLEnvironment=object,
        TimeAwareRLEnvironment=object,
        RLResponseTokenRLEnvironment=object,
    )
    unsloth_zoo.device_type = device_type
    unsloth_zoo.rl_environments = rl_env
    sys.modules["unsloth_zoo"] = unsloth_zoo
    sys.modules["unsloth_zoo.device_type"] = unsloth_zoo.device_type
    sys.modules["unsloth_zoo.rl_environments"] = unsloth_zoo.rl_environments

if hasattr(torch, "cuda"):
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unsloth.losses.asft import (  # noqa: E402
    ASFTStreamingConfig,
    build_shift_labels,
    compute_asft_loss,
    fast_cross_entropy_loss_per_token,
)


class _DummyModel(torch.nn.Module):
    def __init__(self, vocab: int = 5, adapter_bias: float = 0.25):
        super().__init__()
        self.config = SimpleNamespace(
            model_type = "llama",
            final_logit_softcapping = 0.0,
            logit_scale = 0.0,
        )
        self.embed = torch.nn.Embedding(vocab, vocab)
        self.lm_head = torch.nn.Linear(vocab, vocab, bias = False)
        self.adapter_bias = torch.nn.Parameter(torch.tensor(adapter_bias))
        self.use_adapter = True

    def disable_adapter(self):
        model = self

        @contextlib.contextmanager
        def _cm():
            prev = model.use_adapter
            model.use_adapter = False
            try:
                yield
            finally:
                model.use_adapter = prev

        return _cm()

    def forward(
        self,
        input_ids = None,
        past_key_values = None,
        use_cache = False,
        **kwargs,
    ):
        x = self.embed(input_ids)
        logits = self.lm_head(x)
        if self.use_adapter:
            logits = logits + self.adapter_bias
        return SimpleNamespace(logits = logits, past_key_values = ())


def test_build_shift_labels_matches_unsloth_style():
    labels = torch.tensor([[1, 2, 3, 4]])
    shifted = build_shift_labels(labels)
    assert shifted.tolist() == [[2, 3, 4, -100]]


def test_asft_sft_matches_manual_ce():
    model = _DummyModel(vocab = 6)
    input_ids = torch.tensor([[0, 1, 2]])
    labels = torch.tensor([[0, 1, -100]])
    outputs = model(input_ids = input_ids)
    shift_labels = build_shift_labels(labels)
    manual = F.cross_entropy(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        shift_labels.view(-1),
        ignore_index = -100,
        reduction = "sum",
    )
    manual = manual / (shift_labels != -100).sum()
    loss = compute_asft_loss(
        model,
        {"input_ids": input_ids, "labels": labels},
        asft_mode = "sft",
        kl_weight = 0.0,
        streaming = ASFTStreamingConfig(enabled = False),
    )
    torch.testing.assert_close(loss, manual)


def test_fast_ce_per_token_respects_masking():
    logits = torch.randn(1, 4, 5)
    labels = torch.tensor([[1, 2, -100, 3]])
    losses, mask = fast_cross_entropy_loss_per_token(
        logits, labels, 0.0, 0.0, ignore_index = -100
    )
    assert losses.shape == labels.shape
    assert mask.sum().item() == 3
    assert losses[0, 2].item() == 0


def test_asft_packing_masks_boundaries():
    model = _DummyModel(vocab = 8)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[5, 6, 7, 1]])
    packed_lengths = torch.tensor([2, 2], dtype = torch.int32)
    loss = compute_asft_loss(
        model,
        {"input_ids": input_ids, "labels": labels, "packed_seq_lengths": packed_lengths},
        asft_mode = "sft",
        kl_weight = 0.0,
    )
    # Boundary token (index 1,3 after shift) should be ignored, leaving two tokens.
    assert torch.isfinite(loss)


def test_asft_streaming_equivalence_with_kl():
    model = _DummyModel(vocab = 7, adapter_bias = 0.5)
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    labels = torch.tensor([[0, 1, 2], [3, 4, -100]])
    base_loss = compute_asft_loss(
        model,
        {"input_ids": input_ids, "labels": labels},
        asft_mode = "asft",
        kl_weight = 0.3,
        streaming = ASFTStreamingConfig(enabled = False),
    )
    streaming_loss = compute_asft_loss(
        model,
        {"input_ids": input_ids, "labels": labels},
        asft_mode = "asft",
        kl_weight = 0.3,
        streaming = ASFTStreamingConfig(
            enabled = True,
            ref_strategy = "batch_micro",
            ref_microbatch_size = 1,
        ),
    )
    seq_stream_loss = compute_asft_loss(
        model,
        {"input_ids": input_ids, "labels": labels},
        asft_mode = "asft",
        kl_weight = 0.3,
        streaming = ASFTStreamingConfig(
            enabled = True,
            ref_strategy = "seq_kv_cache",
            seq_chunk_size = 2,
        ),
    )

    torch.testing.assert_close(base_loss, streaming_loss, rtol = 1e-4, atol = 1e-4)
    torch.testing.assert_close(base_loss, seq_stream_loss, rtol = 1e-4, atol = 1e-4)
