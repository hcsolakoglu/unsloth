# SPDX-License-Identifier: Apache-2.0
"""Independent dense oracle; no Unsloth import or GPU needed for the loss tests."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("asft_test_core", Path(__file__).resolve().parents[1] / "unsloth/losses/asft.py")
asft = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = asft
spec.loader.exec_module(asft)


def oracle(logits, ref, labels, mode, weight, direction, denominator=None):
    valid = labels[:, 1:] != -100
    p = logits[:, :-1][valid]
    q = ref[:, :-1][valid].detach()
    y = labels[:, 1:][valid]
    ce = F.cross_entropy(p, y, reduction="none")
    if mode in ("dft", "asft"):
        ce = ce * p.softmax(-1).gather(-1, y[:, None]).squeeze(-1).detach()
    if mode in ("sft+kl", "asft"):
        if direction == "forward":
            kl = F.kl_div(p.log_softmax(-1), q.softmax(-1), reduction="none").sum(-1)
        else:
            kl = F.kl_div(q.log_softmax(-1), p.softmax(-1), reduction="none").sum(-1)
        ce = ce + weight * kl
    return ce.sum() / (denominator if denominator is not None else valid.sum().clamp_min(1))


class Tiny(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.proj = torch.nn.Embedding(11, 11, dtype=torch.float64)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(11, 11, generator=torch.Generator().manual_seed(seed), dtype=torch.float64))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.proj(input_ids))


def batch():
    return {"input_ids": torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 1]]), "labels": torch.tensor([[-100, 2, -100, 4], [-100, -100, 6, 7], [-100, 8, 9, 1]])}


@pytest.mark.parametrize("mode", ["sft", "dft", "sft+kl", "asft"])
@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("micro", [None, 1, 2])
def test_loss_gradient_and_step(mode, direction, micro):
    p, q = Tiny(11), Tiny(17)
    r = Tiny(11)
    data = batch()
    cfg = asft.ASFTStreamingConfig(enabled=micro is not None, ref_strategy="batch_micro", ref_microbatch_size=micro, kl_token_chunk_size=2)
    got = asft.compute_asft_loss(p, data, asft_mode=mode, kl_weight=0.1, kl_direction=direction, original_model=q, reference_policy="frozen_copy", streaming_config=cfg)
    want = oracle(r(**data).logits, q(**data).logits, data["labels"], mode, 0.1, direction)
    torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)
    got.backward(); want.backward()
    torch.testing.assert_close(p.proj.weight.grad, r.proj.weight.grad, atol=1e-12, rtol=1e-12)
    assert q.proj.weight.grad is None
    for model in (p, r):
        torch.optim.AdamW(model.parameters(), lr=0.001).step()
    torch.testing.assert_close(p.proj.weight, r.proj.weight, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_unequal_accumulation(direction):
    p, q, r = Tiny(1), Tiny(2), Tiny(1)
    data = batch(); count = data["labels"].ne(-100).sum()
    common = dict(kl_direction=direction, original_model=q, reference_policy="frozen_copy", num_items_in_batch=count)
    asft.compute_asft_loss(r, data, **common).backward()
    for start, end in [(0, 1), (1, 3)]:
        asft.compute_asft_loss(p, {k: v[start:end] for k, v in data.items()}, **common).backward()
    torch.testing.assert_close(p.proj.weight.grad, r.proj.weight.grad, atol=1e-12, rtol=1e-12)


def test_zero_kl_and_all_ignored():
    p = Tiny(1); data = batch()
    torch.testing.assert_close(asft.compute_asft_loss(p, data, kl_weight=0), asft.compute_asft_loss(p, data, asft_mode="dft"))
    data["labels"].fill_(-100)
    loss = asft.compute_asft_loss(p, data)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(p.proj.weight.grad) == 0


def test_asymmetric_kl_and_stop_gradient():
    p = torch.tensor([[2., -1., 0.]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[-3., 0., 1.]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0]); a = asft.token_loss(p, y, q, kl_weight=1., kl_direction="forward"); b = asft.token_loss(p, y, q, kl_weight=1., kl_direction="reverse")
    assert not torch.isclose(a, b)
    dft = asft.token_loss(p, y, mode="dft"); dft.backward()
    prob = p.detach().softmax(-1)
    expected = prob.clone(); expected[0, 0] -= 1; expected *= prob[0, 0]
    torch.testing.assert_close(p.grad, expected)
    assert q.grad is None


def test_invalid_settings():
    with pytest.raises(NotImplementedError):
        asft.ASFTStreamingConfig(enabled=True, ref_strategy="seq_kv_cache").validate()
    with pytest.raises(ValueError):
        asft.validate_objective("asft", float("nan"), "forward")
    with pytest.raises(ValueError):
        asft.compute_asft_loss(Tiny(1), batch())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_low_precision(dtype):
    p = torch.tensor([[12., -18., 1., -7.], [-9., 0., 5., -10.]], dtype=dtype, requires_grad=True)
    q = torch.tensor([[2., 5., -8., 0.], [2., 4., -1., -3.]], dtype=dtype)
    y = torch.tensor([0, 1])
    for direction in ("forward", "reverse"):
        p.grad = None
        loss = asft.token_loss(p, y, q, kl_direction=direction)
        fp32 = p.detach().float().requires_grad_()
        expected = asft.token_loss(fp32, y, q.float(), kl_direction=direction)
        loss.backward(); expected.backward()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(p.grad, fp32.grad.to(dtype))
        assert torch.isfinite(p.grad).all()


def test_reference_state_and_unchanged_inputs():
    p, q = Tiny(1), Tiny(2)
    q.train(); q.proj.eval()
    before = q.proj.weight.detach().clone()
    data = batch(); keys = set(data); labels = data['labels'].clone()
    for _ in range(2):
        loss = asft.compute_asft_loss(p, data, original_model=q, reference_policy="frozen_copy")
        loss.backward(); torch.optim.AdamW(p.parameters(), lr=.01).step()
    assert q.training and not q.proj.training
    torch.testing.assert_close(q.proj.weight, before)
    assert q.proj.weight.grad is None
    assert set(data) == keys
    torch.testing.assert_close(data['labels'], labels)


def test_reference_adapter_context():
    from contextlib import contextmanager
    class Adapter(Tiny):
        def __init__(self):
            super().__init__(11)
            self.adapter = torch.nn.Parameter(torch.ones(11, dtype=torch.float64) * .2)
            self.active = True
        @contextmanager
        def disable_adapter(self):
            self.active = False
            try:
                yield
            finally:
                self.active = True
        def forward(self, input_ids, **kwargs):
            logits = self.proj(input_ids)
            return SimpleNamespace(logits=logits + self.adapter * torch.arange(11) if self.active else logits)
    p = Adapter(); data = batch()
    reference = p.proj(data['input_ids']).detach()
    got = asft.compute_asft_loss(p, data)
    expected = oracle(p(**data).logits, reference, data['labels'], 'asft', .03, 'forward')
    torch.testing.assert_close(got, expected)
    got.backward(); assert p.active and p.training and p.adapter.grad is not None
