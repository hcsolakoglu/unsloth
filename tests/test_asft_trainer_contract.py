# SPDX-License-Identifier: Apache-2.0
"""Exercise trainer adapter contract without importing Unsloth or downloading a model."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

root = Path(__file__).resolve().parents[1] / "unsloth"
for name, path in (("asft_contract", root), ("asft_contract.losses", root / "losses")):
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
spec = importlib.util.spec_from_file_location("asft_contract.asft_trainer", root / "asft_trainer.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(7, 7)
        self.config = SimpleNamespace()

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.embed(input_ids))


class Parent:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.data_collator = lambda examples: examples[0]
        self.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
        self.init_logits_flag = os.environ.get("UNSLOTH_RETURN_LOGITS")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        return "parent", kwargs


class Trainer(module.ASFTTrainerMixin, Parent):
    pass


def config(**kwargs):
    return SimpleNamespace(**dict(dict(loss_type="chunked_nll", packing=False, padding_free=False, gradient_accumulation_steps=2), **kwargs))


def test_enabled_path_reference_and_window_count(monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising=False)
    model = Toy(); trainer = Trainer(model, config(), asft_enabled=True)
    assert trainer.init_logits_flag == "1" and "UNSLOTH_RETURN_LOGITS" not in os.environ
    assert trainer.args.loss_type == "nll" and trainer.model_accepts_loss_kwargs
    raw = {"input_ids": torch.tensor([[0, 2, 3], [1, 4, 0]]), "labels": torch.tensor([[0, 2, 3], [1, 4, 0]]), "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 0]])}
    batch = trainer.data_collator([raw]); assert batch["labels"].tolist() == [[-100, -100, 3], [-100, 4, -100]]
    assert raw["labels"].tolist() == [[0, 2, 3], [1, 4, 0]]
    ref = trainer._asft_original_model
    before = ref.embed.weight.detach().clone()
    for _ in range(2):
        loss = trainer.compute_loss(model, batch, num_items_in_batch=2)
        loss.backward(); torch.optim.AdamW(model.parameters(), lr=.01).step()
    torch.testing.assert_close(ref.embed.weight, before)
    assert ref.embed.weight.grad is None and not ref.training
    assert "UNSLOTH_RETURN_LOGITS" not in os.environ
    with pytest.raises(RuntimeError, match="num_items_in_batch"):
        trainer.compute_loss(model, batch)


def test_disabled_and_unsupported():
    trainer = Trainer(Toy(), config())
    assert trainer.args.loss_type == "chunked_nll"
    assert trainer.compute_loss(trainer.model, {}, num_items_in_batch=7) == ("parent", {"num_items_in_batch": 7})
    with pytest.raises(NotImplementedError, match="packing"):
        Trainer(Toy(), config(packing=True), asft_enabled=True)
