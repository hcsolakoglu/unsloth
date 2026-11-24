# SPDX-License-Identifier: Apache-2.0
"""Opt-in real Unsloth/TRL smoke test; requires CUDA and model download."""
import os
import pytest


@pytest.mark.skipif(os.environ.get("RUN_ASFT_GPU_SMOKE") != "1", reason="opt-in CUDA model test")
def test_real_asft_trainer(tmp_path):
    from unsloth import FastLanguageModel
    from unsloth.trainer import ASFTTrainer, ASFTStreamingConfig
    import torch
    from datasets import Dataset
    from trl import SFTConfig

    assert torch.cuda.is_available()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-0.5B", max_seq_length=128, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=8, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "v_proj"], random_state=42,
    )
    trainable = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    data = Dataset.from_dict({"text": ["A short example.", "A longer example for unequal token counts.", "One more.", "Another test sentence with several tokens."]})
    cfg = SFTConfig(output_dir=str(tmp_path), max_length=128, per_device_train_batch_size=1,
                    gradient_accumulation_steps=2, max_steps=2, packing=False, padding_free=False,
                    learning_rate=5e-4, report_to="none", save_strategy="no")
    trainer = ASFTTrainer(model=model, processing_class=tokenizer, train_dataset=data, args=cfg,
                          asft_enabled=True, asft_streaming=ASFTStreamingConfig(kl_token_chunk_size=32))
    result = trainer.train()
    assert result.global_step == 2
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert any(not torch.equal(p.detach().cpu(), trainable[n]) for n, p in model.named_parameters() if n in trainable)
