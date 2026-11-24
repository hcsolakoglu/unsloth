# ASFT+ port (2026-11-24)

## Scope and provenance

Base: Unsloth `1fe75adde2e3c83ed7a810274144947563389f24`.
Previous fork: `hcsolakoglu/unsloth`, branch `asft-plus`. The new implementation is a native rewrite of the loss path, not a transplant of donor code. Reference specifications and comparisons:

- [ASFT v3 (2026-02-01), sections 4.2 and 5.3](https://arxiv.org/html/2509.23753v3): forward KL, unlike v2's reverse-KL description.
- [Official main implementation](https://github.com/zhuchichi56/ASFT/blob/main/train_v2.py) (blob `871efe0ed9b248db0a96b1eeb7053ba31d61db9c`): forward KL.
- [Official verl implementation](https://github.com/zhuchichi56/ASFT/blob/verl/verl/verl/trainer/fsdp_sft_trainer.py) (blob `3a69b19ab8cfe7ec06f32cfb27f1bba35d643735`): forward KL, explicit FP32 KL.
- [LLaMA-Factory #10174](https://github.com/hiyouga/LlamaFactory/pull/10174) and current trainer utils (blob `a0e898e399e7d77d4dbe66f4327e9e04c390f4c2`): forward KL.

The objective on each supervised next-token position is `stopgrad(p[y]) * (-log p[y]) + lambda * KL(q || p)`, where p is policy and q is the fixed reference. Reverse KL is an explicit ablation, not the v3 default. The sum is divided by the supervised token count across the optimizer accumulation window. The model's returned logits are used directly; do not apply softcapping or logit scaling again. The detached DFT factor is not a differentiable multiplier. FP16/BF16 inputs use FP32 arithmetic; FP64 remains FP64. The loss recomputes token chunks during backward using non-reentrant checkpointing. No claimed throughput or VRAM improvement until measured.

## Usage

```python
from unsloth.trainer import ASFTTrainer, ASFTStreamingConfig
from trl import SFTConfig

# model, tokenizer and dataset are prepared as usual with Unsloth.
trainer = ASFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(output_dir="out", packing=False, padding_free=False),
    asft_enabled=True,
    asft_mode="asft",
    kl_weight=0.03,
    kl_direction="forward",
    reference_policy="disable_adapter",
    asft_streaming=ASFTStreamingConfig(
        enabled=True, ref_strategy="batch_micro", ref_microbatch_size=1,
        kl_token_chunk_size=256,
    ),
)
trainer.train()
```

`asft_enabled=False` retains upstream SFT. `asft_mode` supports `sft`, `dft`, `sft+kl`, `asft`. Setting `kl_weight=0` explicitly skips the reference. `disable_adapter` requires only LoRA parameters trainable; use a separate `ref_model` for trainable base parameters or nonstandard adapter setups. For full fine-tuning, `frozen_copy` snapshots at trainer initialization; on resume from a trained checkpoint pass the **original** frozen reference model, not that checkpoint. The explicit reference must already reside on the appropriate device. Quantized adapter-disabled anchoring uses the quantized base, not the original BF16 distribution.

The current port targets single-process, unpacked, text-only PyTorch causal LM training. Packed/padding-free, sequence KV-cache streaming, distributed/FSDP/DeepSpeed, MLX, VLM, and label smoothing are deliberately rejected where requested, rather than silently using an unvalidated path. The full-logits path is not a fused LM-head kernel. The old branch is untouched. No GPU training or paper benchmark has been verified here.

## Verification and reproduction

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_asft.py tests/test_asft_trainer_contract.py
RUN_ASFT_GPU_SMOKE=1 python -m pytest -q tests/test_asft_gpu_smoke.py
```

The CPU tests use an independent dense PyTorch oracle (CE and `kl_div`), nonidentical policy/reference logits, and compare losses, gradients and AdamW updates across directions, modes and microbatch sizes. Additional tests cover mixed-precision input, reference invariance, masking, zero coefficient, no valid targets, accumulation denominator and disabled-path behavior. The trainer contract uses a stub parent; the opt-in test is the real CUDA integration test and is not a substitute for paper reproduction.

For the paper experiment, download the exact datasets from [chichi56/ASFT](https://huggingface.co/datasets/chichi56/ASFT) and use the official evaluation scripts. Preserve the original checkpoint, data subset/order, response-only labels, prompt template, global batch, optimizer/schedule, epochs, seeds, and generation settings. Start with LLaMA-2-7B medical 10k (512 tokens, global batch 64, 3 epochs, 2e-5, cosine, warmup 0.1, lambda 0.05 for paper reproduction); then compare the authors' current mixed-precision recommendation of 0.03 separately. Qwen2.5-7B math 10k uses 2048 tokens, global batch 256, one epoch, 5e-5, cosine, warmup 0.1. Do not compare QLoRA to full fine-tuning as if identical. First reproduce official forward-KL training, then compare port with equal inputs and trainable parameters, then reverse KL and chunk-size/performance ablations. Use at least three seeds for final quality comparison and report actual accuracy, peak allocated/reserved VRAM, throughput, GPU-hours and confidence intervals.
