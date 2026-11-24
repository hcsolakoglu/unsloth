# SPDX-License-Identifier: Apache-2.0
"""ASFT adapter for Unsloth's current PyTorch trainer."""
from contextlib import contextmanager
from copy import deepcopy
import os

from .losses.asft import ASFTStreamingConfig, compute_asft_loss, needs_reference, validate_objective


@contextmanager
def _return_logits():
    previous = os.environ.get("UNSLOTH_RETURN_LOGITS")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
        else:
            os.environ["UNSLOTH_RETURN_LOGITS"] = previous


class _CausalLabelsCollator:
    def __init__(self, collator):
        self.collator = collator

    def __getattr__(self, name):
        return getattr(self.collator, name)

    def __call__(self, examples):
        batch = dict(self.collator(examples))
        labels = batch["labels"].clone()
        if labels.ndim != 2 or not labels.shape[1]:
            raise ValueError("ASFT requires nonempty, unpacked causal-LM labels")
        labels[:, 0] = -100
        attention = batch.get("attention_mask")
        if attention is not None:
            if attention.ndim != 2 or attention.shape != labels.shape:
                raise ValueError("ASFT requires a two-dimensional padding mask")
            labels.masked_fill_(attention.eq(0), -100)
            labels[:, 1:].masked_fill_(attention[:, :-1].eq(0), -100)
        batch["labels"] = labels
        return batch


class ASFTTrainerMixin:
    """Opt-in; the disabled path delegates unchanged to UnslothTrainer."""

    def __init__(
        self, *args, asft_enabled=False, asft_mode="asft", kl_weight=0.03,
        reference_policy="disable_adapter", asft_streaming=None,
        kl_direction="forward", ref_model=None, **kwargs,
    ):
        self.asft_enabled = asft_enabled
        if not asft_enabled:
            super().__init__(*args, **kwargs)
            return
        validate_objective(asft_mode, kl_weight, kl_direction)
        cfg = asft_streaming or ASFTStreamingConfig()
        cfg.validate()
        if reference_policy not in ("disable_adapter", "frozen_copy"):
            raise ValueError("Unknown ASFT reference policy")
        config = args[1] if len(args) >= 2 else kwargs.get("args")
        if config is None:
            from trl import SFTConfig
            config = SFTConfig()
            if len(args) >= 2:
                args = (args[0], config, *args[2:])
            else:
                kwargs["args"] = config
        for name in ("packing", "padding_free", "eval_packing", "use_liger_kernel", "deepspeed", "fsdp", "label_smoothing_factor"):
            if getattr(config, name, None) or kwargs.get(name):
                raise NotImplementedError(f"ASFT does not yet support {name}")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise NotImplementedError("ASFT distributed normalization is not yet validated")
        for name in ("context_parallel_size", "tensor_parallel_size", "pipeline_parallel_size"):
            if (getattr(config, name, 1) or 1) != 1:
                raise NotImplementedError(f"ASFT does not yet support {name} > 1")
        if kwargs.get("compute_loss_func") is not None:
            raise ValueError("ASFT owns the loss; do not supply compute_loss_func")
        # TRL >= 1.15 can bypass the model forward with chunked_nll. Our loss needs
        # actual logits, so choose NLL's normal forward without using its loss.
        if hasattr(config, "loss_type"):
            config.loss_type = "nll"
        if hasattr(config, "packing"):
            config.packing = False
        if hasattr(config, "padding_free"):
            config.padding_free = False
        with _return_logits():
            super().__init__(*args, **kwargs)
        model_config = getattr(self.model, "config", None)
        if getattr(model_config, "is_encoder_decoder", False) or getattr(model_config, "vision_config", None) is not None:
            raise NotImplementedError("ASFT currently supports text-only causal language models")
        if getattr(self.args, "packing", False) or getattr(self.args, "padding_free", False):
            raise RuntimeError("ASFT initialization unexpectedly enabled packing")
        self.asft_mode = asft_mode
        self.kl_weight = kl_weight
        self.kl_direction = kl_direction
        self.reference_policy = reference_policy
        self.asft_streaming = cfg
        self._asft_original_model = None
        self.data_collator = _CausalLabelsCollator(self.data_collator)
        # HF Trainer passes the full accumulation-window token denominator and
        # does not divide the already-normalized loss by accumulation steps again.
        self.model_accepts_loss_kwargs = True
        if needs_reference(asft_mode, kl_weight):
            if ref_model is not None:
                self.reference_policy = "frozen_copy"
                if ref_model is self.model:
                    raise ValueError("The reference model must not be the policy")
                self._asft_original_model = ref_model
            elif reference_policy == "disable_adapter" and hasattr(self.model, "disable_adapter"):
                trainable = [name for name, param in self.model.named_parameters() if param.requires_grad]
                if not trainable or any("lora_" not in name for name in trainable):
                    raise ValueError("Adapter-disabled anchoring requires frozen base weights and LoRA-only trainable parameters; supply ref_model otherwise")
            else:
                self.reference_policy = "frozen_copy"
                # Snapshot at initialization, never on the first training step.
                self._asft_original_model = deepcopy(self.model)
            if self._asft_original_model is not None:
                trainable_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
                if any(id(p) in trainable_ids for p in self._asft_original_model.parameters()):
                    raise ValueError("Reference shares trainable policy parameters")
                self._asft_original_model.requires_grad_(False)
                self._asft_original_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        if not self.asft_enabled:
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        if kwargs:
            raise TypeError(f"Unexpected ASFT loss arguments: {sorted(kwargs)}")
        if model.training and self.args.gradient_accumulation_steps > 1 and num_items_in_batch is None:
            raise RuntimeError("ASFT requires num_items_in_batch for gradient accumulation")
        with _return_logits():
            return compute_asft_loss(
                model, inputs, asft_mode=self.asft_mode, kl_weight=self.kl_weight,
                kl_direction=self.kl_direction, reference_policy=self.reference_policy,
                original_model=self._asft_original_model, streaming_config=self.asft_streaming,
                return_outputs=return_outputs, num_items_in_batch=num_items_in_batch,
                reference_model=self.accelerator.unwrap_model(model),
            )
