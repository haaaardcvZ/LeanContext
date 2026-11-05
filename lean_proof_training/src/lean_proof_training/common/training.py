"""Architecture-neutral training loop; losses remain architecture-specific."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, set_seed

from lean_proof_training.common.batching import move_batch
from lean_proof_training.common.checkpointing import append_jsonl, save_checkpoint
from lean_proof_training.common.data import PairedViewDataset, TokenCollator
from lean_proof_training.common.jsonl import IndexedJsonlDataset
from lean_proof_training.common.loss_types import LossOutput
from lean_proof_training.common.metrics import evaluate_loader
from lean_proof_training.common.modeling import load_model, load_tokenizer, resolved_loss_config
from lean_proof_training.common.runtime import build_accelerator


LossFunction = Callable[..., LossOutput]


def resolve_data_paths(config: dict[str, Any]) -> dict[str, Any]:
    data = dict(config["data"])
    policy_path = data.get("policy_file")
    if policy_path:
        policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
        selected = str(policy["selected_budget"])
        if selected == "none":
            data["train_with_file"] = data["train_without_file"]
            data["validation_with_file"] = data["validation_without_file"]
        else:
            for key in ("train_with_file", "validation_with_file"):
                if key in data:
                    data[key] = str(data[key]).format(selected_budget=selected)
        data["selected_budget"] = selected
    return data


def build_datasets(config: dict[str, Any]) -> tuple[Any, Any]:
    data = resolve_data_paths(config)
    mode = str(data.get("mode", "single"))
    stage = data.get("stage")
    if mode == "single":
        train = IndexedJsonlDataset(data["train_file"], stage=stage)
        validation = IndexedJsonlDataset(data["validation_file"])
    elif mode == "paired":
        train = PairedViewDataset(
            data["train_without_file"],
            data["train_with_file"],
            stage=stage,
            seed=int(config.get("seed", 20260721)),
            context_probability=float(data.get("context_probability", 0.5)),
        )
        validation = PairedViewDataset(
            data["validation_without_file"],
            data["validation_with_file"],
            stage=None,
            seed=int(config.get("seed", 20260721)),
            context_probability=0.5,
        )
    else:
        raise ValueError("data.mode must be single or paired")
    expected = data.get("expected_train_samples")
    if expected is not None and len(train) != int(expected):
        raise ValueError(f"expected {expected} train samples, found {len(train)}")
    expected_families = data.get("expected_train_family_counts")
    if expected_families is not None:
        actual = dict(train.proof_family_counts)
        expected_families = {str(key): int(value) for key, value in expected_families.items()}
        if actual != expected_families:
            raise ValueError(
                f"unexpected proof-family distribution: expected {expected_families}, found {actual}"
            )
    return train, validation


def build_optimizer(model: Any, config: dict[str, Any]) -> AdamW:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    training = config["training"]
    return AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
        betas=(float(training.get("adam_beta1", 0.9)), float(training.get("adam_beta2", 0.95))),
        eps=float(training.get("adam_epsilon", 1e-8)),
        fused=bool(training.get("fused_optimizer", True)) and torch.cuda.is_available(),
    )


def run_training(
    config: dict[str, Any],
    *,
    architecture: str,
    loss_function: LossFunction,
) -> dict[str, Any]:
    set_seed(int(config.get("seed", 20260721)))
    accelerator = build_accelerator(config)
    output_dir = Path(config["output_dir"])
    if bool(config["training"].get("fresh_start", True)):
        # Reproduction must never destroy a previous run. All ranks perform
        # the same read-only guard before any directory or checkpoint write.
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"fresh_start refuses a nonempty output directory: {output_dir}. "
                "Choose a new CHECKPOINT_ROOT/output_dir; existing data is preserved."
            )
        accelerator.wait_for_everyone()
    tokenizer = load_tokenizer(config, tokenized=True)
    loss_config = resolved_loss_config(config, architecture=architecture, tokenizer=tokenizer)
    train_dataset, validation_dataset = build_datasets(config)
    collator = TokenCollator(
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        multiple=int(config["training"].get("pad_to_multiple_of", 8)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["per_device_train_batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=True,
        persistent_workers=int(config["training"].get("num_workers", 4)) > 0,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["training"]["per_device_eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=True,
        persistent_workers=int(config["training"].get("num_workers", 4)) > 0,
        collate_fn=collator,
    )
    model = load_model(config, architecture=architecture, tokenizer=tokenizer, trainable=True)
    optimizer = build_optimizer(model, config)
    epochs = int(config["training"]["epochs"])
    per_rank_batches = math.ceil(len(train_loader) / accelerator.num_processes)
    updates_per_epoch = math.ceil(
        per_rank_batches / int(config["training"].get("gradient_accumulation_steps", 1))
    )
    total_updates = max(1, updates_per_epoch * epochs)
    # Accelerate's scheduler wrapper advances once per process when batches are
    # sharded (not split), so the underlying scheduler uses world-size-scaled
    # steps while the optimizer still performs `total_updates` updates.
    scheduler_steps = total_updates * accelerator.num_processes
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(
            1,
            int(scheduler_steps * float(config["training"].get("warmup_ratio", 0.03))),
        ),
        num_training_steps=scheduler_steps,
    )
    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, scheduler
    )
    metrics_path = output_dir / "metrics.jsonl"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    best_loss = float("inf")
    global_step = 0
    micro_step = 0
    started = time.time()
    for epoch in range(epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()
        epoch_loss = torch.zeros(2, device=accelerator.device, dtype=torch.float64)
        for batch in train_loader:
            batch = move_batch(batch, accelerator.device)
            with accelerator.accumulate(model):
                output = loss_function(
                    model,
                    batch,
                    config=loss_config,
                    evaluating=False,
                    repeat_index=global_step,
                )
                accelerator.backward(output.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), float(config["training"].get("max_grad_norm", 1.0))
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    global_step += 1
            epoch_loss[0] += output.per_sample_loss.detach().double().sum()
            epoch_loss[1] += output.per_sample_loss.numel()
            micro_step += 1
        epoch_loss = accelerator.reduce(epoch_loss, reduction="sum")
        validation = evaluate_loader(
            model,
            validation_loader,
            accelerator,
            loss_function,
            loss_config,
        )
        row = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "micro_step": micro_step,
            "train_macro_loss": epoch_loss[0].item() / max(epoch_loss[1].item(), 1.0),
            "validation": validation,
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.time() - started,
        }
        if accelerator.is_main_process:
            append_jsonl(metrics_path, row)
        if validation["macro_loss"] < best_loss:
            best_loss = validation["macro_loss"]
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                name="best",
                config=config,
                metadata=row,
            )
        if (epoch + 1) % int(config["training"].get("save_every_epochs", 2)) == 0:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                name=f"epoch_{epoch + 1}",
                config=config,
                metadata=row,
            )
    save_checkpoint(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        name="last",
        config=config,
        metadata={
            "epochs": epochs,
            "global_step": global_step,
            "micro_step": micro_step,
            "best_macro_loss": best_loss,
        },
    )
    return {
        "best_macro_loss": best_loss,
        "epochs": epochs,
        "global_step": global_step,
        "micro_step": micro_step,
    }
