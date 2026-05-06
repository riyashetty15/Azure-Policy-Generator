import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_qlora")


@dataclass
class TrainConfig:
    model_name: str
    data_file: str
    output_dir: str
    adapter_dir: str
    test_size: float
    seed: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    max_steps: int
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float


def pick_precision() -> Dict[str, bool]:
    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}

    major, _ = torch.cuda.get_device_capability(0)
    # bf16 is generally a safe default on Ampere+.
    if major >= 8:
        return {"fp16": False, "bf16": True}
    return {"fp16": True, "bf16": False}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_with_fallback(model_name: str, precision: Dict[str, bool], device: str) -> Tuple[AutoModelForCausalLM, bool]:
    """Return model and whether 4-bit quantization is active."""
    use_4bit = False

    if device == "cuda":
        try:
            import bitsandbytes  # noqa: F401

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if precision["bf16"] else torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant,
                device_map="auto",
            )
            use_4bit = True
            return model, use_4bit
        except Exception as e:
            logger.warning("4-bit load unavailable (%s). Falling back to regular precision.", e)

    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if device in {"cuda", "mps"}:
        model = model.to(device)
    return model, use_4bit


def estimate_max_len(ds, tok, fallback: int, cap: int = 2048) -> int:
    lengths: List[int] = []
    for ex in ds:
        text = f"{ex['prompt']}\n{ex['completion']}"
        n = len(tok(text, add_special_tokens=True).input_ids)
        lengths.append(n)

    if not lengths:
        return fallback

    lengths = sorted(lengths)
    idx = int(0.95 * (len(lengths) - 1))
    p95 = lengths[idx]
    # Round up to nearest 64 for better kernel efficiency.
    rounded = int(math.ceil(p95 / 64.0) * 64)
    chosen = min(max(rounded, 256), cap)
    logger.info("Token length stats: min=%s p95=%s max=%s -> max_length=%s", lengths[0], p95, lengths[-1], chosen)
    return chosen


def tokenize_fn_builder(tok, max_len: int):
    def _tokenize(ex):
        prompt = ex["prompt"].strip()
        completion = ex["completion"].strip()
        full = f"{prompt}\n{completion}"

        full_enc = tok(full, truncation=True, max_length=max_len)
        prompt_enc = tok(prompt, truncation=True, max_length=max_len)

        input_ids = full_enc["input_ids"]
        labels = input_ids.copy()
        # Mask prompt tokens so loss focuses on completion.
        prompt_len = len(prompt_enc["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": full_enc["attention_mask"],
            "labels": labels,
        }

    return _tokenize


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Azure Policy QLoRA adapter")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="azure_policies_ft_supervised_augmented.jsonl")
    parser.add_argument("--output-dir", default="azure-policy-qlora")
    parser.add_argument("--adapter-dir", default="azure-policy-qlora-adapter")
    parser.add_argument("--test-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="If >0, overrides epochs for a bounded run")
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=0, help="0 = auto from p95 token length")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    return TrainConfig(
        model_name=args.model,
        data_file=args.data,
        output_dir=args.output_dir,
        adapter_dir=args.adapter_dir,
        test_size=args.test_size,
        seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


def main() -> None:
    cfg = parse_args()
    device = pick_device()
    logger.info("Training device: %s", device)

    if not Path(cfg.data_file).exists():
        raise FileNotFoundError(f"Training data not found: {cfg.data_file}")

    logger.info("Loading tokenizer: %s", cfg.model_name)
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    logger.info("Loading dataset: %s", cfg.data_file)
    ds = load_dataset("json", data_files=cfg.data_file, split="train")
    ds = ds.train_test_split(test_size=cfg.test_size, seed=cfg.seed)

    max_len = cfg.max_length if cfg.max_length > 0 else estimate_max_len(ds["train"], tok, fallback=1024)
    tokenize_fn = tokenize_fn_builder(tok, max_len=max_len)

    ds_tok = ds.map(
        tokenize_fn,
        remove_columns=ds["train"].column_names,
        num_proc=min(os.cpu_count() or 1, 8),
        desc="Tokenizing",
    )

    logger.info("Loading base model")
    model, use_4bit = load_model_with_fallback(cfg.model_name, pick_precision(), device)
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    precision = pick_precision()
    optim_name = "paged_adamw_8bit" if use_4bit else "adamw_torch"
    train_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=precision["fp16"],
        bf16=precision["bf16"],
        gradient_checkpointing=True,
        optim=optim_name,
        report_to="none",
        dataloader_num_workers=min(os.cpu_count() or 1, 4),
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving adapter and tokenizer to: %s", cfg.adapter_dir)
    Path(cfg.adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.adapter_dir)
    tok.save_pretrained(cfg.adapter_dir)

    results = trainer.evaluate(eval_dataset=ds_tok["test"])
    logger.info("Eval: %s", json.dumps(results, indent=2))

    metrics_path = Path(cfg.output_dir) / "eval_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote metrics: %s", metrics_path)


if __name__ == "__main__":
    main()