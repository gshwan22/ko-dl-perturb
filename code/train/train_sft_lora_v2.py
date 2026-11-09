#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_sft_lora_v2.py — 다중 모델 LoRA SFT 학습 스크립트
- train_sft_lora_min.py 기반, trust_remote_code 지원 추가
- Gemma/Qwen/Llama/Mistral/EXAONE/SKT-AX/Phi 호환
- torchrun 2GPU DDP 지원
"""
import argparse, inspect, os
from typing import List, Dict, Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


def apply_chat_template_or_fallback(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass
    chunks = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        chunks.append(f"[{role}]\n{content}")
    return "\n".join(chunks) + "\n"


def pick_lora_target_modules(model) -> List[str]:
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "fc1", "fc2", "dense", "query", "key", "value",
        "qkv_proj", "gate_up_proj",  # Phi-4 style
    ]
    found = set()
    for name, _ in model.named_modules():
        for c in candidates:
            if name == c or name.endswith("." + c):
                found.add(c)
    picked = [c for c in candidates if c in found]
    if not picked:
        picked = ["q_proj", "k_proj", "v_proj", "o_proj"]
    return picked


def make_training_args(args) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    ta_kwargs: Dict[str, Any] = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=True,
    )

    if torch.cuda.is_available():
        ta_kwargs["bf16"] = True
        ta_kwargs["fp16"] = False

    if "evaluation_strategy" in params:
        ta_kwargs["evaluation_strategy"] = args.eval_strategy
    elif "eval_strategy" in params:
        ta_kwargs["eval_strategy"] = args.eval_strategy

    if "report_to" in params:
        ta_kwargs["report_to"] = "none"

    return TrainingArguments(**ta_kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--eval_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--trust_remote_code", default="false",
                     help="true/false — EXAONE, SKT 등에 필요")

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--eval_strategy", type=str, default="steps")
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--dataloader_num_workers", type=int, default=2)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    args = ap.parse_args()
    trust_remote = args.trust_remote_code.lower() == "true"
    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=None,  # DDP
        trust_remote_code=trust_remote,
    )

    # LoRA
    target_modules = pick_lora_target_modules(model)
    print(f"[INFO] LoRA target_modules: {target_modules}")
    print(f"[INFO] trust_remote_code: {trust_remote}")
    print(f"[INFO] Model type: {type(model).__name__}")

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Dataset
    ds = load_dataset("json", data_files={"train": args.train_file, "eval": args.eval_file})

    def format_and_tokenize(ex):
        text = apply_chat_template_or_fallback(tokenizer, ex["messages"])
        tok = tokenizer(text, truncation=True, max_length=args.max_length, padding=False)
        return tok

    remove_cols = ds["train"].column_names
    ds_tok = ds.map(format_and_tokenize, remove_columns=remove_cols)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    targs = make_training_args(args)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["eval"],
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[DONE] saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
