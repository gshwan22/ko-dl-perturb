#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_model_v2.py — 배치 생성으로 10배 이상 속도 개선
- 기존: 1건씩 생성 (34초/건 × 5485건 = 52시간)
- 개선: 8건 배치 생성 (~3초/건 × 1000건 = 50분)
"""
import argparse, json, os, torch, gc
from tqdm import tqdm


def load_test_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def parse_chat_item(item):
    messages = item.get("messages", [])
    question, reference = "", ""
    for msg in messages:
        if msg["role"] == "user":
            question = msg["content"]
        elif msg["role"] == "assistant":
            reference = msg["content"]
    return question, reference


def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        pass
    # system 포함 시도
    messages_sys = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages_sys, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        pass
    return f"### Instruction:\n{question}\n\n### Response:\n"


def batch_generate(model, tokenizer, prompts, max_new_tokens, device):
    """좌측 패딩 배치 생성"""
    # Left-pad for batch generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    input_lengths = [inputs["attention_mask"][i].sum().item() for i in range(len(prompts))]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    predictions = []
    for i, (output, inp_len) in enumerate(zip(outputs, input_lengths)):
        new_tokens = output[inp_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        predictions.append(text)

    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_path", required=True)
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--trust_remote_code", default="false")
    args = ap.parse_args()

    trust_remote = args.trust_remote_code.lower() == "true"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Base model: {args.base_model_path}")
    print(f"[INFO] Adapter: {args.adapter_path}")
    print(f"[INFO] Test data: {args.test_data}")
    print(f"[INFO] Batch size: {args.batch_size}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=trust_remote,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=dtype,
        device_map="auto", trust_remote_code=trust_remote,
    )

    from peft import PeftModel
    if not os.path.exists(os.path.join(args.adapter_path, "adapter_config.json")):
        print(f"[ERROR] adapter not found: {args.adapter_path}")
        return

    model = PeftModel.from_pretrained(model, args.adapter_path, torch_dtype=dtype)
    model.eval()

    test_data = load_test_data(args.test_data)
    print(f"[INFO] {len(test_data)} test samples, batch_size={args.batch_size}")

    # Prepare all prompts
    all_items = []
    for item in test_data:
        q, ref = parse_chat_item(item)
        if q:
            prompt = build_prompt(tokenizer, q)
            all_items.append({"item": item, "question": q, "reference": ref, "prompt": prompt})

    # Batch generation
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    bs = args.batch_size

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(all_items), bs), desc="Generating batches"):
            batch = all_items[i:i+bs]
            prompts = [b["prompt"] for b in batch]

            try:
                predictions = batch_generate(
                    model, tokenizer, prompts, args.max_new_tokens, device
                )
            except torch.cuda.OutOfMemoryError:
                # OOM시 배치 크기 절반으로 줄여서 재시도
                torch.cuda.empty_cache()
                gc.collect()
                print(f"[WARN] OOM at batch {i}, retrying with batch_size=1")
                predictions = []
                for b in batch:
                    try:
                        pred = batch_generate(
                            model, tokenizer, [b["prompt"]], args.max_new_tokens, device
                        )
                        predictions.extend(pred)
                    except Exception as e:
                        print(f"[WARN] Single generation failed: {e}")
                        predictions.append("")
                        torch.cuda.empty_cache()
            except Exception as e:
                print(f"[WARN] Batch error: {e}")
                predictions = [""] * len(batch)

            for b, pred in zip(batch, predictions):
                meta = b["item"].get("meta", {})
                record = {
                    "input": b["question"],
                    "reference": b["reference"],
                    "prediction": pred,
                    "domain": meta.get("domain", ""),
                    "doc_id": meta.get("doc_id", -1),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[DONE] {len(all_items)} predictions → {args.output_file}")


if __name__ == "__main__":
    main()
