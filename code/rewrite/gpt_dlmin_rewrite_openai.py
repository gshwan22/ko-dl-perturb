#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpt_dlmin_rewrite_openai.py

OpenAI API(Responses)로 SFT JSONL을 읽어서, 지정한 role(user/assistant)의 content를
"DL-min(의존거리 최소화) 목적"으로 재작성한 뒤 JSONL로 저장.

핵심 기능
- 대용량 안전: batch 단위 동시처리(concurrency), 체크포인트(state)로 재시작(resume)
- 오류 내구성: 429/5xx 재시도(backoff), 실패 시 원문 유지 옵션
- 출력: 한 줄당 1 example(JSONL). 원본을 유지하면서 view/meta/orig_index를 붙임.
- 프롬프트 모드:
  - length_preserve(기본): 요약/삭제 방지(길이/문장수 유지 제약)
  - free: 더 공격적으로 DL을 줄이도록 유도(권장 X, 길이 줄어들 수 있음)

요구사항
- openai>=2.x (너는 2.8.1 확인됨)
- 환경변수 OPENAI_API_KEY 설정

예시(샘플 200)
  python gpt_dlmin_rewrite_openai.py \
    --src /home/user/Desktop/TestG/Prj-vol1/tools/qa_dataset/sft_side_p1.0.train.jsonl \
    --out /home/user/Desktop/TestG/Prj-vol1/tools/qa_dataset/sft_side_p1.0.gptdlmin.sample200.jsonl \
    --model gpt-4.1 \
    --limit 200 \
    --rewrite_roles assistant \
    --mode length_preserve \
    --concurrency 12 \
    --batch_size 24 \
    --max_output_tokens 512 \
    --log /home/user/Desktop/TestG/Prj-vol1/tools/qdata/metrics/gptdlmin_sample200.log \
    --state /home/user/Desktop/TestG/Prj-vol1/tools/qdata/metrics/gptdlmin_sample200.state.json \
    --resume

전체(주의: 비용/시간 큼)
  python gpt_dlmin_rewrite_openai.py \
    --src /home/user/Desktop/TestG/Prj-vol1/tools/qa_dataset/sft_side_p1.0.train.jsonl \
    --out /home/user/Desktop/TestG/Prj-vol1/tools/qa_dataset/sft_side_p1.0.gptdlmin.train.jsonl \
    --model gpt-4.1 \
    --rewrite_roles assistant \
    --mode length_preserve \
    --concurrency 24 \
    --batch_size 48 \
    --max_output_tokens 1024 \
    --log /home/user/Desktop/TestG/Prj-vol1/tools/qdata/metrics/gptdlmin_train.log \
    --state /home/user/Desktop/TestG/Prj-vol1/tools/qdata/metrics/gptdlmin_train.state.json \
    --resume
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


# -----------------------------
# Prompts
# -----------------------------

DL_DEF_KO = (
    "Dependency Length(DL)은 문장에서 의존관계(dependency) (i→h)들의 거리 |i-h|의 합 τ=Σ|i-h|로 정의한다. "
    "여기서 i는 의존어 토큰 위치, h는 지배어(head) 토큰 위치이다(루트 제외). "
    "DL-min 재작성은 의미를 유지하면서 관련 단어들을 더 가깝게 배치하여 τ를 줄이도록 단어/구/절의 순서를 재배열하는 것이다."
)

SYSTEM_FREE = (
    "You are a Korean scientific writing assistant. "
    "Rewrite Korean text to minimize total dependency length (DL) while preserving meaning and factual content. "
    "Return ONLY the rewritten text (no commentary)."
)

SYSTEM_LENGTH_PRESERVE = (
    "You are a Korean scientific writing assistant. "
    "Rewrite Korean text to minimize total dependency length (DL) while preserving meaning and factual content. "
    "IMPORTANT CONSTRAINTS:\n"
    "- Do NOT summarize, omit, or add information.\n"
    "- Keep the number of sentences the same if possible (at most ±1).\n"
    "- Keep length similar (roughly within ±10% tokens/characters).\n"
    "- Keep all key terms, numbers, lists, and entities.\n"
    "- Keep the style formal and explanatory.\n"
    "Return ONLY the rewritten text (no commentary)."
)

USER_TEMPLATE = (
    "다음 한국어 문장을 DL-min 관점으로 재작성하라.\n"
    "정의: {dl_def}\n"
    "지침:\n"
    "1) 의미/정보 보존(요약/삭제 금지), 사실/수치/전문용어 유지.\n"
    "2) 가능하면 문장 수 유지(±1 이내), 길이 유사 유지(±10% 권장).\n"
    "3) 한국어 문장으로 자연스럽게.\n"
    "4) 출력은 재작성된 본문만.\n"
    "\n"
    "[원문]\n{text}\n"
    "\n"
    "[재작성]"
)

# 간단한 "설명 섞임" 제거용 (모델이 규칙을 어기면 방어)
RX_LEADING_META = re.compile(r"^\s*(재작성(된)?\s*문장|결과|답변|Output|Rewritten)\s*[:：]\s*", re.I)


# -----------------------------
# Utils
# -----------------------------

def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def read_jsonl_line(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield i, line

def safe_json_load(line: str) -> Optional[dict]:
    try:
        return json.loads(line)
    except Exception:
        return None

def get_first_role_content(messages: list, role: str) -> Optional[str]:
    role_l = role.lower()
    for m in messages or []:
        if isinstance(m, dict) and str(m.get("role", "")).lower() == role_l:
            return str(m.get("content", ""))
    return None

def rewrite_roles_in_messages(messages: list, roles_to_rewrite: set, new_texts: Dict[str, str]) -> list:
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            out.append(m)
            continue
        r = str(m.get("role", "")).lower()
        if r in roles_to_rewrite and r in new_texts and isinstance(m.get("content", None), str):
            nm = dict(m)
            nm["content"] = new_texts[r]
            out.append(nm)
        else:
            out.append(m)
    return out

def strip_bad_prefix(s: str) -> str:
    s2 = RX_LEADING_META.sub("", s.strip())
    return s2.strip()

def clamp_min_max_output_tokens(x: int) -> int:
    # OpenAI Responses API: max_output_tokens >= 16
    return max(16, int(x))

def mk_prompt(mode: str, text: str) -> Tuple[str, str]:
    if mode == "free":
        sys_prompt = SYSTEM_FREE
    else:
        sys_prompt = SYSTEM_LENGTH_PRESERVE
    user_prompt = USER_TEMPLATE.format(dl_def=DL_DEF_KO, text=text)
    return sys_prompt, user_prompt


# -----------------------------
# Retry logic
# -----------------------------

@dataclass
class RetryConf:
    max_retries: int = 6
    base_sleep: float = 1.0
    max_sleep: float = 30.0

def jitter_sleep(t: float) -> float:
    return t * (0.75 + random.random() * 0.5)

def is_retryable_exception(e: Exception) -> bool:
    # openai.* errors are not always import-stable across versions; detect by name / message
    name = e.__class__.__name__.lower()
    msg = str(e).lower()
    if "rate limit" in msg or "429" in msg:
        return True
    if "timeout" in msg or "timed out" in msg:
        return True
    if "server error" in msg or "500" in msg or "502" in msg or "503" in msg or "504" in msg:
        return True
    if "temporarily" in msg:
        return True
    if "connection" in msg:
        return True
    if "apierror" in name or "apiconnectionerror" in name or "ratelimiterror" in name:
        return True
    return False


# -----------------------------
# Core OpenAI call
# -----------------------------

async def dlmin_rewrite_text(
    client: AsyncOpenAI,
    model: str,
    mode: str,
    text: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    retry: RetryConf,
) -> str:
    sys_prompt, user_prompt = mk_prompt(mode, text)
    max_output_tokens = clamp_min_max_output_tokens(max_output_tokens)

    sleep = retry.base_sleep
    last_err = None

    for attempt in range(retry.max_retries + 1):
        try:
            # Responses API
            r = await client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            out = getattr(r, "output_text", None) or ""
            out = strip_bad_prefix(out)
            # 빈 출력이면 방어적으로 원문 유지
            if not out.strip():
                return text
            return out

        except Exception as e:
            last_err = e
            if attempt >= retry.max_retries or not is_retryable_exception(e):
                raise
            await asyncio.sleep(jitter_sleep(min(sleep, retry.max_sleep)))
            sleep *= 2.0

    # 여기 도달하진 않지만 안전
    if last_err:
        raise last_err
    return text


# -----------------------------
# Processing one example
# -----------------------------

async def process_one(
    client: AsyncOpenAI,
    ex: Dict[str, Any],
    idx: int,
    model: str,
    rewrite_roles: set,
    mode: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    retry: RetryConf,
    keep_original_on_error: bool,
    log_fp,
) -> Dict[str, Any]:
    """
    입력 ex(JSON object)을 받아 rewrite_roles에 해당하는 메시지 content를 gptdlmin으로 바꾼 새 ex를 반환.
    """
    messages = ex.get("messages", [])
    if not isinstance(messages, list):
        # 이상한 형식이면 그대로
        ex2 = dict(ex)
        ex2["view"] = ex2.get("view", "orig")
        ex2["view"] = "gptdlmin"
        ex2["orig_index"] = ex.get("orig_index", idx)
        return ex2

    # 역할별 원문 수집
    new_texts = {}
    for r in rewrite_roles:
        orig = get_first_role_content(messages, r)
        if orig is None or not str(orig).strip():
            continue
        try:
            new = await dlmin_rewrite_text(
                client=client,
                model=model,
                mode=mode,
                text=str(orig),
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                retry=retry,
            )
            new_texts[r] = new
        except Exception as e:
            # 실패 처리
            log_fp.write(f"{now_ts()} ERROR idx={idx} role={r} err={repr(e)}\n")
            log_fp.flush()
            if keep_original_on_error:
                new_texts[r] = str(orig)
            else:
                raise

    # 메시지 업데이트
    messages2 = rewrite_roles_in_messages(messages, rewrite_roles, new_texts)

    ex2 = dict(ex)
    ex2["messages"] = messages2
    ex2["orig_index"] = ex.get("orig_index", idx)
    ex2["view"] = "gptdlmin"

    # meta 정리(있으면 유지 + used_answer_view 표시)
    meta = ex2.get("meta", {})
    if isinstance(meta, dict):
        meta2 = dict(meta)
        meta2["used_answer_view"] = meta2.get("used_answer_view", meta2.get("used_answer_view", "orig"))
        meta2["rewritten_by"] = "openai"
        meta2["rewrite_mode"] = mode
        meta2["rewrite_roles"] = sorted(list(rewrite_roles))
        meta2["rewrite_model"] = model
        ex2["meta"] = meta2

    return ex2


# -----------------------------
# State / resume
# -----------------------------

def load_state(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(path: str, state: dict):
    if not path:
        return
    ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# -----------------------------
# Main
# -----------------------------

async def run(args):
    ensure_dir(args.out)
    if args.log:
        ensure_dir(args.log)
    if args.state:
        ensure_dir(args.state)

    # roles
    roles = set([r.strip().lower() for r in args.rewrite_roles.split(",") if r.strip()])
    valid_roles = {"assistant", "user"}
    if not roles.issubset(valid_roles):
        raise ValueError(f"--rewrite_roles must be subset of {valid_roles}, got {roles}")
    if not roles:
        roles = {"assistant"}

    # state
    st = load_state(args.state) if (args.resume and args.state) else {}
    start_line = int(st.get("next_line", 0)) if (args.resume and args.state) else 0
    done = int(st.get("done", 0)) if (args.resume and args.state) else 0
    errc = int(st.get("errors", 0)) if (args.resume and args.state) else 0

    # if user provided start overrides state
    if args.start is not None:
        start_line = max(0, int(args.start))

    # open logs
    log_fp = open(args.log, "a", encoding="utf-8") if args.log else open(os.devnull, "w", encoding="utf-8")
    log_fp.write(f"{now_ts()} INFO start run model={args.model} mode={args.mode} roles={sorted(list(roles))}\n")
    log_fp.flush()

    client = AsyncOpenAI()

    sem = asyncio.Semaphore(args.concurrency)
    retry = RetryConf(max_retries=args.max_retries, base_sleep=args.base_sleep, max_sleep=args.max_sleep)

    async def guarded_process(ex, idx):
        async with sem:
            return await process_one(
                client=client,
                ex=ex,
                idx=idx,
                model=args.model,
                rewrite_roles=roles,
                mode=args.mode,
                max_output_tokens=args.max_output_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                retry=retry,
                keep_original_on_error=args.keep_original_on_error,
                log_fp=log_fp,
            )

    # output file open (append if resume)
    out_mode = "a" if (args.resume and os.path.exists(args.out) and os.path.getsize(args.out) > 0) else "w"
    with open(args.out, out_mode, encoding="utf-8") as fout:
        batch = []
        batch_idx = []
        wrote = 0

        # stream input
        for i, line in read_jsonl_line(args.src):
            if i < start_line:
                continue
            if args.end is not None and i >= int(args.end):
                break
            if args.limit is not None and wrote >= int(args.limit):
                break

            ex = safe_json_load(line)
            if ex is None:
                errc += 1
                log_fp.write(f"{now_ts()} ERROR bad_json idx={i}\n")
                log_fp.flush()
                continue

            # assign orig_index if missing
            ex.setdefault("orig_index", i)

            batch.append(ex)
            batch_idx.append(i)

            if len(batch) >= args.batch_size:
                tasks = [asyncio.create_task(guarded_process(exj, idxj)) for exj, idxj in zip(batch, batch_idx)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # write in order of batch
                for idxj, res in zip(batch_idx, results):
                    if isinstance(res, Exception):
                        errc += 1
                        log_fp.write(f"{now_ts()} ERROR idx={idxj} exc={repr(res)}\n")
                        log_fp.flush()
                        if args.keep_original_on_error:
                            # fallback: write original line but mark view=gptdlmin_failed
                            ex_fallback = batch[batch_idx.index(idxj)]
                            ex2 = dict(ex_fallback)
                            ex2["view"] = "gptdlmin_failed"
                            ex2["orig_index"] = ex_fallback.get("orig_index", idxj)
                            fout.write(jdump(ex2) + "\n")
                            wrote += 1
                        else:
                            # strict: stop
                            raise res
                    else:
                        fout.write(jdump(res) + "\n")
                        wrote += 1

                fout.flush()

                # update state to next line after this batch's last index
                next_line = batch_idx[-1] + 1
                done += len(batch_idx)
                if args.state:
                    save_state(args.state, {"next_line": next_line, "done": done, "errors": errc, "out": args.out})

                log_fp.write(f"{now_ts()} INFO progress next_line={next_line} wrote={wrote} errors={errc}\n")
                log_fp.flush()

                batch, batch_idx = [], []

        # flush remaining
        if batch:
            tasks = [asyncio.create_task(guarded_process(exj, idxj)) for exj, idxj in zip(batch, batch_idx)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idxj, res in zip(batch_idx, results):
                if args.limit is not None and wrote >= int(args.limit):
                    break
                if isinstance(res, Exception):
                    errc += 1
                    log_fp.write(f"{now_ts()} ERROR idx={idxj} exc={repr(res)}\n")
                    log_fp.flush()
                    if args.keep_original_on_error:
                        ex_fallback = batch[batch_idx.index(idxj)]
                        ex2 = dict(ex_fallback)
                        ex2["view"] = "gptdlmin_failed"
                        ex2["orig_index"] = ex_fallback.get("orig_index", idxj)
                        fout.write(jdump(ex2) + "\n")
                        wrote += 1
                    else:
                        raise res
                else:
                    fout.write(jdump(res) + "\n")
                    wrote += 1

            fout.flush()

            next_line = batch_idx[-1] + 1
            done += len(batch_idx)
            if args.state:
                save_state(args.state, {"next_line": next_line, "done": done, "errors": errc, "out": args.out})

    log_fp.write(f"{now_ts()} DONE wrote={wrote} errors={errc} out={args.out}\n")
    log_fp.flush()
    if log_fp is not sys.stdout:
        log_fp.close()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="input JSONL (SFT messages)")
    ap.add_argument("--out", required=True, help="output JSONL (gptdlmin)")
    ap.add_argument("--model", default="gpt-4.1", help="OpenAI model name (e.g., gpt-4.1, gpt-4.1-mini)")
    ap.add_argument("--mode", choices=["length_preserve", "free"], default="length_preserve")
    ap.add_argument("--rewrite_roles", default="assistant", help="comma-separated: assistant,user (default assistant)")
    ap.add_argument("--max_output_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=24)

    ap.add_argument("--limit", type=int, default=None, help="max number of written examples (for sampling)")
    ap.add_argument("--start", type=int, default=None, help="start line index in src (overrides state)")
    ap.add_argument("--end", type=int, default=None, help="end line index (exclusive) in src")
    ap.add_argument("--resume", action="store_true", help="resume from --state")
    ap.add_argument("--state", default=None, help="state json path for resume/checkpoint")
    ap.add_argument("--log", default=None, help="log file path")

    ap.add_argument("--keep_original_on_error", action="store_true", help="on error, write fallback example and continue")
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--base_sleep", type=float, default=1.0)
    ap.add_argument("--max_sleep", type=float, default=30.0)
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        print("[FATAL] OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(2)

    # safety clamp
    args.concurrency = max(1, args.concurrency)
    args.batch_size = max(1, args.batch_size)
    args.max_output_tokens = clamp_min_max_output_tokens(args.max_output_tokens)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()

