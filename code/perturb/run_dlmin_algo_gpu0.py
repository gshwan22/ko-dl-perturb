#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_dlmin_algo_gpu0.py — DL-min 알고리즘 리라이팅 (GPU 0, 전반부)
"""

import json, os, random, time, csv
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import stanza

SRC = "tools/qa_dataset/sft_side_p0.0.train.jsonl"
OUT = "tools/qa_dataset/sft_side_p0.0.dlmin_algo.part0.jsonl"
STATE = "tools/qdata/metrics/dlmin_algo_gpu0.state.json"
N_SHUFFLES = 100
N_VERIFY = 8
SEED = 42
CHECKPOINT = 500

START_LINE = 0
END_LINE = 268077

nlp = stanza.Pipeline("ko", processors="tokenize,pos,lemma,depparse",
                       use_gpu=True, logging_level="WARN")

def calc_mdd(text):
    doc = nlp(text)
    dl, n = 0, 0
    for s in doc.sentences:
        for w in s.words:
            if w.head > 0:
                dl += abs(w.id - w.head)
                n += 1
    return dl / n if n else 0

def dlmin_rewrite(text, seed_offset=0):
    rng = random.Random(SEED + seed_offset)
    doc = nlp(text)

    rewritten_sents = []
    for sent in doc.sentences:
        words = [(w.id, w.text, w.head) for w in sent.words]
        if len(words) < 3:
            rewritten_sents.append(sent.text)
            continue

        root_ids = set(w[0] for w in words if w[2] == 0)
        id2text = {w[0]: w[1] for w in words}

        candidates = []
        for _ in range(N_SHUFFLES):
            movable = [w[0] for w in words if w[0] not in root_ids]
            rng.shuffle(movable)
            order = movable + list(root_ids)
            candidates.append(" ".join(id2text[wid] for wid in order))

        sent_best = sent.text
        sent_best_mdd = calc_mdd(sent.text)

        for cand in rng.sample(candidates, min(N_VERIFY, len(candidates))):
            mdd = calc_mdd(cand)
            if mdd < sent_best_mdd:  # 최소화!
                sent_best_mdd = mdd
                sent_best = cand

        rewritten_sents.append(sent_best)

    return " ".join(rewritten_sents)

def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {}

def save_state(state):
    d = os.path.dirname(STATE)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE)

def main():
    st = load_state()
    resume_line = st.get("next_line", START_LINE)
    wrote = st.get("wrote", 0)
    errors = st.get("errors", 0)

    out_mode = "a" if wrote > 0 else "w"
    d = os.path.dirname(OUT)
    if d:
        os.makedirs(d, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] GPU0 DL-min START: lines {resume_line}-{END_LINE}, wrote={wrote}")

    t0 = time.time()

    with open(SRC) as fin, open(OUT, out_mode) as fout:
        for line_no, line in enumerate(fin):
            if line_no < resume_line:
                continue
            if line_no >= END_LINE:
                break

            line = line.strip()
            if not line:
                continue

            try:
                ex = json.loads(line)
                ex.setdefault("orig_index", line_no)

                messages = ex.get("messages", [])
                for i, msg in enumerate(messages):
                    if msg.get("role") == "assistant" and msg.get("content", "").strip():
                        messages[i]["content"] = dlmin_rewrite(msg["content"], seed_offset=line_no)

                ex["messages"] = messages
                ex["view"] = "dlmin_algo"
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                wrote += 1

            except Exception as e:
                errors += 1
                print(f"  [ERROR] line={line_no}: {repr(e)}")
                try:
                    ex["view"] = "dlmin_algo_failed"
                    fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    wrote += 1
                except:
                    pass

            if wrote % CHECKPOINT == 0:
                elapsed = time.time() - t0
                speed = wrote / elapsed if elapsed > 0 else 0
                remaining = (END_LINE - line_no) / speed / 3600 if speed > 0 else 0
                print(f"[{time.strftime('%H:%M:%S')}] GPU0 DL-min wrote={wrote:,} err={errors} "
                      f"speed={speed:.1f}/s ETA={remaining:.1f}h")
                fout.flush()
                save_state({"next_line": line_no + 1, "wrote": wrote, "errors": errors})

    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] GPU0 DL-min DONE! wrote={wrote:,} errors={errors} "
          f"elapsed={elapsed/3600:.1f}h")

if __name__ == "__main__":
    main()
