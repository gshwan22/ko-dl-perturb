"""
제2 파서(UDPipe korean-kaist) 재파싱 — parser-artifact 방어용
로직은 parse_all_outputs.py와 동일, 파서만 Stanza→UDPipe 교체
출력: H/relation_analysis/parses_udpipe/
"""
import json, re, time, os
import spacy_udpipe
from difflib import SequenceMatcher
from collections import defaultdict

nlp = spacy_udpipe.load("ko-kaist")

MODELS = ["gemma3-4b", "llama31-8b", "mistral-7b", "qwen3-8b", "skt-ax4-light", "exaone35-7.8b"]
CONDS = ["C0_original", "C1_gpt_dlmin", "C2_algo_dlmin", "C3_algo_dlmax", "C4_algo_dlneu"]
PRED_DIR = "H/eval/predictions"
OUT_DIR = "H/relation_analysis/parses_udpipe"
os.makedirs(OUT_DIR, exist_ok=True)

def strip_markers(raw):
    text = raw
    for marker in ['[|assistant|]', '<|assistant|>', 'assistant\n', 'model\n', '[/INST]']:
        if marker in text:
            text = text[text.rfind(marker)+len(marker):]; break
    else:
        for marker in ['\nmodel\n', '\nassistant\n']:
            if marker in text:
                text = text[text.rfind(marker)+len(marker):]; break
    text = re.sub(r'<think>\s*</think>|</?think>|<\|[^|]*\|>|\[\|[^|]*\|\]', '', text)
    return text.strip()

def sent_sim(a, b):
    return SequenceMatcher(None, re.sub(r'\s+','',a), re.sub(r'\s+','',b)).ratio()

def is_degenerate(text):
    toks = text.split()
    if len(toks) < 5: return True
    return len(set(toks))/len(toks) < 0.3

def parse_answer(text, question):
    if is_degenerate(text):
        return None
    doc = nlp(text[:1500])

    rel_counts = defaultdict(int)
    rel_distances = defaultdict(list)
    total_edges = 0
    total_dist = 0
    n_tokens = 0
    used_sents = 0

    for sent in doc.sents:
        s = sent.text.strip()
        if len(s) < 5: continue
        if sent_sim(s, question) > 0.6: continue
        used_sents += 1
        start = sent.start
        for tok in sent:
            n_tokens += 1
            rel_full = tok.dep_.lower()
            if rel_full in ('punct', 'root'): continue
            if tok.head == tok: continue  # root 토큰 (head 자기자신)
            wid = tok.i - start + 1
            hid = tok.head.i - start + 1
            dist = abs(wid - hid)
            rel = rel_full.split(':')[0]
            rel_counts[rel] += 1
            rel_distances[rel].append(dist)
            total_edges += 1
            total_dist += dist

    if total_edges == 0:
        return None

    rel_mean_dist = {r: sum(d)/len(d) for r, d in rel_distances.items()}
    return {
        "n_tokens": n_tokens,
        "n_sents": used_sents,
        "total_edges": total_edges,
        "mean_dist": total_dist/total_edges,
        "rel_counts": dict(rel_counts),
        "rel_mean_dist": rel_mean_dist,
    }

# === reference 파싱 (조건 무관 공유) ===
ref_path = f"{OUT_DIR}/reference_features.json"
if not os.path.exists(ref_path):
    t0 = time.time()
    recs = [json.loads(l) for l in open(f"{PRED_DIR}/C0_original_gemma3-4b.jsonl")]
    features = {}
    for i, r in enumerate(recs):
        doc_id = r.get("doc_id", i)
        features[str(doc_id)] = parse_answer(r["reference"], r["input"])
    json.dump(features, open(ref_path, "w"), ensure_ascii=False)
    print(f"✓ reference: {time.time()-t0:.0f}s", flush=True)
else:
    print("⏭️  reference 이미 완료", flush=True)

# === 각 조건 × 모델 ===
for c in CONDS:
    for m in MODELS:
        if not os.path.exists(f"{PRED_DIR}/{c}_{m}.jsonl"):
            print(f"⏭️  {c}_{m} 입력 없음"); continue
        out_path = f"{OUT_DIR}/{c}_{m}_features.json"
        if os.path.exists(out_path):
            print(f"⏭️  {c}_{m} 이미 완료", flush=True)
            continue
        t0 = time.time()
        recs = [json.loads(l) for l in open(f"{PRED_DIR}/{c}_{m}.jsonl")]
        features = {}
        n_degen = 0
        for i, r in enumerate(recs):
            doc_id = r.get("doc_id", i)
            feat = parse_answer(strip_markers(r["prediction"]), r["input"])
            if feat is None:
                n_degen += 1
            features[str(doc_id)] = feat
        json.dump(features, open(out_path, "w"), ensure_ascii=False)
        print(f"✓ {c}_{m}: {len(recs)-n_degen} parsed, {n_degen} degen ({time.time()-t0:.0f}s)", flush=True)

print("\n===== UDPIPE PARSING COMPLETE =====", flush=True)
