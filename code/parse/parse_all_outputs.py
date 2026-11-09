"""
전체 output + reference 파싱 → UD relation feature 추출
- 24000 predictions (6 model × 4 cond × 1000) 
- + 1000 references (조건 무관, 공유)
- degenerate output은 flag만 하고 skip
"""
import json, re, time, os
import stanza
from difflib import SequenceMatcher
from collections import defaultdict

nlp = stanza.Pipeline('ko', processors='tokenize,pos,lemma,depparse',
                       verbose=False, use_gpu=True)

MODELS = ["gemma3-4b", "llama31-8b", "mistral-7b", "qwen3-8b", "skt-ax4-light", "exaone35-7.8b"]
CONDS = ["C0_original", "C1_gpt_dlmin", "C2_algo_dlmin", "C3_algo_dlmax", "C4_algo_dlneu"]
PRED_DIR = "H/eval/predictions"
OUT_DIR = "H/relation_analysis/parses"
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
    """마커제거 → 문장분리 → echo제거 → depparse → relation 통계"""
    if is_degenerate(text):
        return None  # degenerate flag
    doc = nlp(text[:1500])
    
    # relation별 집계
    rel_counts = defaultdict(int)
    rel_distances = defaultdict(list)  # relation별 dependency distance
    total_edges = 0
    total_dist = 0
    n_tokens = 0
    used_sents = 0
    
    for sent in doc.sentences:
        s = sent.text.strip()
        if len(s) < 5: continue
        # echo 문장 제거
        if sent_sim(s, question) > 0.6: continue
        used_sents += 1
        words = sent.words
        for w in words:
            n_tokens += 1
            if w.deprel in ('punct', 'root'): continue
            if w.head == 0: continue
            dist = abs(w.id - w.head)
            rel = w.deprel.split(':')[0]  # nsubj:pass → nsubj
            rel_counts[rel] += 1
            rel_distances[rel].append(dist)
            total_edges += 1
            total_dist += dist
    
    if total_edges == 0:
        return None
    
    # relation별 평균 거리
    rel_mean_dist = {r: sum(d)/len(d) for r, d in rel_distances.items()}
    
    return {
        "n_tokens": n_tokens,
        "n_sents": used_sents,
        "total_edges": total_edges,
        "mean_dist": total_dist / total_edges,  # output MDD
        "rel_counts": dict(rel_counts),
        "rel_mean_dist": rel_mean_dist,
    }

# === Reference 먼저 파싱 (1000개, 공유) ===
print("=== Reference 파싱 (1000) ===")
t0 = time.time()
ref_recs = [json.loads(l) for l in open(f"{PRED_DIR}/C0_original_gemma3-4b.jsonl")]
ref_features = {}
for i, r in enumerate(ref_recs):
    doc_id = r.get("doc_id", i)
    feat = parse_answer(r["reference"], r["input"])
    ref_features[str(doc_id)] = feat
    if (i+1) % 200 == 0:
        print(f"  {i+1}/1000 ({time.time()-t0:.0f}s)")
json.dump(ref_features, open(f"{OUT_DIR}/reference_features.json","w"), ensure_ascii=False)
print(f"✓ Reference 완료 ({time.time()-t0:.0f}s)\n")

# === 각 조건 × 모델 output 파싱 ===
for c in CONDS:
    for m in MODELS:
        if not os.path.exists(f"{PRED_DIR}/{c}_{m}.jsonl"):
            print(f"⏭️  {c}_{m} 입력 없음"); continue
        out_path = f"{OUT_DIR}/{c}_{m}_features.json"
        if os.path.exists(out_path):
            print(f"⏭️  {c}_{m} 이미 완료")
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
            features[str(doc_id)] = feat  # None이면 degenerate
        json.dump(features, open(out_path,"w"), ensure_ascii=False)
        print(f"✓ {c}_{m}: {len(recs)-n_degen} parsed, {n_degen} degen ({time.time()-t0:.0f}s)")

print("\n===== ALL PARSING COMPLETE =====")
