#!/usr/bin/env python3
"""pi0: MDD-neutral scramble. accept iff |dMDD|/MDD<=tol AND tau>=tau_min.
   pi+/pi- 기계와 동일 구조(best-of-N, 고정 트리 채점, 실패 시 원문 유지)."""
import json, random, argparse, os, sys
random.seed(12345)

def tau_dist(perm):  # normalized Kendall distance vs identity
    n = len(perm); inv = 0
    for i in range(n):
        for j in range(i+1, n):
            if perm[i] > perm[j]: inv += 1
    return inv / (n*(n-1)/2) if n > 1 else 0.0

def sent_mdd(order_pos, heads):  # heads: 1-based, 0=root; order_pos[i]=position of word i
    ds = [abs(order_pos[i] - order_pos[h-1]) for i, h in enumerate(heads) if h > 0]
    return sum(ds)/len(ds) if ds else 0.0

def rewrite_sentence(words, heads, tol, tau_min, restarts=8, repair_iters=400):
    n = len(words)
    if n < 3: return words, "short", 0.0, 0.0
    ident = list(range(n))
    edges = [(i, h-1) for i, h in enumerate(heads) if h > 0]
    if not edges: return words, "short", 0.0, 0.0
    def mdd_of(pos): return sum(abs(pos[a]-pos[b]) for a, b in edges) / len(edges)
    mdd0 = mdd_of({i: i for i in ident})
    if mdd0 == 0: return words, "short", 0.0, 0.0
    best = None
    for _ in range(restarts):
        perm = ident[:]; random.shuffle(perm)
        pos = {w: k for k, w in enumerate(perm)}
        cur = abs(mdd_of(pos) - mdd0)
        for _ in range(repair_iters):
            if cur / mdd0 <= tol: break
            i, j = random.randrange(n), random.randrange(n)
            if i == j: continue
            perm[i], perm[j] = perm[j], perm[i]
            pos[perm[i]], pos[perm[j]] = i, j
            nxt = abs(mdd_of(pos) - mdd0)
            if nxt < cur:
                cur = nxt
            else:
                perm[i], perm[j] = perm[j], perm[i]
                pos[perm[i]], pos[perm[j]] = i, j
        rel = cur / mdd0
        t = tau_dist([pos[i] for i in ident])
        if rel <= tol and t >= tau_min:
            if best is None or (rel, -t) < (best[0], -best[1]):
                best = (rel, t, perm[:])
            if rel == 0: break
    if best is None: return words, "fallback", 0.0, 0.0
    rel, t, perm = best
    return [words[i] for i in perm], "ok", rel, t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--mdd_tol", type=float, default=0.01)
    ap.add_argument("--tau_min_file", default="")
    ap.add_argument("--tau_min", type=float, default=0.35)
    ap.add_argument("--field", default="")
    args = ap.parse_args()
    tau_min = args.tau_min
    if args.tau_min_file and os.path.exists(args.tau_min_file):
        tau_min = float(open(args.tau_min_file).read().strip())
    print(f"[INFO] tau_min={tau_min:.3f} mdd_tol={args.mdd_tol}")

    import stanza
    nlp = stanza.Pipeline("ko", processors="tokenize,pos,lemma,depparse",
                          use_gpu=False, verbose=False)

    field = args.field
    stats = {"ok": 0, "fallback": 0, "short": 0}; rels = []; taus = []
    announced = False
    with open(args.in_file) as f, open(args.out_file, "w") as out:
        for ln, line in enumerate(f):
            r = json.loads(line)
            targets = []
            if "messages" in r and isinstance(r["messages"], list):
                for msg in r["messages"]:
                    if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                        targets.append((msg, "content"))
                if not announced:
                    print("[INFO] format=messages (assistant turns)"); announced = True
            else:
                if not field:
                    for k in ["output", "answer", "response", "completion", "reference"]:
                        if k in r and isinstance(r[k], str):
                            field = k; break
                    if not field:
                        sys.exit(f"[ABORT] answer field not found in keys={list(r)}")
                    print(f"[INFO] field={field}")
                targets.append((r, field))
            for holder, key in targets:
                doc = nlp(holder[key])
                new_sents = []
                for s in doc.sentences:
                    words = [w.text for w in s.words]
                    heads = [w.head for w in s.words]
                    nw, st, rel, t = rewrite_sentence(words, heads, args.mdd_tol, tau_min)
                    stats[st] += 1
                    if st == "ok":
                        rels.append(rel); taus.append(t)
                    new_sents.append(" ".join(nw))
                holder[key] = " ".join(new_sents)
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (ln + 1) % 500 == 0:
                print(f"[{ln+1}] {stats}", flush=True)
    tot = sum(stats.values())
    print(f"[DONE] {stats} accept={stats['ok']/max(tot,1)*100:.1f}% "
          f"mean|dMDD|={sum(rels)/max(len(rels),1)*100:.2f}% mean_tau={sum(taus)/max(len(taus),1):.3f}")

if __name__ == "__main__":
    main()
