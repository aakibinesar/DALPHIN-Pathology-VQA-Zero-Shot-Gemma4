# CPU notebook: minimum-risk selection of the diagnosis answer. For every question we have several candidate answers (earlier runs). The DALPHIN diagnosis score is the cosine
# similarity of BioBERT sentence embeddings (pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb, texts lower-cased) between answer and reference, so we pick, per question,
# the candidate with the highest mean similarity to the OTHER candidates (the most "central" answer). Ties go to the earlier candidate (priority order = best run first).
import os, sys, json, time, subprocess, collections
import numpy as np
T0 = time.time()
def stamp(m): print(f"[t+{time.time() - T0:5.0f}s] {m}", flush=True)
CANDS = __CANDS__          # {run name: {question-id: answer}}, in PRIORITY order
FAKE = os.environ.get("MBR_FAKE") == "1"          # local logic test only: hashed character n-gram vectors instead of the BioBERT model
names = list(CANDS); ids = sorted(CANDS[names[0]]); stamp(f"{len(names)} candidate runs {names} | {len(ids)} questions")
assert all(set(CANDS[n]) == set(ids) for n in names)

if FAKE:
    from collections import Counter
    def embed(texts):
        out = np.zeros((len(texts), 4096), dtype=np.float32)
        for i, t in enumerate(texts):
            t = f" {t.lower()} "
            for g in (t[k:k + 3] for k in range(len(t) - 2)): out[i, hash(g) % 4096] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
else:
    try: from sentence_transformers import SentenceTransformer
    except Exception:
        stamp("installing sentence-transformers"); print(subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], capture_output=True, text=True).stdout[-300:], flush=True)
        from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", device="cpu"); stamp("BioBERT sentence model loaded")
    def embed(texts):
        e = model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)

flat = [str(CANDS[n][q]).strip().lower() for q in ids for n in names]
E = embed(flat).reshape(len(ids), len(names), -1); stamp(f"embedded {len(flat)} texts")
rows, chosen = [], collections.Counter(); changed = 0; sims = []
for qi, q in enumerate(ids):
    S = E[qi] @ E[qi].T                                   # cosine similarity matrix between the candidates of this question
    score = [(S[i].sum() - S[i, i]) / (len(names) - 1) for i in range(len(names))]
    best = max(range(len(names)), key=lambda i: (round(score[i], 6), -i))        # highest mean similarity to the others; ties -> earlier candidate
    chosen[names[best]] += 1; changed += int(best != 0); sims.append(float(np.mean(score)))
    rows.append((q, str(CANDS[names[best]][q]).strip()))
os.makedirs("/kaggle/working", exist_ok=True)
import csv
with open("/kaggle/working/diagnosis_mbr.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["question-id", "response"]); w.writerows(rows)
print("chosen candidate counts:", dict(chosen), "| answers different from the first candidate:", changed, f"({changed / len(ids):.1%}) | mean agreement among candidates {np.mean(sims):.3f}", flush=True)
pair = {f"{names[i]} vs {names[j]}": round(float(np.mean([E[q, i] @ E[q, j] for q in range(len(ids))])), 3) for i in range(len(names)) for j in range(i + 1, len(names))}
print("mean cosine similarity between runs:", json.dumps(pair), flush=True)
stamp("DONE"); os._exit(0)
