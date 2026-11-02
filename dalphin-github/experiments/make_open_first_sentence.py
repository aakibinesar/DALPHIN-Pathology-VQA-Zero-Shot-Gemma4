"""Open-answer variant "first sentence": post-processing of the EXISTING Gemma-4-31B (budget 560) open answers, no new model run.
Rule (fixed before looking at any score of it): for questions that start with "Name", "What is the" or "What are the", keep only the first sentence of the answer;
all other answers are left untouched. Sentences are split at '.', '!' or '?' followed by whitespace and a capital letter, except after common abbreviations;
if the first sentence is shorter than 15 characters the original answer is kept."""
import re, sys
import pandas as pd
sys.path.insert(0, ".")
import dalphin_pipeline as dp

ABBR = {"e.g", "i.e", "vs", "dr", "no", "fig", "approx", "cf", "etc", "mm", "cm", "ca", "st", "spp"}
RULE = re.compile(r"(?i)^\s*(name\b|what is the\b|what are the\b)")

def first_sentence(text):
    t = " ".join(str(text).split())
    for m in re.finditer(r"([.!?])\s+(?=[A-Z])", t):
        head = t[:m.start()]
        last = re.findall(r"[A-Za-z.]+$", head)
        if last and last[0].lower().rstrip(".") in ABBR: continue
        return t[:m.end(1)]
    return t

meta = pd.read_csv("dalphin_metadata.csv"); q = meta[meta["question-id"].str.contains("_open")].set_index("question-id")["question"].astype(str)
ans = pd.read_csv("out_open/submission/open.csv")
out, changed = [], 0
for qid, a in zip(ans["question-id"], ans["response"].fillna("")):
    new = a
    if RULE.match(q[qid]):
        fs = first_sentence(a)
        if len(fs) >= 15 and fs != " ".join(a.split()): new = fs
    changed += int(new != a); out.append(new)
res = pd.DataFrame({"question-id": ans["question-id"], "response": out})
import os; os.makedirs("out_open_first", exist_ok=True); res.to_csv("out_open_first/open.csv", index=False)
sel = q.index[q.str.match(RULE)]
print(f"questions: {len(res)} | matching the rule: {len(sel)} | answers changed: {changed}")
print(f"mean length: before {ans['response'].fillna('').str.len().mean():.0f} chars -> after {res['response'].str.len().mean():.0f} | on the changed answers: "
      f"{ans.set_index('question-id').response.fillna('').loc[[i for i, a, b in zip(ans['question-id'], ans['response'].fillna(''), out) if a != b]].str.len().mean():.0f} -> "
      f"{res.set_index('question-id').response.loc[[i for i, a, b in zip(ans['question-id'], ans['response'].fillna(''), out) if a != b]].str.len().mean():.0f}")
# validate with the organisers-style validator of our pipeline
meta_open = meta[meta["question-id"].str.match(r"^case\d+_open\d*$")]
print("validator:", dp.validate("open", res.assign(raw=res.response), meta_open))
for i in (7, 25, 60):
    qid = ans["question-id"].iloc[i]
    if out[i] != ans["response"].fillna("").iloc[i]: print(f"\n{qid}\nQ: {q[qid][:140]}\nBEFORE: {ans['response'].iloc[i][:260]}\nAFTER : {out[i]}")
