import json, glob, re, sys
import pandas as pd, fitz
sys.path.insert(0, ".")
import dalphin_pipeline as dp
for d in ("out_hybdiag_a", "out_hybdiag_b"):
    for e in json.load(open(glob.glob(d + "/*.log")[0], encoding="utf-8")):
        x = e["data"].strip()
        if "[cases]" in x and "150/150" in x or "model loaded" in x or "task instruction removed" in x or "DONE total" in x or "hybrid run:" in x: print(d[-1], x[:190])
a = pd.read_csv("out_hybdiag_a/submission/diagnosis.csv").fillna(""); b = pd.read_csv("out_hybdiag_b/submission/diagnosis.csv").fillna("")
assert not set(a["question-id"]) & set(b["question-id"]); m = pd.concat([a, b])
meta = pd.read_csv("dalphin_metadata.csv"); exp = meta[meta["question-id"].str.match(r"^case\d+_diagnosis\d*$")]
m = m.set_index("question-id").loc[exp["question-id"]].reset_index()
prev = pd.read_csv("out_diagfree_clean/diagnosis.csv").set_index("question-id").response.astype(str)      # the 0.719 file = fallback
PATS = [r"(?i)(?:most likely|likely)\s+(?:diagnosis|lesion|entity|diagnoses)\s*(?:is|would be|of)\s*:?\s*(?:an?\s+|the\s+)?(.+)$",
        r"(?i)\b(?:diagnosis|lesion|entity)\s+is\s+(?:most likely\s+|likely\s+)?:?\s*(?:an?\s+|the\s+)?(.+)$",
        r"(?i)\bis\s+most\s+likely\s+(?:an?\s+|the\s+)?(.+)$"]
def clean(t, q):
    t = " ".join(str(t).replace("*", "").replace("\u2014", "-").replace("\u2013", "-").replace('"', "").split())
    if len(t) <= 80: return t, False
    for p in PATS:
        mm = re.search(p, t)
        if mm: t = mm.group(1); break
    else: t = re.split(r"(?<=[.!?])\s+", t)[0]
    t = re.split(r"\.\s|,\s+as\b|,\s+which\b|,\s+with\b|;", t)[0]
    t = re.sub(r"\s*\(.*?\)", "", t).strip().rstrip(".:;, ")
    if len(t) > 60 or t.lower().startswith("based on") or len(t) < 3: return str(prev[q]), True
    return t, False
res = [clean(r, q) for q, r in zip(m["question-id"], m.response)]
raw_long = int((m.response.str.len() > 80).sum()); m["response"] = [r[0] for r in res]; fb = sum(r[1] for r in res)
errs = dp.validate("diagnosis", m.assign(raw=m.response), exp); m.to_csv("out_hybdiag/diagnosis.csv", index=False)
print(f"rows {len(m)} (expected {len(exp)}) | answers > 80 chars before cleaning: {raw_long} | fell back to the 0.719 answer: {fb} | max length now {m.response.str.len().max()} | non-ASCII {sum(1 for c in ''.join(m.response) if ord(c) > 127)} | asterisks {int(m.response.str.contains('*', regex=False).sum())} | validator {errs}")
j = m.merge(prev.reset_index(), on="question-id", suffixes=("_h", "_p")); print(f"agreement with the 0.719 file (independent, cleaned): {(j.response_h.str.lower().str.strip() == j.response_p.str.lower().str.strip()).mean():.3f}")
story = fitz.Story(html="<h2>DALPHIN diagnosis: Gemma-4-31B-it, zero-shot, contextual</h2><p>Predictions were generated zero-shot (no training or fine-tuning) with Gemma-4-31B-it (Google, via Kaggle Models), loaded in 4-bit (NF4, bitsandbytes), on the public DALPHIN images and prompts, using 560 image tokens per image and the organisers' prompt only. The model also received the neoplasm question and its answer (taken from our independent run) as text context. Answers longer than 80 characters were reduced to the stated diagnosis name by a fixed rule (formatting and parenthetical text removed; the answer of our previous run was used where no name was stated).</p>", user_css="body{font-family:sans-serif;font-size:11pt;line-height:1.35;} h2{font-size:13pt;margin:0 0 6pt 0;}")
w = fitz.DocumentWriter("DALPHIN_method_gemma31b_hybdiag_short.pdf"); mb = fitz.paper_rect("a4"); more = 1
while more:
    dev = w.begin_page(mb); more, _ = story.place(mb + (60, 60, -60, -60)); story.draw(dev); w.end_page()
w.close(); print("pdf pages", len(fitz.open("DALPHIN_method_gemma31b_hybdiag_short.pdf")))
