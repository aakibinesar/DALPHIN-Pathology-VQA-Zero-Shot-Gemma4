import json, glob, sys
import pandas as pd, fitz
sys.path.insert(0, ".")
import dalphin_pipeline as dp

for e in json.load(open(glob.glob("out_hyba/*.log")[0], encoding="utf-8")):
    x = e["data"].strip()
    if e["time"] < 400 or ("Warning" in x[:150] and "Traceback" not in x): continue
    if "it/s]" in x or "s/it]" in x or "NbConvert" in x or "re.sub" in x or x.startswith("text ="): continue
    if x.startswith("File ") and "site-packages" in x: continue
    if "[cases]" in x and "150/150" not in x: continue
    print(f"{e['time']:6.0f} {x[:200]}")

meta = pd.read_csv("dalphin_metadata.csv")
for t, old in (("behavior", "out_t560a/submission/behavior.csv"), ("mc", "out_t560a/submission/mc.csv")):
    a = pd.read_csv(f"out_hyba/submission/{t}.csv"); b = pd.read_csv(f"out_hybb/submission/{t}.csv")
    assert not set(a["question-id"]) & set(b["question-id"]), "halves overlap"
    m = pd.concat([a, b])
    exp = meta[meta["question-id"].str.match(rf"^case\d+_{t}\d*$")]
    m = m.set_index("question-id").loc[exp["question-id"]].reset_index()
    errs = dp.validate(t, m.assign(raw=m.response), exp)
    m.to_csv(f"out_hybrid/{t}.csv", index=False)
    o = pd.read_csv(old); j = m.merge(o, on="question-id", suffixes=("_h", "_i")); ag = (j.response_h.astype(str) == j.response_i.astype(str)).mean()
    print(f"{t}: first half {len(a)} + second half {len(b)} = merged {len(m)} (expected {len(exp)}) | validator {errs} | blank {int(m.response.isna().sum())} | agree with independent {ag:.3f}")
    print(f"   hybrid {m.response.value_counts().to_dict()} | independent {o.response.value_counts().to_dict()}")

def pdf(out, title, text):
    story = fitz.Story(html=f"<h2>{title}</h2><p>{text}</p>", user_css="body{font-family:sans-serif;font-size:11pt;line-height:1.35;} h2{font-size:13pt;margin:0 0 6pt 0;}")
    w = fitz.DocumentWriter(out); mb = fitz.paper_rect("a4"); more = 1
    while more:
        dev = w.begin_page(mb); more, _ = story.place(mb + (60, 60, -60, -60)); story.draw(dev); w.end_page()
    w.close(); print(out, len(fitz.open(out)), "page")

pdf("DALPHIN_method_gemma31b_hybrid_short.pdf", "DALPHIN behaviour / multiple choice: Gemma-4-31B-it, zero-shot, contextual",
    "Predictions were generated zero-shot (no training or fine-tuning) with Gemma-4-31B-it (Google, via Kaggle Models), loaded in 4-bit (NF4, bitsandbytes), on the public DALPHIN images and prompts, using 560 image tokens per image. "
    "Questions of a case were asked in order and the model's earlier answers were given as text context; the earlier answers for the neoplasm and diagnosis questions were taken from our independent runs, and the behaviour answer (for multiple choice) was generated in the same run.")
