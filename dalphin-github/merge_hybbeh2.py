import json, glob, sys
import pandas as pd, fitz
sys.path.insert(0, ".")
import dalphin_pipeline as dp
for d in ("out_hybbeh2_a", "out_hybbeh2_b"):
    for e in json.load(open(glob.glob(d + "/*.log")[0], encoding="utf-8")):
        x = e["data"].strip()
        if ("[cases]" in x and "150/150" in x) or "model loaded" in x or "DONE total" in x or "ERROR" in x: print(d[-1], x[:170])
a = pd.read_csv("out_hybbeh2_a/submission/behavior.csv"); b = pd.read_csv("out_hybbeh2_b/submission/behavior.csv")
assert not set(a["question-id"]) & set(b["question-id"]); m = pd.concat([a, b])
meta = pd.read_csv("dalphin_metadata.csv"); exp = meta[meta["question-id"].str.match(r"^case\d+_behavior\d*$")]
m = m.set_index("question-id").loc[exp["question-id"]].reset_index()
errs = dp.validate("behavior", m.assign(raw=m.response), exp); m.to_csv("out_hybbeh2/behavior.csv", index=False)
v1 = pd.read_csv("out_hybrid/behavior.csv"); ind = pd.read_csv("out_t560a/submission/behavior.csv")
j = m.merge(v1, on="question-id", suffixes=("_n", "_v1")).merge(ind.rename(columns={"response": "response_ind"}), on="question-id")
print(f"rows {len(m)} (expected {len(exp)}) | validator {errs} | blank {int(m.response.isna().sum())}")
print(f"agree with the 0.663 contextual run: {(j.response_n == j.response_v1).mean():.3f} | with the independent 0.560 run: {(j.response_n == j.response_ind).mean():.3f}")
print("new:", m.response.value_counts().to_dict(), "| 0.663 run:", v1.response.value_counts().to_dict(), "| independent:", ind.response.value_counts().to_dict())
story = fitz.Story(html="<h2>DALPHIN behaviour: Gemma-4-31B-it, zero-shot, contextual</h2><p>Predictions were generated zero-shot (no training or fine-tuning) with Gemma-4-31B-it (Google, via Kaggle Models), loaded in 4-bit (NF4, bitsandbytes), on the public DALPHIN images and prompts, using 560 image tokens per image. The model also received the neoplasm and diagnosis questions with their answers (taken from earlier runs) as text context.</p>", user_css="body{font-family:sans-serif;font-size:11pt;line-height:1.35;} h2{font-size:13pt;margin:0 0 6pt 0;}")
w = fitz.DocumentWriter("DALPHIN_method_hybbeh2_short.pdf"); mb = fitz.paper_rect("a4"); more = 1
while more:
    dev = w.begin_page(mb); more, _ = story.place(mb + (60, 60, -60, -60)); story.draw(dev); w.end_page()
w.close(); print("pdf pages", len(fitz.open("DALPHIN_method_hybbeh2_short.pdf")))
