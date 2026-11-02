"""Builds hist.json {question-id: answer} for the HYBRID contextual run from the answer CSVs of our independent runs (the earlier tasks of each case).
usage: python make_hist.py <out.json> <neoplasm.csv> <diagnosis.csv>      (the CSVs are the files that were submitted, columns question-id,response).
Tissue answers are NOT used: the organisers' preamble of every later question already names the organ (295/300 neoplasm, diagnosis, behaviour and MC questions), and our tissue answers are often wrong, i.e. they would contradict the preamble."""
import sys, json, pandas as pd
out, files = sys.argv[1], sys.argv[2:]
assert len(files) == 2, "give neoplasm and diagnosis CSVs"
h = {}
for f, t in zip(files, ("neoplasm", "diagnosis")):
    d = pd.read_csv(f).fillna(""); d = d[d["question-id"].str.contains(f"_{t}")]
    assert len(d) >= 270, (f, len(d)); h.update({q: " ".join(str(a).split()) for q, a in zip(d["question-id"], d["response"])})
    print(f"{t}: {len(d)} answers from {f}")
json.dump(h, open(out, "w", encoding="utf-8")); print("hist.json:", len(h), "answers")
