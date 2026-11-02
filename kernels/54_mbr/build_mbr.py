import json, pandas as pd, os, sys
base = "../../"
cands = [("hybdiag_0734", "out_hybdiag/diagnosis.csv"), ("diagfree_clean_0719", "out_diagfree_clean/diagnosis.csv"), ("instructed_t560a_0713", "out_t560a/submission/diagnosis.csv"), ("gemma12b_0664", "out/gemma4_12b/diagnosis.csv")]
d = {}
for n, p in cands:
    x = pd.read_csv(base + p).fillna(""); x = x[x["question-id"].str.contains("_diagnosis")]; assert len(x) == 300, (n, len(x))
    d[n] = {q: " ".join(str(a).split()) for q, a in zip(x["question-id"], x["response"])}
src = open("mbr_template.py", encoding="utf-8").read().replace("__CANDS__", json.dumps(d, ensure_ascii=True))
open("mbr.py", "w", encoding="utf-8").write(src); print("mbr.py written:", len(src), "chars,", len(d), "candidate runs x 300 answers")
json.dump({"id": "YOUR_KAGGLE_USER/dalphin-54-diag-mbr", "title": "dalphin-54-diag-mbr", "code_file": "mbr.py", "language": "python", "kernel_type": "script", "is_private": True,
           "enable_gpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, open("kernel-metadata.json", "w"), indent=1)
