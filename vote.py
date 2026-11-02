"""Majority / plurality voting over the answers of several runs (same model family, different settings or context).
usage: python vote.py <task: neoplasm|mc> <out.csv> <voter1.csv> <voter2.csv> ...   (voters in PRIORITY order: ties go to the earlier voter; each csv has question-id,response)"""
import sys, collections
import pandas as pd
sys.path.insert(0, ".")
import dalphin_pipeline as dp
task, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
origs = [pd.read_csv(f).fillna("").set_index("question-id").response.astype(str).str.strip() for f in files]
dfs = [o.str.lower().str.rstrip(".") for o in origs]            # compare case-insensitively, without a trailing full stop; the OUTPUT keeps the first matching voter's original text
ids = dfs[0].index; assert all(set(d.index) == set(ids) for d in dfs), "voters cover different questions"
res, changed = [], 0
for q in ids:
    votes = [d[q] for d in dfs]; cnt = collections.Counter(votes); top = max(cnt.values())
    winner = next(v for v in votes if cnt[v] == top)            # first voter (priority order) among the most-voted answers
    res.append(next(o[q] for d, o in zip(dfs, origs) if d[q] == winner)); changed += int(winner != votes[0])
meta = pd.read_csv("dalphin_metadata.csv"); exp = meta[meta["question-id"].str.match(rf"^case\d+_{task}\d*$")]
o = pd.DataFrame({"question-id": ids, "response": res}).set_index("question-id").loc[exp["question-id"]].reset_index()
print(f"{task}: {len(files)} voters | {len(o)} rows | validator {dp.validate(task, o.assign(raw=o.response), exp)} | answers different from voter 1 ({files[0]}): {changed} ({changed / len(o):.1%})")
for i, (f, d) in enumerate(zip(files, dfs)): print(f"   voter {i+1} agrees with the vote on {(d.loc[o['question-id']].values == o.response.str.lower().str.rstrip('.').values).mean():.3f}  [{f}]")
o.to_csv(out, index=False)
