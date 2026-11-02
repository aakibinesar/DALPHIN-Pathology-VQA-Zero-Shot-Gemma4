import sys, json
sys.path.insert(0, "../..")
import pandas as pd
from dalphin_pipeline import *
src = open("body_hybrid.py", encoding="utf-8").read()
seg = src.split("# ==SEQ-BEGIN==")[1].split("# ==SEQ-END==")[0]
cur = {"task": "x"}; peak_gb = lambda: 0.0
import time
exec(seg)
meta = pd.read_csv("../../dalphin_metadata.csv"); H = json.load(open("hist.json", encoding="utf-8"))
calls = []
def gen(msgs): calls.append(msgs); return "Some diagnosis."
asst = lambda t: {"role": "assistant", "content": [{"type": "text", "text": t}]}
SUF = dict(SUFFIX); SUF["diagnosis"] = ""
order = list(dict.fromkeys(meta["case-id"]))[:2]
out, errs = run_hybrid(meta, order, gen, asst, task_of, build_messages, postprocess, "/img", H, {"diagnosis"}, SUF, log=print)
print({k: len(v) for k, v in out.items()}, "errors", errs, "calls", len(calls))
m = calls[0]; print("messages in the diagnosis call:", len(m), [x["role"] for x in m])
for x in m[:-1]: print("   ", x["role"][:4], "|", x["content"][0]["text"][:110].replace("\n", " / "))
last = m[-1]["content"]; print("   LAST user text tail:", last[0]["text"][-120:].replace("\n", " / "), "| images:", sum(1 for p in last if p["type"] == "image"))
