# Builds body_hybrid.py (the HYBRID contextual run: some tasks are generated in a per-case chain, the earlier tasks' answers are taken from files (HIST_ANS) of our independent runs).
# Same model loading and image handling as body.py; earlier questions + answers of a case are given as TEXT-ONLY history (no images in earlier turns).
import sys
src = open("body.py", encoding="utf-8").read()
head, rest = src.split("report = {}\n", 1)

# 1. early check of the multi-turn chat template (before the slow model load): choose the assistant-turn format that renders, or stop with a clear message
a = "assert proc is not None, \"no processor could be loaded\"\n"
assert head.count(a) == 1
check = a + '''
ASST_LIST = True
def _asst(t): return {"role": "assistant", "content": ([{"type": "text", "text": t}] if ASST_LIST else t)}
def _render_ok():
    test = [{"role": "user", "content": [{"type": "text", "text": "QQQ1"}]}, _asst("AAA1"), {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "QQQ2"}]}]
    t = proc.apply_chat_template(test, tokenize=False, add_generation_prompt=True)
    assert "QQQ1" in t and "AAA1" in t and "QQQ2" in t, "history missing from rendered prompt"
    return t
try: _t = _render_ok()
except Exception as e:
    print("assistant turn as list failed:", repr(e)[:200], flush=True); ASST_LIST = False
    try: _t = _render_ok()
    except Exception as e2: import traceback; traceback.print_exc(); print("CHAT TEMPLATE CHECK FAILED:", repr(e2)[:300], flush=True); os._exit(4)
print("chat template OK (assistant content as", "list" if ASST_LIST else "string", ") sample:", _t[-400:].replace("\\n", " | "), flush=True)
'''
head = head.replace(a, check)

# 2. multi-turn generate
g0 = head.index("def generate(messages):")
g1 = head.index("\n\n\n", g0) if "\n\n\n" in head[g0:] else len(head)
new_gen = '''def generate(messages):
    ims, msgs = [], []
    for m in messages:
        content = []
        for p in m["content"]:
            if p["type"] == "image": ims.append(load_img(p["path"])); content.append({"type": "image"})
            else: content.append(p)
        msgs.append({"role": m["role"], "content": content})
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True); LAST["text"] = text
    inp = proc(text=[text], images=([ims] if (KIND in ("gemma4", "gemma31", "gemma31b") and ims) else (ims or None)), return_tensors="pt").to(DEV_IN)
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=MAXTOK[cur["task"]], do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
'''
head = head[:g0] + "LAST = {}\n" + new_gen + head[g1:]

loop = '''# ==SEQ-BEGIN==
INTRO = "The following question(s) and image(s) are related to the same medical case.\\n"        # as in the organisers' feedback-mode script


def run_hybrid(meta, order, generate_fn, asst_fn, task_of_fn, build_fn, post_fn, img_dir, hist_ans, gen_tasks, suffix, log=print, save=None, max_hist_chars=300, lookahead=None):
    """HYBRID contextual run, per case in file order. Tasks in gen_tasks are generated with the model; every other task contributes its answer from hist_ans
    (taken from our independent runs). All earlier questions + answers of the case go in as text-only turns (no images in earlier turns).
    lookahead {task: [other tasks]} (default none): when generating `task`, the question + answer (from hist_ans) of those other tasks of the same case are put in front as history even if they come later in the file."""
    lookahead = lookahead or {}; grp = meta.groupby("case-id", sort=False); out = {}; errs = 0; ngen = 0; t2 = time.time()
    for ci, cid in enumerate(order, 1):
        hist = []; first = True
        for _, r in grp.get_group(cid).iterrows():
            qid = r["question-id"]; task = task_of_fn(qid); cur["task"] = task
            q_only = (INTRO if first else "") + str(r["question"]) + suffix.get(task, "")           # history version: no preamble (it describes images that are not repeated)
            if task in gen_tasks:
                parts = build_fn(r, img_dir)[0]["content"]; text, imgs = parts[0]["text"], parts[1:]
                if first: text = INTRO + text
                msgs = []; pre = []
                for lt in lookahead.get(task, []):
                    lr = next((x for _, x in grp.get_group(cid).iterrows() if task_of_fn(x["question-id"]) == lt), None)
                    if lr is not None and hist_ans.get(lr["question-id"]) is not None: pre.append((str(lr["question"]) + suffix.get(lt, ""), str(hist_ans[lr["question-id"]])[:max_hist_chars]))
                if pre and first: pre[0] = (INTRO + pre[0][0], pre[0][1]); text = text[len(INTRO):]          # the intro line goes on the first turn of the conversation
                for hq, ha in pre + hist: msgs +=[{"role": "user", "content": [{"type": "text", "text": hq}]}, asst_fn(ha)]
                msgs.append({"role": "user", "content": [{"type": "text", "text": text}] + imgs})
                ngen += 1
                try: raw = generate_fn(msgs)
                except Exception as e: raw = ""; errs += 1; log(f"  ERROR {qid}: {type(e).__name__} {str(e)[:150]}")
                if errs >= 3 and errs == ngen: log("the first 3 generations all failed - stopping instead of writing empty answers"); os._exit(7)
                resp = post_fn(task, raw, str(r["question"]))
                out.setdefault(task, []).append((qid, resp, raw)); ans = resp or raw or "(no answer)"
            else:
                ans = hist_ans.get(qid)
                if ans is None: continue                                   # no earlier answer for this question (e.g. tissue: the organisers' preamble already names the organ in every later question): leave the turn out
            hist.append((q_only, str(ans)[:max_hist_chars])); first = False
        if ci % 10 == 0 or ci == len(order):
            log(f"  [cases] {ci}/{len(order)} | {time.time() - t2:.0f}s | {(time.time() - t2) / ci:.0f} s/case | generated {ngen} | errors {errs} | peak {peak_gb():.1f} GB")
            if save: save(out)
    return out, errs
# ==SEQ-END==


def _save(out, final=False):
    for task, rows in out.items():
        df = pd.DataFrame(rows, columns=["question-id", "response", "raw"])
        if final: df.to_csv(f"{WORK}/{task}_raw.csv", index=False); df[["question-id", "response"]].to_csv(f"{OUTD}/{task}.csv", index=False)
        else: df.to_csv(f"{WORK}/{task}_partial.csv", index=False)


HIST = globals().get("HIST_ANS", {}); GEN = set(ONLY_TASKS)
assert HIST and GEN, "hybrid run needs HIST_ANS (earlier answers) and ONLY_TASKS (tasks to generate)"
order = list(dict.fromkeys(meta["case-id"]))
if CASE_RANGE[1]: order = order[CASE_RANGE[0]:CASE_RANGE[1]]
if LIMIT: order = order[:LIMIT]
print(f"hybrid run: {len(order)} cases ({order[0]} .. {order[-1]}) | generated tasks {sorted(GEN)} | earlier answers from files for {len(HIST)} questions | text-only history | lookahead {globals().get('LOOKAHEAD_ARG')} | KIND {KIND}", flush=True)
out, errs = run_hybrid(meta, order, generate, _asst, task_of, build_messages, postprocess, IMG, HIST, GEN, SUFFIX, log=lambda m: print(m, flush=True), save=_save, lookahead=globals().get("LOOKAHEAD_ARG"))
_save(out, final=True)
report = {}
for task, rows in out.items():
    pred = pd.DataFrame(rows, columns=["question-id", "response", "raw"]); full = meta[meta["question-id"].str.match(rf"^case\\d+_{task}\\d*$")]
    report[task] = validate(task, pred, full) if (not LIMIT and not CASE_RANGE[1]) else "skipped (partial range or limit)"
    print(f"[{task}] {len(pred)} answers | validation: {report[task]}", flush=True)
    print("   sample:", [(a, b[:60]) for a, b in zip(pred['question-id'][:3], pred['response'][:3])], flush=True)
print("last prompt text sample:", LAST.get("text", "")[-900:].replace("\\n", " | "), flush=True)
print("VALIDATION REPORT", json.dumps(report), flush=True)
print(f"DONE total {time.time() - t00:.0f}s", flush=True)
'''
open("body_hybrid.py", "w", encoding="utf-8").write(head + loop)
print("body_hybrid.py written", len(head + loop), "chars")
