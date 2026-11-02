# Builds body_think.py from body.py (body.py itself is not touched): Gemma-4 THINKING mode timing/pilot body.
# Changes: enable_thinking=True in the chat template; the original text-first order is kept (images first only with IMAGES_FIRST_ARG = 1: that test showed no benefit); sampling with the model card's
# recommended parameters (temperature 1.0, top_p 0.95, top_k 64; fixed seed); a thinking budget THINK_MAX_ARG (new tokens); if the thought channel is still open at the budget it is closed by hand ("<channel|>")
# and the answer is generated greedily; per-question token counts / seconds / forced flag are printed and saved (think_log.csv) so the cost of a full run can be projected.
src = open("body.py", encoding="utf-8").read()
i0 = src.index("def generate(messages):"); i1 = src.index("report = {}\n")
new = r'''THINK_MAX = int(globals().get("THINK_MAX_ARG", 1024)); THINK_LOG = []
torch.manual_seed(0)


def split_thinking(s):
    """s = decoded new tokens WITH special tokens. Returns (thought, answer, closed). Format (Gemma-4): <|channel>thought\n[reasoning]<channel|>[answer]<turn|>"""
    i = s.find("<|channel>thought"); j = s.find("<channel|>", max(i, 0))
    if i < 0 and j < 0: closed, thought, ans = True, "", s
    elif j < 0: return s[i + len("<|channel>thought"):], "", False
    else: thought = s[(i + len("<|channel>thought")) if i >= 0 else 0:j]; ans = s[j + len("<channel|>"):]; closed = True
    ans = ans.split("<turn|>")[0]
    return thought, re.sub(r"<[^<>]{1,20}>", " ", ans).strip(), closed


def generate(messages):
    parts = messages[0]["content"]; ims, content = [], []
    for p in parts:
        if p["type"] == "image": ims.append(load_img(p["path"])); content.append({"type": "image"})
        else: content.append(p)
    if globals().get("IMAGES_FIRST_ARG"): content = [c for c in content if c["type"] == "image"] + [c for c in content if c["type"] != "image"]          # images before the text only when asked (the images-first test scored 0.515 vs 0.560 text-first: no benefit)
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    imgs = [ims] if ims else None
    inp = proc(text=[text], images=imgs, return_tensors="pt").to(DEV_IN); n_in = inp["input_ids"].shape[1]; t_a = time.time()
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=THINK_MAX + MAXTOK[cur["task"]] + 8, do_sample=True, temperature=1.0, top_p=0.95, top_k=64)
    n_gen = int(out.shape[1] - n_in); dec = proc.batch_decode(out[:, n_in:], skip_special_tokens=False)[0]
    thought, ans, closed = split_thinking(dec); forced = False
    if not closed:                                                                # budget reached inside the thought: close the channel by hand and let the model answer (greedy)
        forced = True; text2 = text + dec.split("<turn|>")[0].rstrip() + "<channel|>"
        inp2 = proc(text=[text2], images=imgs, return_tensors="pt").to(DEV_IN)
        with torch.no_grad(): out2 = model.generate(**inp2, max_new_tokens=MAXTOK[cur["task"]] + 8, do_sample=False)
        ans = proc.batch_decode(out2[:, inp2["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip(); thought = dec
    secs = time.time() - t_a; THINK_LOG.append(dict(n_new_tokens=n_gen, secs=round(secs, 1), forced=forced, thought_chars=len(thought), answer=ans[:80], thought=thought[:1500].replace("\n", " ")))
    print(f"   think: {n_gen} new tokens, {secs:.0f}s ({n_gen / max(secs, 1e-6):.1f} tok/s), thought chars {len(thought)}, forced stop {forced}, answer {ans[:50]!r}", flush=True)
    if len(THINK_LOG) == 2:                                                       # cost guards: stop after 2 questions if thinking did not switch on, or if it is too expensive at this budget
        d2 = pd.DataFrame(THINK_LOG)
        why = "NO THINKING DETECTED in the first 2 questions (enable_thinking did not switch thinking on)" if (d2.thought_chars == 0).all() else ("TOO EXPENSIVE: mean %.0f s per question for the first 2 (limit 200 s)" % d2.secs.mean() if d2.secs.mean() > 200 else "")
        if why:
            d2.to_csv(f"{WORK}/think_log.csv", index=False); print("STOPPED EARLY:", why, "| think log saved", flush=True); sys.stdout.flush(); os._exit(0)
    return ans


'''
tail = '''
if THINK_LOG:
    import statistics as _st
    d = pd.DataFrame(THINK_LOG); d.to_csv(f"{WORK}/think_log.csv", index=False)
    print(f"THINKING SUMMARY: {len(d)} questions | new tokens mean {d.n_new_tokens.mean():.0f} (min {d.n_new_tokens.min()}, max {d.n_new_tokens.max()}) | s/question mean {d.secs.mean():.0f} | forced stops {int(d.forced.sum())}/{len(d)} | projected 300 questions: {d.secs.mean() * 300 / 3600:.1f} h (+ load)", flush=True)
'''
out = src[:i0] + new + src[i1:]
mk = 'print("VALIDATION REPORT", json.dumps(report), flush=True)'; assert out.count(mk) == 1; out = out.replace(mk, tail.strip("\n") + "\n" + mk)
if "import re" not in out.split("def generate")[0]: out = out.replace("import os, sys, time, json, glob\n", "import os, sys, time, json, glob, re\n", 1)
open("body_think.py", "w", encoding="utf-8").write(out); print("body_think.py written", len(out), "chars")
