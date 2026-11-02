# Builds body_medgemma_debug.py from body_medgemma.py (run make_medgemma_body.py first): MedGemma-27B DEBUG session - one fresh HF download + 4-bit bf16 load (the setting that gave real answers), NO expandable_segments, then a few diagnostic generations
# with and without the one-image-at-a-time vision patch, printing whether the logits are finite, the top tokens, the generated ids and the text (with and without special tokens), plus a direct comparison of the vision features.
src = open("body_medgemma.py", encoding="utf-8").read()
i1 = src.index("report = {}\n")
dbg = r'''
import gc
def prep(row):
    parts = build_messages(row, IMG)[0]["content"]; ims, content = [], []
    for p in parts:
        if p["type"] == "image": ims.append(load_img(p["path"])); content.append({"type": "image"})
        else: content.append(p)
    content = [c for c in content if c["type"] == "image"] + [c for c in content if c["type"] != "image"]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    return proc(text=[text], images=[ims], return_tensors="pt", add_special_tokens=False).to(DEV_IN), len(ims)

_orig_gif = model.model.get_image_features
def _gif_chunked(pixel_values, **kw):                                           # one image at a time through the vision tower
    outs = [_orig_gif(pixel_values[i:i + 1], **kw) for i in range(pixel_values.shape[0])]
    res = outs[0]; res.pooler_output = torch.cat([o.pooler_output for o in outs], 0); return res
def set_patch(on): model.model.get_image_features = _gif_chunked if on else _orig_gif

RES = []
def diag(tag, qid, patch, max_new=20):
    row = meta[meta["question-id"] == qid].iloc[0]
    try:
        set_patch(patch); inp, n_img = prep(row); t_a = time.time(); cur["task"] = task_of(qid)
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float(); fin = bool(torch.isfinite(lg).all()); top = torch.topk(torch.nan_to_num(lg, nan=-1e9), 5).indices.tolist()
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        new = out[0, inp["input_ids"].shape[1]:]; txt = proc.decode(new, skip_special_tokens=True)
        print(f"[{tag}] {qid} images {n_img} patch={patch} | prompt tokens {inp['input_ids'].shape[1]} | last logits finite {fin} | top5 {proc.tokenizer.convert_ids_to_tokens(top)} | new ids {new[:10].tolist()} | text {txt[:110]!r} | with special {proc.decode(new, skip_special_tokens=False)[:90]!r} | {time.time() - t_a:.0f}s | peak {peak_gb():.1f} GB", flush=True)
        RES.append((tag, qid, patch, n_img, fin, bool(txt.strip())))
    except Exception as e:
        import traceback; print(f"[{tag}] {qid} patch={patch} ERROR {type(e).__name__}: {str(e)[:200]} | " + " | ".join(traceback.format_exc().strip().split(chr(10))[-6:])[:700], flush=True); RES.append((tag, qid, patch, -1, None, False))
    gc.collect(); torch.cuda.empty_cache()

nimg = {q: len(split_images(r)) for q, r in zip(meta["question-id"], meta.to_dict("records")) if q.endswith("_tissue")}
q7 = max(nimg, key=nimg.get); q4 = next(q for q, n in nimg.items() if n == 4); print("tissue questions with most images:", q7, nimg[q7], "| a 4-image one:", q4, flush=True)
set_patch(False); inp, _ = prep(meta[meta["question-id"] == "case001_tissue"].iloc[0])
with torch.no_grad():
    set_patch(False); a = model.model.get_image_features(inp["pixel_values"], return_dict=True).pooler_output
    set_patch(True); b = model.model.get_image_features(inp["pixel_values"], return_dict=True).pooler_output
print(f"VISION FEATURES case001 (3 images) unchunked vs one-at-a-time: shape {tuple(a.shape)} | max abs diff {(a.float() - b.float()).abs().max():.4f} | mean abs {a.float().abs().mean():.4f} | finite {bool(torch.isfinite(a).all())} {bool(torch.isfinite(b).all())}", flush=True)
del a, b, inp; gc.collect(); torch.cuda.empty_cache()
diag("D0 no patch", "case001_tissue", False)
diag("D1 patch", "case001_tissue", True)
diag("D2 patch", "case002_tissue", True)
diag("D3 patch, most images", q7, True)
diag("D4 no patch, 4 images (expected OOM)", q4, False)
print("DEBUG SUMMARY:", [(t, q, p, n, f, ok) for t, q, p, n, f, ok in RES], flush=True)
print(f"DONE total {time.time() - t00:.0f}s", flush=True)
'''
out = src[:i1] + dbg
open("body_medgemma_debug.py", "w", encoding="utf-8").write(out); print("body_medgemma_debug.py written", len(out), "chars")
