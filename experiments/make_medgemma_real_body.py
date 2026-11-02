# Builds body_medgemma_real.py from body_medgemma.py (run make_medgemma_body.py first): the REAL MedGemma-27B run for tissue + diagnosis, in the recipe the debug session proved:
# fresh HF download, bf16, 4-bit nf4, one-image-at-a-time vision patch (needed for 3+ image questions, outputs identical to unpatched), NO expandable_segments, max_memory 13 GiB per GPU.
# Guards: stop after 3 failures (as before) AND after 3 EMPTY answers (the earlier empty-answer runs must not waste hours); error tracebacks are printed. Raw answers are saved (<task>_raw.csv) and cleaned offline.
src = open("body_medgemma.py", encoding="utf-8").read()
def rep(a, b):
    global src
    assert src.count(a) == 1, (src.count(a), a[:70]); src = src.replace(a, b)
b = 'DEV_IN = getattr(model, "device", "cuda")\n'
rep(b, b + r'''_orig_gif = model.model.get_image_features
def _gif_chunked(pixel_values, **kw):                                           # one image at a time through the vision tower (all images at once ran out of memory with 3+ images; outputs identical)
    outs = [_orig_gif(pixel_values[i:i + 1], **kw) for i in range(pixel_values.shape[0])]
    res = outs[0]; res.pooler_output = torch.cat([o.pooler_output for o in outs], 0); return res
model.model.get_image_features = _gif_chunked; print("vision tower: one image at a time", flush=True)
''')
e = "raw = \"\"; errs += 1; print(f\"  ERROR {r['question-id']}: {type(e).__name__} {str(e)[:150]}\", flush=True)"
rep(e, "raw = \"\"; errs += 1; print(f\"  ERROR {r['question-id']}: {type(e).__name__} {str(e)[:150]}\", flush=True); import traceback, gc; print('   ' + ' | '.join(traceback.format_exc().strip().split(chr(10))[-8:])[:900], flush=True); gc.collect(); torch.cuda.empty_cache()")
a = '        rows.append((r["question-id"], postprocess(task, raw, str(r["question"])), raw))\n'
rep(a, a + '        if k == 3 and all(not str(x[2]).strip() for x in rows[:3]): print("the first 3 answers are all EMPTY - stopping instead of spending hours on empty answers", flush=True); os._exit(9)\n')
open("body_medgemma_real.py", "w", encoding="utf-8").write(src); print("body_medgemma_real.py written", len(src), "chars")
