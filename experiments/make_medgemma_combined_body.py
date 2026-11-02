# Builds body_medgemma_combined.py from body_medgemma.py (run make_medgemma_body.py first): HF download + 4-bit load + SAVE of the 4-bit copy right after the load, then the fixed test:
# images go through the vision tower one at a time, error tracebacks are printed, expandable segments, and a failure of the first questions does NOT stop the run with an error code
# (the run must finish cleanly so that its output, the 4-bit copy, can be attached as a source by later notebooks).
src = open("body_medgemma.py", encoding="utf-8").read()
def rep(a, b):
    global src
    assert src.count(a) == 1, (src.count(a), a[:70]); src = src.replace(a, b)
rep("import torch, transformers\n", 'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"\nimport torch, transformers\n')
rep('max_memory={k: "13GiB" for k in range(torch.cuda.device_count())}', 'max_memory={0: "10GiB", 1: "13GiB"}')
b = 'DEV_IN = getattr(model, "device", "cuda")\n'
rep(b, b + r'''_orig_gif = model.model.get_image_features
def _gif_chunked(pixel_values, **kw):                                           # one image at a time through the vision tower (all images at once needed ~1 GiB each)
    outs = [_orig_gif(pixel_values[i:i + 1], **kw) for i in range(pixel_values.shape[0])]
    res = outs[0]; res.pooler_output = torch.cat([o.pooler_output for o in outs], 0); return res
model.model.get_image_features = _gif_chunked; print("vision tower: one image at a time", flush=True)
''')
e = "raw = \"\"; errs += 1; print(f\"  ERROR {r['question-id']}: {type(e).__name__} {str(e)[:150]}\", flush=True)"
rep(e, "raw = \"\"; errs += 1; print(f\"  ERROR {r['question-id']}: {type(e).__name__} {str(e)[:150]}\", flush=True); import traceback, gc; print('   ' + ' | '.join(traceback.format_exc().strip().split(chr(10))[-10:])[:1200], flush=True); gc.collect(); torch.cuda.empty_cache()")
rep('print("the first 3 questions all failed - stopping instead of writing empty answers", flush=True); os._exit(7)', 'print("the first 3 questions all failed - skipping the rest of this task (the run still finishes cleanly)", flush=True); break')
open("body_medgemma_combined.py", "w", encoding="utf-8").write(src); print("body_medgemma_combined.py written", len(src), "chars")
