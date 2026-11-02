# Builds body_medgemma_local2.py from body.py (body.py itself is not touched): MedGemma-27B loaded from the 4-bit copy saved by dalphin-52-medgemma27-smoke (attached as a notebook source; no Hugging Face token needed).
# Fixes over the first smoke test: images go through the vision tower ONE AT A TIME (3-4 images at once needed ~1 GiB each and ran out of memory), the chat template falls back to the tokenizer's, a single <bos> is used,
# the bitsandbytes compute dtype is forced to the run dtype, error tracebacks are printed, expandable segments reduce fragmentation.
src = open("body.py", encoding="utf-8").read()
def rep(a, b, cnt=1):
    global src
    assert src.count(a) == cnt, (src.count(a), a[:70]); src = src.replace(a, b)
a = "assert proc is not None, \"no processor could be loaded\"\n"
rep(a, a + r'''if not getattr(proc, "chat_template", None) and getattr(proc.tokenizer, "chat_template", None): proc.chat_template = proc.tokenizer.chat_template; print("processor chat template taken from the tokenizer", flush=True)
try:
    _names = split_images(meta.iloc[0])[:3]; _ims = [Image.open(f"{IMG}/{n}").convert("RGB") for n in _names]
    _t = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}] * len(_ims) + [{"type": "text", "text": "QQQ"}]}], tokenize=False, add_generation_prompt=True)
    _ids = proc(text=[_t], images=[_ims], return_tensors="pt", add_special_tokens=False)["input_ids"][0]; _n = int(_ids.shape[0])
    print(f"prompt check: {_n} tokens for 3 images of case 1 | first ids {_ids[:4].tolist()} (bos id {proc.tokenizer.bos_token_id}) | template head: {_t[:60]!r}", flush=True)
    assert "QQQ" in _t and _n > 3 * 200 and _t.startswith("<bos>") and int(_ids[0]) == proc.tokenizer.bos_token_id and int(_ids[1]) != proc.tokenizer.bos_token_id, "prompt check failed"
except Exception as e:
    import traceback; traceback.print_exc(); print("PROMPT CHECK FAILED - stopping before the model load:", repr(e)[:300], flush=True); os._exit(6)
''')
rep('kw.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DT, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)); print("4-bit nf4 bitsandbytes on load", flush=True)',
    'print("pre-quantised 4-bit checkpoint: its own quantization_config is used (nf4)", flush=True)')
rep('max_memory={k: "13GiB" for k in range(torch.cuda.device_count())}', 'max_memory={0: "10GiB", 1: "13GiB"}')
b = 'DEV_IN = getattr(model, "device", "cuda")\n'
rep(b, b + r'''try:
    import bitsandbytes as bnb
    _n4 = 0
    for _m in model.modules():
        if isinstance(_m, bnb.nn.Linear4bit):
            _m.compute_dtype = DT; _qs = getattr(_m.weight, "quant_state", None)
            if _qs is not None: _qs.dtype = DT
            _n4 += 1
    print(f"bitsandbytes compute dtype set to {DTYPE} on {_n4} 4-bit layers", flush=True)
except Exception as e: print("compute dtype patch failed:", repr(e)[:200], flush=True)
_orig_gif = model.model.get_image_features
def _gif_chunked(pixel_values, **kw):                                           # one image at a time through the vision tower (all images at once needed ~1 GiB each)
    outs = [_orig_gif(pixel_values[i:i + 1], **kw) for i in range(pixel_values.shape[0])]
    res = outs[0]; res.pooler_output = torch.cat([o.pooler_output for o in outs], 0); return res
model.model.get_image_features = _gif_chunked; print("vision tower: one image at a time", flush=True)
''')
g = 'return_tensors="pt").to(DEV_IN)\n    with torch.no_grad(): out = model.generate('
rep(g, 'return_tensors="pt", add_special_tokens=False).to(DEV_IN)\n    with torch.no_grad(): out = model.generate(')
e = "raw = \"\"; errs += 1; print(f\"  ERROR {r['question-id']}: {type(e).__name__} {str(e)[:150]}\", flush=True)"
rep(e, "raw = \"\"; errs += 1; print(f\"  ERROR {r['question-id']}: {type(e).__name__} {str(e)[:150]}\", flush=True); import traceback, gc; print('   ' + ' | '.join(traceback.format_exc().strip().split(chr(10))[-10:])[:1200], flush=True); gc.collect(); torch.cuda.empty_cache()")
open("body_medgemma_local2.py", "w", encoding="utf-8").write(src); print("body_medgemma_local.py written", len(src), "chars")
