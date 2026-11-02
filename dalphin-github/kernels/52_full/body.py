

# ---------------- DALPHIN full run (independent scenario): open VLM from Kaggle Models, images from the dalphin-51-images notebook output ----------------
import os, sys, time, json, glob
import torch, transformers
from PIL import Image
t00 = time.time()
print("torch", torch.__version__, "transformers", transformers.__version__, "| gpu", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB | kind {KIND} limit {LIMIT}", flush=True)
print("gpus:", [(k, torch.cuda.get_device_name(k), round(torch.cuda.get_device_properties(k).total_memory / 2**30, 1)) for k in range(torch.cuda.device_count())], flush=True)
WORK = "/kaggle/working"; OUTD = f"{WORK}/submission"; os.makedirs(OUTD, exist_ok=True)

META = IMG = None
for r, ds, fs in os.walk("/kaggle/input"):
    if META is None and "dalphin_metadata.csv" in fs: META = os.path.join(r, "dalphin_metadata.csv")
    if IMG is None and "case001_wsi1_overview.png" in fs: IMG = r
print("metadata:", META, "| images:", IMG, "| n png:", len(glob.glob(IMG + "/*.png")), flush=True)
meta = pd.read_csv(META); print("questions", len(meta), flush=True)

cands = []
for r, ds, fs in os.walk("/kaggle/input"):
    if "config.json" in fs and any(f.endswith(".safetensors") for f in fs): cands.append(r)
print("model dirs:", cands, flush=True)
from transformers import AutoProcessor
proc = mdir = None
for c in cands:
    try: proc = AutoProcessor.from_pretrained(c); mdir = c; print("processor OK from", c, flush=True); break
    except Exception as e: print("processor failed from", c, type(e).__name__, str(e)[:150], flush=True)
assert proc is not None, "no processor could be loaded"
if "TOKBUDGET_ARG" in globals():                                                 # image token budget test (Gemma-4 gives every image max_soft_tokens = 280 tokens by default, whatever its pixel size): raise it and check BEFORE the slow model load that the prompt really grows
    try:
        _names = split_images(meta.iloc[0])[:3]
        def _count_tokens(side):
            ims = []
            for n in _names:
                im = Image.open(f"{IMG}/{n}").convert("RGB"); k = side / max(im.size)
                ims.append(im.resize((max(28, int(im.size[0] * k)), max(28, int(im.size[1] * k))), Image.BICUBIC) if k < 1 else im)
            t = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}] * len(ims) + [{"type": "text", "text": "x"}]}], tokenize=False, add_generation_prompt=True)
            return int(proc(text=[t], images=[ims], return_tensors="pt")["input_ids"].shape[1])
        _side = globals().get("MAXSIDE_ARG", 896); n_default = _count_tokens(896)
        _ip = getattr(proc, "image_processor", None); _set = []
        for obj, nm in ((_ip, "image_processor"), (proc, "processor")):
            for attr in ("max_soft_tokens", "image_seq_length"):
                if obj is not None and hasattr(obj, attr): setattr(obj, attr, int(TOKBUDGET_ARG)); _set.append(f"{nm}.{attr}")
        n_new = _count_tokens(_side)
        print(f"image token budget -> {TOKBUDGET_ARG} (set: {_set}); prompt tokens for 3 images of case 1: {n_default} (default budget, 896 px) vs {n_new} (budget {TOKBUDGET_ARG}, {_side} px)", flush=True)
        if n_new <= n_default: print("TOKEN BUDGET CHECK: the prompt did not grow, so this test would change nothing - stopping before the model load", flush=True); os._exit(5)
    except Exception as e:
        import traceback; traceback.print_exc(); print("TOKEN BUDGET CHECK FAILED - stopping before the model load:", repr(e)[:300], flush=True); os._exit(6)
MAXSIDE = globals().get("MAXSIDE_ARG", 896)                                     # longest image side after resizing (build.sh env MAXSIDE); the 31B runs so far used 896
if "MAXSIDE_ARG" in globals() and "TOKBUDGET_ARG" not in globals():                # resolution test (skipped when a token budget is set: the processor then resizes every image to fill the budget, so 896 vs 1100 px gives the same token count): check BEFORE the slow model load that the processor really gives more image tokens at this size
    try:
        row0 = meta.iloc[0]; names = split_images(row0)[:3]
        sizes = [Image.open(f"{IMG}/{n}").size for n in sorted(os.listdir(IMG))[:60:2]]
        print("native image sizes (w,h) of 30 images: longest side median", sorted(max(s) for s in sizes)[len(sizes) // 2], "max", max(max(s) for s in sizes), "| processor image settings:", {k: v for k, v in vars(proc.image_processor).items() if not k.startswith("_") and not callable(v)} if hasattr(proc, "image_processor") else "n/a", flush=True)
        def _ntok(side):
            ims = []
            for n in names:
                im = Image.open(f"{IMG}/{n}").convert("RGB"); k = side / max(im.size)
                ims.append(im.resize((max(28, int(im.size[0] * k)), max(28, int(im.size[1] * k))), Image.BICUBIC) if k < 1 else im)
            t = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}] * len(ims) + [{"type": "text", "text": "x"}]}], tokenize=False, add_generation_prompt=True)
            return int(proc(text=[t], images=[ims], return_tensors="pt")["input_ids"].shape[1])
        n896, nnew = _ntok(896), _ntok(MAXSIDE); print(f"prompt tokens for 3 images of case 1: {n896} at 896 px vs {nnew} at {MAXSIDE} px", flush=True)
        if nnew <= n896 and MAXSIDE > 896: print("RESOLUTION CHECK: the processor gives no more tokens at the larger size, so this test would change nothing - stopping before the model load", flush=True); os._exit(5)
    except Exception as e:
        import traceback; traceback.print_exc(); print("resolution check failed (continuing anyway):", repr(e)[:200], flush=True)
if KIND == "q25": from transformers import Qwen2_5_VLForConditionalGeneration as VLM
elif KIND in ("q3", "q3b"):
    try: from transformers import Qwen3VLForConditionalGeneration as VLM
    except Exception: from transformers import AutoModelForImageTextToText as VLM
else: from transformers import AutoModelForImageTextToText as VLM            # gemma4
MULTI = KIND in ("q3b", "gemma4", "gemma31", "gemma31b")                                              # models too big for one T4: spread over all visible GPUs
DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}[DTYPE]
kw = dict(torch_dtype=DT, attn_implementation="sdpa")
if MULTI and torch.cuda.device_count() >= 2: kw.update(device_map="auto", max_memory={k: "13GiB" for k in range(torch.cuda.device_count())})
if KIND == "gemma31b":                                                         # full-precision Gemma-4-31B-it, quantised to 4-bit (nf4) while loading, spread over both GPUs
    try: import bitsandbytes; print("bitsandbytes", bitsandbytes.__version__, flush=True)
    except Exception:
        import subprocess; r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], capture_output=True, text=True); print("pip install bitsandbytes ->", r.returncode, (r.stdout + r.stderr)[-300:], flush=True)
    from transformers import BitsAndBytesConfig
    kw.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DT, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)); print("4-bit nf4 bitsandbytes on load", flush=True)
elif MULTI:                                                                    # one GPU only: 4-bit weights
    try:
        from transformers import BitsAndBytesConfig
        kw.update(device_map="cuda", quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=DT, bnb_4bit_quant_type="nf4")); print("single GPU -> 4-bit bitsandbytes", flush=True)
    except Exception as e: raise RuntimeError("single GPU and no 4-bit support: " + repr(e))
else: kw.update(device_map="cuda")
if KIND == "gemma31":
    try: cfg = json.load(open(f"{mdir}/config.json")); print("quantization_config:", json.dumps(cfg.get("quantization_config", cfg.get("text_config", {}).get("quantization_config")))[:700], "| files:", sorted(os.listdir(mdir))[:20], flush=True)
    except Exception as e: print("config read failed:", repr(e)[:200], flush=True)
    try: import compressed_tensors; print("compressed_tensors", getattr(compressed_tensors, "__version__", "?"), flush=True)
    except Exception:
        import subprocess; r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "compressed-tensors"], capture_output=True, text=True); print("pip install compressed-tensors ->", r.returncode, (r.stdout + r.stderr)[-300:], flush=True)
try: model = VLM.from_pretrained(mdir, **kw).eval()
except Exception as e:
    import traceback; traceback.print_exc(); print("MODEL LOAD FAILED:", repr(e)[:400], flush=True); sys.stdout.flush(); os._exit(3)
DEV_IN = getattr(model, "device", "cuda")
hm = getattr(model, "hf_device_map", None)
if hm: print("device map:", {str(v): sum(1 for x in hm.values() if x == v) for v in set(hm.values())}, flush=True)
def peak_gb(): return max(torch.cuda.max_memory_allocated(k) for k in range(torch.cuda.device_count())) / 2**30
print(f"model loaded in {time.time() - t00:.0f}s | dtype {DTYPE} | gpu mem per device {[round(torch.cuda.memory_allocated(k) / 2**30, 1) for k in range(torch.cuda.device_count())]} GB", flush=True)

MAXTOK = {"neoplasm": 6, "behavior": 8, "mc": 6, "tissue": 24, "diagnosis": 40, "open": 120}
if "OPEN_MAXTOK_ARG" in globals(): MAXTOK["open"] = int(OPEN_MAXTOK_ARG); print("open-answer token limit ->", MAXTOK["open"], flush=True)
if "OPEN_NOSUFFIX_ARG" in globals(): SUFFIX["open"] = ""; print("open-answer task instruction removed (organisers' prompt only)", flush=True)
if "TASKMAXTOK_ARG" in globals(): MAXTOK.update({k: int(v) for k, v in TASKMAXTOK_ARG.items()}); print("new-token limits ->", MAXTOK, flush=True)
if "NOSUFFIX_ARG" in globals():
    for _t in NOSUFFIX_ARG: SUFFIX[_t] = ""
    print("task instruction removed (organisers' prompt only) for:", list(NOSUFFIX_ARG), flush=True)
cur = {"task": "open"}


def load_img(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; k = MAXSIDE / max(w, h)
    return im.resize((max(28, int(w * k)), max(28, int(h * k))), Image.BICUBIC) if k < 1 else im


def generate(messages):
    parts = messages[0]["content"]; ims, content = [], []
    for p in parts:
        if p["type"] == "image": ims.append(load_img(p["path"])); content.append({"type": "image"})
        else: content.append(p)
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=([ims] if (KIND in ("gemma4", "gemma31", "gemma31b") and ims) else (ims or None)), return_tensors="pt").to(DEV_IN)
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=MAXTOK[cur["task"]], do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


report = {}
for task in (ONLY_TASKS or TASKS):                                                 # ONLY_TASKS (from build.sh env TASKS_ONLY) limits the run to some tasks, in that order
    cur["task"] = task
    full = meta[meta["question-id"].str.match(rf"^case\d+_{task}\d*$")]
    sub = full.head(LIMIT) if LIMIT else full; rows = []; t2 = time.time(); errs = 0
    for k, (_, r) in enumerate(sub.iterrows(), 1):
        try: raw = generate(build_messages(r, IMG))
        except Exception as e: raw = ""; errs += 1; print(f"  ERROR {r['question-id']}: {type(e).__name__} {str(e)[:150]}", flush=True)
        if errs >= 3 and errs == k: print("the first 3 questions all failed - stopping instead of writing empty answers", flush=True); os._exit(7)
        rows.append((r["question-id"], postprocess(task, raw, str(r["question"])), raw))
        if k % 50 == 0 and k < len(sub): pd.DataFrame(rows, columns=["question-id", "response", "raw"]).to_csv(f"{WORK}/{task}_partial.csv", index=False)     # partial save: survives a watchdog stop
        if k % 50 == 0 or k == len(sub): print(f"  [{task}] {k}/{len(sub)} | {time.time() - t2:.0f}s | {(time.time() - t2) / k:.1f} s/q | errors {errs} | peak {peak_gb():.1f} GB", flush=True)
    pred = pd.DataFrame(rows, columns=["question-id", "response", "raw"])
    pred.to_csv(f"{WORK}/{task}_raw.csv", index=False)
    pred[["question-id", "response"]].to_csv(f"{OUTD}/{task}.csv", index=False)
    report[task] = validate(task, pred, full) if not LIMIT else "skipped (limit)"
    print(f"[{task}] done {len(pred)} q in {time.time() - t2:.0f}s | validation: {report[task]}", flush=True)
    print("   sample:", [(a, b[:40]) for a, b in zip(pred['question-id'][:3], pred['response'][:3])], flush=True)
print("VALIDATION REPORT", json.dumps(report), flush=True)
print(f"DONE total {time.time() - t00:.0f}s", flush=True)
