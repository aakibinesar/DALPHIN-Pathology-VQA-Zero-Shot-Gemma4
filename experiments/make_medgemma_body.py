# Builds body_medgemma.py from body.py (body.py itself is not touched): the weights of google/medgemma-27b-it are downloaded from Hugging Face (gated; the token comes from the Kaggle secret HF_TOKEN and is never printed)
# instead of being read from /kaggle/input. Everything after the processor (4-bit nf4 load over both T4s, generation, post-processing, validation) is the same code as the Gemma-4-31B runs (KIND gemma31b).
src = open("body.py", encoding="utf-8").read()
i0 = src.index("cands = []\n"); a = "assert proc is not None, \"no processor could be loaded\"\n"; i1 = src.index(a) + len(a)
new = r'''REPO = "google/medgemma-27b-it"
try:
    from kaggle_secrets import UserSecretsClient; HF_TOK = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print("SECRET HF_TOKEN NOT AVAILABLE:", type(e).__name__, "- run this notebook from the Kaggle editor (Save & Run All) with the secret switched on", flush=True); os._exit(3)
import shutil
def _free(p):
    try: os.makedirs(p, exist_ok=True); return shutil.disk_usage(p).free / 1e9
    except Exception: return -1.0
_dirs = {p: _free(p) for p in ("/kaggle/temp", "/tmp", "/kaggle/working")}
print("free disk GB:", {k: round(v, 1) for k, v in _dirs.items()}, "| RAM GB:", round(int([l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]) / 1e6, 1), flush=True)
DL = max(("/kaggle/temp", "/tmp"), key=lambda p: _dirs[p]); DLDIR = f"{DL}/medgemma27"
if _dirs[DL] < 56.5: print(f"NOT ENOUGH DISK: {_dirs[DL]:.1f} GB free in {DL}, the weights are 54.9 GB - stopping before the download", flush=True); os._exit(8)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
try: mdir = snapshot_download(REPO, token=HF_TOK, local_dir=DLDIR, allow_patterns=["*.json", "*.model", "*.jinja"])             # small files first: processor + prompt check BEFORE the 55 GB download
except Exception as e: print("DOWNLOAD (small files) FAILED:", type(e).__name__, str(e)[:300], flush=True); os._exit(4)
proc = AutoProcessor.from_pretrained(mdir); print("processor OK", type(proc).__name__, "| files:", sorted(f for f in os.listdir(mdir) if not f.endswith(".safetensors")), "| processor chat_template:", bool(getattr(proc, "chat_template", None)), "| tokenizer chat_template:", bool(getattr(proc.tokenizer, "chat_template", None)), flush=True)
if not getattr(proc, "chat_template", None) and getattr(proc.tokenizer, "chat_template", None): proc.chat_template = proc.tokenizer.chat_template; print("processor chat template taken from the tokenizer", flush=True)
try:
    _names = split_images(meta.iloc[0])[:3]; _ims = [Image.open(f"{IMG}/{n}").convert("RGB") for n in _names]
    _t = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}] * len(_ims) + [{"type": "text", "text": "QQQ"}]}], tokenize=False, add_generation_prompt=True)
    _ids = proc(text=[_t], images=[_ims], return_tensors="pt", add_special_tokens=False)["input_ids"][0]; _n = int(_ids.shape[0])
    print(f"prompt check: {_n} tokens for 3 images of case 1 (native sizes {[im.size for im in _ims]}) | first ids {_ids[:4].tolist()} (bos id {proc.tokenizer.bos_token_id}) | template head: {_t[:60]!r} | tail: {_t[-160:]!r}", flush=True)
    assert "QQQ" in _t and _n > 3 * 200, "prompt did not render the images"
    assert _t.startswith("<bos>") and int(_ids[0]) == proc.tokenizer.bos_token_id and int(_ids[1]) != proc.tokenizer.bos_token_id, "the template text must start with exactly one <bos> (add_special_tokens=False is used)"
except Exception as e:
    import traceback; traceback.print_exc(); print("PROMPT CHECK FAILED - stopping before the big download:", repr(e)[:300], flush=True); os._exit(6)
t_dl = time.time()
try: mdir = snapshot_download(REPO, token=HF_TOK, local_dir=DLDIR, max_workers=8, allow_patterns=["*.json", "*.model", "*.jinja", "*.safetensors"])
except Exception as e: print("DOWNLOAD FAILED:", type(e).__name__, str(e)[:300], flush=True); os._exit(4)
del HF_TOK
print(f"downloaded {sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(mdir) for f in fs) / 1e9:.1f} GB in {time.time() - t_dl:.0f}s | free disk now {shutil.disk_usage(DL).free / 1e9:.1f} GB", flush=True)
'''
out = src[:i0] + new + src[i1:]
g = 'return_tensors="pt").to(DEV_IN)' + chr(10) + '    with torch.no_grad(): out = model.generate('
assert out.count(g) == 1; out = out.replace(g, 'return_tensors="pt", add_special_tokens=False).to(DEV_IN)' + chr(10) + '    with torch.no_grad(): out = model.generate(')      # the chat template text already starts with <bos>: do not add a second one
v = 'print("VALIDATION REPORT", json.dumps(report), flush=True)'
assert out.count(v) == 1
save = '''if "SAVE4BIT_ARG" in globals():                                                       # optional: keep the 4-bit weights as this notebook's output so later runs need no Hugging Face token (done right after the load)
    try:
        _gb = model.get_memory_footprint() / 1e9; print(f"4-bit model footprint {_gb:.1f} GB", flush=True)
        if _gb < 17.5: t3 = time.time(); model.save_pretrained(f"{WORK}/medgemma27_nf4", safe_serialization=True); proc.save_pretrained(f"{WORK}/medgemma27_nf4"); print(f"saved 4-bit copy in {time.time() - t3:.0f}s: {sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(WORK + '/medgemma27_nf4') for f in fs) / 1e9:.1f} GB", flush=True)
        else: print("4-bit copy NOT saved: too large for the 20 GB output limit", flush=True)
    except Exception as e: print("4-bit save failed:", repr(e)[:300], flush=True)
'''
mk = 'MAXTOK = {"neoplasm"'; assert out.count(mk) == 1; out = out.replace(mk, save + mk)      # saved right after the model load (before the generation), so a watchdog stop later does not lose the copy
open("body_medgemma.py", "w", encoding="utf-8").write(out); print("body_medgemma.py written", len(out), "chars")
