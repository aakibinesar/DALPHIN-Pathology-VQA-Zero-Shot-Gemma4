# CPU-only check that this notebook can read the gated google/medgemma-27b-it repo with the Kaggle secret HF_TOKEN. Downloads only small config files; the token is never printed.
import sys, json
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print("SECRET NOT AVAILABLE:", type(e).__name__, "- open the notebook in the Kaggle editor, Add-ons > Secrets, switch HF_TOKEN on, then run again", flush=True); sys.exit(3)
print("secret HF_TOKEN found (length", len(tok), ")", flush=True)
import transformers, huggingface_hub
print("transformers", transformers.__version__, "| huggingface_hub", huggingface_hub.__version__, flush=True)
from huggingface_hub import HfApi, hf_hub_download
REPO = "google/medgemma-27b-it"
try:
    info = HfApi(token=tok).model_info(REPO, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    print(f"repo listing OK: {len(files)} files, {sum(s for _, s in files) / 1e9:.1f} GB in total", flush=True)
    for n, s in files:
        if n.endswith((".json", ".model")) or s > 1e9: print(f"   {n}  {s / 1e9:.2f} GB", flush=True)
    for fn in ("config.json", "preprocessor_config.json", "generation_config.json"):
        p = hf_hub_download(REPO, fn, token=tok); d = json.load(open(p))
        print(f"downloaded {fn} OK |", {k: d[k] for k in list(d)[:12] if not isinstance(d[k], (dict, list))}, flush=True)
    print("ACCESS OK", flush=True)
except Exception as e:
    print("ACCESS FAILED:", type(e).__name__, str(e)[:300], flush=True); sys.exit(4)
