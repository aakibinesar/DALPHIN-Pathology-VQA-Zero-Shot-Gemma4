# CPU notebook: download DALPHIN images.zip (Zenodo 18609450, CC BY-NC-ND 4.0) with resume + retries, verify md5, extract the PNGs only (no macOS '._' junk).
import os, sys, time, json, hashlib, zipfile, urllib.request

REC = "https://zenodo.org/api/records/18609450"
W = "/kaggle/working"; ZIP = f"{W}/images.zip"; OUT = f"{W}/images"; os.makedirs(OUT, exist_ok=True)
DEADLINE = time.time() + 100 * 60                         # give up after 100 minutes (partial zip is kept for a resume)
t0 = time.time()

rec = json.load(urllib.request.urlopen(REC, timeout=60))
f = next(x for x in rec["files"] if x["key"] == "images.zip")
url, size, md5 = f["links"]["self"], int(f["size"]), f["checksum"].split(":")[-1]
print(f"images.zip {size / 1e6:.0f} MB md5 {md5}", flush=True)

have = os.path.getsize(ZIP) if os.path.exists(ZIP) else 0
tries = 0
while have < size and time.time() < DEADLINE:
    try:
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        with urllib.request.urlopen(req, timeout=120) as r, open(ZIP, "ab") as fo:
            last = time.time(); n0 = have
            while True:
                b = r.read(1 << 20)
                if not b: break
                fo.write(b); have += len(b)
                if time.time() - last > 30:
                    print(f"  {have / 1e6:.0f}/{size / 1e6:.0f} MB | {(have - n0) / (time.time() - last + 1e-9) / 1e6:.1f} MB/s | {time.time() - t0:.0f}s", flush=True); last = time.time(); n0 = have
                if time.time() > DEADLINE: break
    except Exception as e:
        tries += 1; print(f"  retry {tries} at {have / 1e6:.0f} MB: {type(e).__name__} {str(e)[:100]}", flush=True); time.sleep(min(60, 5 * tries))
        have = os.path.getsize(ZIP) if os.path.exists(ZIP) else 0
print(f"download state: {have}/{size} bytes in {time.time() - t0:.0f}s", flush=True)
if have < size: print("INCOMPLETE - hit the time limit; rerun to resume"); sys.exit(1)

h = hashlib.md5()
with open(ZIP, "rb") as fi:
    for chunk in iter(lambda: fi.read(1 << 24), b""): h.update(chunk)
print("md5 ok" if h.hexdigest() == md5 else f"MD5 MISMATCH {h.hexdigest()} vs {md5}", flush=True)
if h.hexdigest() != md5: sys.exit(2)

n = 0
with zipfile.ZipFile(ZIP) as z:
    for info in z.infolist():
        base = os.path.basename(info.filename)
        if not base.lower().endswith(".png") or base.startswith("._") or "__MACOSX" in info.filename: continue
        with z.open(info) as fi, open(f"{OUT}/{base}", "wb") as fo: fo.write(fi.read())
        n += 1
print(f"extracted {n} PNGs, {sum(os.path.getsize(f'{OUT}/{x}') for x in os.listdir(OUT)) / 1e6:.0f} MB", flush=True)
os.remove(ZIP)                                              # keep only the PNGs in the output (no duplicate 1.2 GB)

import pandas as pd
meta = pd.read_csv(urllib.request.urlretrieve(f"{REC}/files/dalphin_metadata.csv/content", f"{W}/dalphin_metadata.csv")[0])
need = set()
for col in ("overviews", "rois"):
    for v in meta[col].dropna(): need |= {p.strip() for p in str(v).split(",") if p.strip()}
have_n = set(os.listdir(OUT)); miss = sorted(need - have_n)
print(f"metadata references {len(need)} images; present {len(need & have_n)}; missing {len(miss)} {miss[:5]}", flush=True)
print(f"DONE in {time.time() - t0:.0f}s", flush=True)
