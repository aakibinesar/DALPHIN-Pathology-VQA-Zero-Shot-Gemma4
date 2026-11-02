"""Merge the MedGemma diagnosis answers (250 from the first run + the last 50 from the tail run) and reduce long answers to a diagnosis name with FIXED rules (no labels used), in the spirit of the organisers' curation
(extract the predicted diagnosis; drop bracketed text; keep the first diagnosis). Answers of at most 60 characters are kept as the model wrote them. For longer answers, in this order:
  1. the LAST **bold** span that is not a list header (no trailing ':')
  2. text after 'most likely diagnosis ... is/are/would be (:)', 'consistent with (:)', 'diagnosis ... is:'
  3. 'appears to be (a|an) X': X cut at the first ',' ';' '.' (a 'likely Y' in the same sentence wins)
  every candidate is cut at ', which|, a ...'-style continuations and REJECTED when it describes morphology instead of naming a diagnosis ('proliferation of', 'characterized', 'cells ...'), is a single generic word,
  or looks cut off by the 96-token limit (the answer ends without a closing mark and the last word has no usual ending) -> then the question keeps the Gemma-31B answer of the 0.734 file; every such case is listed and counted.
usage: python clean_medgemma_diag.py"""
import re, sys
import pandas as pd
sys.path.insert(0, ".")
import dalphin_pipeline as dp

a = pd.read_csv("out_medgemma_real/diagnosis_partial.csv", keep_default_na=False); b = pd.read_csv("out_medgemma_tail/diagnosis_raw.csv", keep_default_na=False)
d = pd.concat([a[["question-id", "raw"]], b[["question-id", "raw"]]], ignore_index=True); assert len(d) == 300 and d["question-id"].is_unique
fallback = pd.read_csv("out_hybdiag/diagnosis.csv", keep_default_na=False).set_index("question-id").response.astype(str)
OK_END = re.compile(r"(oma|omas|osis|itis|emia|ia|ism|ine|ase|cyst|tumou?r|neoplasm|polyp|polyposis|sis|gist|net|cele|ome|ity|noma)$", re.I)
BAD = re.compile(r"proliferation|characterized|involving|possibilit|\bcells\b|appears|features|pattern", re.I)

def tidy(x):
    x = re.sub(r"\*+", "", x); x = re.sub(r"\s+", " ", x).strip(" \t\n:;,.-\"'")
    x = dp.strip_brackets(x) or x
    x = re.split(r",\s*(?:which|a |an |the |and |with |specifically|likely|given|as |that |this )", x, maxsplit=1, flags=re.I)[0]
    x = re.split(r"\s+(?:given|due to|based on|because|which|since)\b", x, maxsplit=1, flags=re.I)[0]
    return re.sub(r"^(a|an|the)\s+", "", x, flags=re.I).strip(" ,.;:")

def acceptable(x, t, truncated_tail):
    if len(x) < 4 or len(x) > 80 or BAD.search(x): return False
    if len(x.split()) == 1 and x.lower() in {"stroma", "cellularity", "sarcoma", "carcinoma", "neoplasm", "tumor", "tumour", "lymphoma"}: return False
    if truncated_tail and not OK_END.search(x.split()[-1]): return False
    return True

def extract(raw):
    t = " ".join(str(raw).split())
    if len(t) <= 60: return tidy(t), "short"
    odd_bold = t.count("**") % 2 == 1
    for mm in reversed(list(re.finditer(r"\*\*([^*]{3,90}?)\*\*(?!\s*:)", t))):          # bold spans; list headers like '**Stroma:**' are skipped
        x = tidy(mm.group(1))
        if not mm.group(1).rstrip().endswith(":") and acceptable(x, t, False): return x, "bold"
    for pat in (r"most likely (?:specific )?diagnosis[^.:]{0,80}?(?: is| are| would be| being)\s*:?\s+([^.;\n]{4,100})", r"(?:is|are|be) consistent with:?\s+([^.;\n]{4,100})", r"likely diagnosis[^.:]{0,80}?would be:?\s+([^.;\n]{4,100})", r"diagnosis[^.:]{0,40}?is:\s*([^.;\n]{4,100})"):
        ms = list(re.finditer(pat, t, flags=re.I))
        if ms:
            mm = ms[-1]; x = tidy(mm.group(1)); at_end = t.rstrip().endswith(mm.group(1).rstrip()) and not t.rstrip().endswith((".", "**"))
            if acceptable(x, t, at_end or odd_bold): return x, "diagnosis-is"
    mm = re.search(r"appears to be (?:a|an)\s+([^.]{4,160})", t, flags=re.I)
    if mm:
        s = mm.group(1); lk = re.search(r"\blikely\s+(?:a\s+|an\s+)?([^,.;]{4,80})", s, flags=re.I)
        x = tidy(lk.group(1)) if lk else tidy(re.split(r"[,.;]", s)[0])
        if acceptable(x, t, False): return x, "appears-to-be"
    return None, "flag"

res, how = [], []
for q, raw in zip(d["question-id"], d["raw"]):
    x, h = extract(raw)
    if x is None: x, h = fallback[q], "FALLBACK(Gemma)"
    res.append(x); how.append(h)
d["response"] = res; d["rule"] = how
print("rules used:", d["rule"].value_counts().to_dict())
long = d[d["rule"] != "short"]
for q, r, h in zip(long["question-id"], long["response"], long["rule"]): print(f"{q[:7]} [{h:15s}] {r}")
meta = pd.read_csv("dalphin_metadata.csv"); exp = meta[meta["question-id"].str.match(r"^case\d+_diagnosis\d*$")]
out = d.set_index("question-id").loc[exp["question-id"], ["response"]].reset_index()
errs = dp.validate("diagnosis", out.assign(raw=out.response), exp)
print("rows", len(out), "| validator", errs, "| blank", int((out.response == "").sum()), "| max length", out.response.str.len().max(), "| non-ASCII", sum(1 for c in "".join(out.response) if ord(c) > 127), "| asterisks", int(out.response.str.contains("*", regex=False).sum()))
out.to_csv("out_medgemma_tail/diagnosis_medgemma_clean.csv", index=False); d.to_csv("out_medgemma_tail/diagnosis_medgemma_clean_audit.csv", index=False)
g = fallback.reindex(out["question-id"]).str.lower().values; m_ = out.response.str.lower().values
print("exact agreement with the Gemma hybdiag file:", round(float((g == m_).mean()), 3), "| FALLBACK answers:", int((d["rule"] == "FALLBACK(Gemma)").sum()))
