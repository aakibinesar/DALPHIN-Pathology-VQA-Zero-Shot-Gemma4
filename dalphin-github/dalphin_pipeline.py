"""DALPHIN pipeline (model-agnostic): build prompts from dalphin_metadata.csv, post-process raw model text into the answer vocabulary each task requires,
write one CSV per task (columns question-id, response) and validate the format exactly as the organisers' evaluation code does.
The model is plugged in as generate(messages) -> str (OpenAI-style chat messages with local image paths)."""
import re, sys
import pandas as pd
from pathlib import Path

TASKS = ["tissue", "neoplasm", "behavior", "diagnosis", "mc", "open"]
SUFFIX = {   # instruction appended to the question text (the organisers' preamble is kept unchanged in front of it)
    "neoplasm": "\nAnswer with exactly one word: yes or no.",
    "behavior": "\nAnswer with exactly one of: benign, uncertain, in situ, malignant. Do not add any other text.",
    "mc": "\nAnswer with the single letter of the correct option only.",
    "tissue": "\nAnswer with the organ or tissue name only (a few words).",
    "diagnosis": "\nAnswer with the single most likely diagnosis name only.",
    "open": "\nAnswer concisely in 2 to 4 sentences.",
}


def task_of(qid: str) -> str:
    return re.sub(r"\d+$", "", re.sub(r"^case\d+_", "", qid))


def split_images(row):
    out = []
    for col in ("overviews", "rois"):
        v = row.get(col)
        if isinstance(v, str) and v.strip(): out += [p.strip() for p in v.split(",") if p.strip()]
    seen, res = set(), []
    for p in out:
        if p not in seen: seen.add(p); res.append(p)
    return res


def build_messages(row, image_dir):
    t = task_of(row["question-id"]); pre = str(row.get("preamble", "")) if isinstance(row.get("preamble"), str) else ""
    text = (pre + "\n" if pre else "") + str(row["question"]) + SUFFIX.get(t, "")
    parts = [{"type": "text", "text": text}] + [{"type": "image", "path": str(Path(image_dir) / n)} for n in split_images(row)]
    return [{"role": "user", "content": parts}]


# ---- post-processing: raw text -> allowed vocabulary --------------------------------------------------------------------------------
def norm_yesno(s):
    t = s.strip().lower()
    m = re.search(r"\b(yes|no)\b", t)
    if m: return m.group(1)
    if re.search(r"non-?neoplastic|no neoplas|not neoplas|benign reactive|normal", t): return "no"
    return "yes"


def norm_behavior(s):
    t = s.strip().lower().replace("-", " ")
    pos = {k: (m.start() if (m := re.search(p, t)) else 10 ** 9) for k, p in (("in situ", r"in\s?situ"), ("benign", r"\bbenign\b|non neoplastic"), ("malignant", r"malignan|carcinoma|cancer"), ("uncertain", r"uncertain|borderline|undetermined"))}
    k = min(pos, key=pos.get)
    return k if pos[k] < 10 ** 9 else "malignant"


def norm_letter(s, question=""):
    allowed = sorted(set(re.findall(r"\b([A-F])[.)]\s", question))) or list("ABCDEF")
    m = re.search(r"\b([A-Fa-f])\b", s.strip())
    c = (m.group(1).upper() if m else allowed[0])
    return (c if c in allowed else allowed[0]).lower()


def norm_text(s):
    return re.sub(r"\s+", " ", s.strip()).strip()


def strip_brackets(s):
    """Diagnosis only: remove parenthetical content (abbreviations, qualifiers), e.g. 'Ductal Carcinoma In Situ (DCIS)' -> 'Ductal Carcinoma In Situ'."""
    out = s
    while True:                                                 # innermost brackets first, so nested ones go too
        new = re.sub(r"\s*\([^()]*\)", "", out)
        if new == out: break
        out = new
    out = re.sub(r"\s*\([^()]*$", "", out)                      # an unclosed bracket at the end
    return re.sub(r"\s+", " ", out).strip(" ,;:.")


def postprocess(task, raw, question=""):
    raw = raw if isinstance(raw, str) else ""
    if task == "neoplasm": return norm_yesno(raw)
    if task == "behavior": return norm_behavior(raw)
    if task == "mc": return norm_letter(raw, question)
    if task == "tissue": return norm_text(raw.split("\n")[0]).rstrip(".")
    if task == "diagnosis":
        t = strip_brackets(norm_text(raw)); return t if t else norm_text(raw)      # never return an empty answer
    return norm_text(raw)


# ---- validation, mirroring code/gc_evaluation/*/evaluate.py -------------------------------------------------------------------------
ALLOWED = {"neoplasm": {"yes", "no"}, "behavior": {"benign", "malignant", "uncertain", "in situ"}}


def validate(task, pred: pd.DataFrame, meta: pd.DataFrame):
    exp = set(meta[meta["question-id"].str.match(rf"^case\d+_{task}\d*$")]["question-id"].astype(str)); errs = []
    for c in ("question-id", "response"):
        if c not in pred.columns: errs.append(f"missing column {c}")
    if not errs:
        got = list(pred["question-id"].astype(str))
        if set(got) != exp or len(got) != len(set(got)): errs.append(f"question-id mismatch: missing {len(exp - set(got))}, extra {len(set(got) - exp)}, duplicates {len(got) - len(set(got))}")
        if task in ALLOWED:
            bad = [(q, r) for q, r in zip(pred["question-id"], pred["response"]) if str(r).strip().rstrip(".,").lower() not in ALLOWED[task] | {""}]
            if bad: errs.append(f"invalid responses e.g. {bad[:3]}")
        if task == "mc":
            bad = [(q, r) for q, r in zip(pred["question-id"], pred["response"]) if not re.fullmatch(r"[a-f]", str(r).strip().rstrip(".,").lower())]
            if bad: errs.append(f"mc responses must be a single letter a-f, e.g. {bad[:3]}")
    return errs


def run(meta_path, image_dir, generate, out_dir, limit=None, tasks=TASKS):
    meta = pd.read_csv(meta_path); Path(out_dir).mkdir(parents=True, exist_ok=True); report = {}
    for task in tasks:
        sub = meta[meta["question-id"].str.match(rf"^case\d+_{task}\d*$")]
        if limit: sub = sub.head(limit)
        rows = []
        for _, r in sub.iterrows():
            raw = generate(build_messages(r, image_dir)); rows.append((r["question-id"], postprocess(task, raw, str(r["question"])), raw))
        pred = pd.DataFrame(rows, columns=["question-id", "response", "raw"])
        pred[["question-id", "response"]].to_csv(f"{out_dir}/{task}.csv", index=False); pred.to_csv(f"{out_dir}/{task}_raw.csv", index=False)
        full = meta[meta["question-id"].str.match(rf"^case\d+_{task}\d*$")]
        report[task] = dict(n=len(pred), errors=validate(task, pred, full) if not limit else "skipped (limit)")
    return report


if __name__ == "__main__":
    pass
