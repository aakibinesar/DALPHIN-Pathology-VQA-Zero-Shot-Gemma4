import random, re, pandas as pd, sys
sys.path.insert(0, ".")
import dalphin_pipeline as dp
random.seed(0)
meta = pd.read_csv("dalphin_metadata.csv")
# stub 'model' that answers in messy ways to exercise the post-processing
def stub(messages):
    t = messages[0]["content"][0]["text"]
    if "yes or no" in t: return random.choice(["Yes.", "No", "yes, there is a neoplasm", "The lesion is non-neoplastic.", "NO."])
    if "benign, uncertain, in situ, malignant" in t: return random.choice(["Malignant.", "benign", "in-situ carcinoma", "Uncertain behavior", "adenocarcinoma, so malignant"])
    if "single letter" in t: return random.choice(["B", "The answer is C.", "d)", "A. Mucosal erosions", "E"])
    if "organ or tissue" in t: return random.choice(["Appendix.\nBecause ...", "skin", "Colon "])
    return random.choice(["Adenocarcinoma, intestinal type.", "The lesion shows glands with cribriform growth.  Mitoses are present."])
rep = dp.run("dalphin_metadata.csv", "data/images", stub, "test_out")      # full run with the stub: every task, every question
for t, r in rep.items(): print(f"{t:10s} n={r['n']:3d} validation errors: {r['errors']}")
# check the validator catches problems
bad = pd.DataFrame({"question-id": ["case001_neoplasm", "case001_neoplasm"], "response": ["maybe", "yes"]})
print("validator on a broken file:", dp.validate("neoplasm", bad, meta))
print("behavior samples:", [dp.postprocess("behavior", x) for x in ["in-situ carcinoma", "Benign lesion", "likely a carcinoma", "uncertain malignant potential", "??"]])
print("mc samples:", [dp.postprocess("mc", x, "Which? A. x; B. y; C. z; D. w; E. v") for x in ["B", "The answer is C.", "d)", "A. foo", "none", "F"]])
print("yes/no samples:", [dp.postprocess("neoplasm", x) for x in ["Yes.", "No neoplasm seen", "non-neoplastic lesion", "yes, a neoplasm", ""]])
