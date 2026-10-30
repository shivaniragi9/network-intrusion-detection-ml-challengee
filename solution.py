"""Network intrusion detection (Macro F1). sklearn only, runs in ~1 min.

Key finding from the data: 99.99% of train rows with id.resp_p == 21 are
Availability (79,698 of 79,699) and nothing else is, so port 21 is the
Availability signal. Flow features such as bwd_pkts_tot are NOT safe for it:
only 0.1% of train Availability flows have no reply, but 88% of test port-21
flows have none (shift), and models using them wrongly call these Recon/Benign.
So: port 21 -> Availability; the other 3 classes come from a model trained
only on Benign/Reconnaissance/Spoofing rows.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier

SEED = 42
OUT = Path("/kaggle/working/submission.csv")
if not OUT.parent.exists():
    OUT = Path("submission.csv")
LABELS = ["Availability", "Benign", "Reconnaissance", "Spoofing"]
REST = LABELS[1:]


def find_dir():
    need = {"train.csv", "test.csv", "sample_submission.csv"}
    roots = [Path("/kaggle/input"), Path(".")]
    hits = {p.parent for r in roots if r.exists() for p in r.rglob("train.csv")
            if need <= {q.name for q in p.parent.iterdir()}}
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one data folder, found {sorted(map(str, hits))}")
    return hits.pop()


d = find_dir()
tr, te, sample = (pd.read_csv(d / f) for f in ("train.csv", "test.csv", "sample_submission.csv"))
assert list(sample.columns) == ["id", "Target"] and sample["id"].equals(te["id"])
assert set(tr["Target"]) == set(LABELS)

# 3-class training set (no Availability rows)
rest = tr[tr["Target"] != "Availability"].reset_index(drop=True)
y = rest["Target"].map({l: i for i, l in enumerate(REST)}).values
feats = [c for c in tr.columns if c not in ("id", "Target")]
cats = ["proto", "service"]
for c in cats:
    vocab = pd.Categorical(pd.concat([tr[c], te[c]]).astype(str)).categories
    rest[c] = pd.Categorical(rest[c].astype(str), categories=vocab)
    te[c] = pd.Categorical(te[c].astype(str), categories=vocab)
X, Z = rest[feats], te[feats]
Xc, Zc = X.copy(), Z.copy()
for c in cats:
    Xc[c], Zc[c] = X[c].cat.codes, Z[c].cat.codes
Xc, Zc = Xc.fillna(-1), Zc.fillna(-1)

hgb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, class_weight="balanced",
                                     categorical_features="from_dtype", random_state=SEED).fit(X, y)
et = ExtraTreesClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1,
                          random_state=SEED).fit(Xc, y)
P = hgb.predict_proba(Z) + et.predict_proba(Zc)
pred = np.array(REST)[P.argmax(1)]
pred = np.where(te["id.resp_p"].values == 21, "Availability", pred)

sub = pd.DataFrame({"id": te["id"], "Target": pred})
sub.to_csv(OUT, index=False)
chk = pd.read_csv(OUT)
assert list(chk.columns) == ["id", "Target"] and len(chk) == len(sample)
assert chk["id"].equals(sample["id"]) and set(chk["Target"]) <= set(LABELS)
print(chk["Target"].value_counts(normalize=True).round(4).to_dict(), "->", OUT)
