"""5-fold stratified CV of the final pipeline's 3-class model (Benign/Recon/Spoofing).
Run:  python cv_evaluation.py   (needs train.csv in ./data or /kaggle/input)
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

SEED = 42
REST = ["Benign", "Reconnaissance", "Spoofing"]
p = next(q for r in (Path("/kaggle/input"), Path(".")) if r.exists() for q in r.rglob("train.csv"))
tr = pd.read_csv(p)
rest = tr[tr["Target"] != "Availability"].reset_index(drop=True)
y = rest["Target"].map({l: i for i, l in enumerate(REST)}).values
feats = [c for c in tr.columns if c not in ("id", "Target")]
cats = ["proto", "service"]
for c in cats:
    rest[c] = pd.Categorical(rest[c].astype(str))
X = rest[feats]
Xc = X.copy()
for c in cats:
    Xc[c] = X[c].cat.codes
Xc = Xc.fillna(-1)

oof = {k: np.zeros((len(y), 3)) for k in ("hgb", "et")}
for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
    h = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, class_weight="balanced",
                                       categorical_features="from_dtype", random_state=SEED).fit(X.iloc[a], y[a])
    e = ExtraTreesClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1,
                             random_state=SEED).fit(Xc.iloc[a], y[a])
    oof["hgb"][b], oof["et"][b] = h.predict_proba(X.iloc[b]), e.predict_proba(Xc.iloc[b])
for name, P in (("HistGB only", oof["hgb"]), ("ExtraTrees only", oof["et"]), ("Ensemble (final)", oof["hgb"] + oof["et"])):
    print(f"{name:18s} macro-F1 (3 classes) = {f1_score(y, P.argmax(1), average='macro'):.4f}")
print(classification_report(y, (oof["hgb"] + oof["et"]).argmax(1), target_names=REST, digits=4))
