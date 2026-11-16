"""Train the layer-weighted head on frozen mHuBERT-147 embeddings.

Experiments
-----------
E1 main            train on ArA-DF train (EGY + Gulf + MSA) -> every test set
E2 unseen dialect  leave-one-dialect-out: train on 2 dialects, test on the 3rd
E3 layer probe     logistic regression on each layer alone -> which depth knows?
E4 english_only    same head trained on ASVspoof English only -> Arabic test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arabic_ssl import data  # noqa: E402
from arabic_ssl.head import LayerWeightedHead  # noqa: E402
from arabic_ssl.metrics import summary  # noqa: E402

CACHE = data.DATA_ROOT.parent / "cache"
RES = ROOT / "results"
SEED = 0
TESTS = {"aradf_dev": "Arabic dev (EGY/Gulf/MSA)", "arad": "Arabic RVC voice-conversion (unseen generator)",
         "aradf_track2": "Arabic, unseen channels (Track 2)", "asvspoof": "English ASVspoof 2019"}


def load(name):
    p = CACHE / f"ssl_{name}.npy"
    if not p.exists():
        return None, None
    meta = pd.DataFrame(json.loads(str(np.load(CACHE / f"{name}.npz")["meta"])))
    return np.load(p).astype(np.float32), meta


def train_head(X, y, epochs=40, lr=1e-3, bs=128, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X)); nv = max(len(X) // 10, 50)
    va, tr = perm[:nv], perm[nv:]
    head = LayerWeightedHead().set_norm(X[tr])
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), dtype=torch.float32)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    best, best_state, hist = 1.0, None, []
    for ep in range(epochs):
        head.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            x = Xt[b] + 0.05 * torch.randn_like(Xt[b]) * head.sd[0]   # noise in standardised space
            loss = F.binary_cross_entropy_with_logits(head(x), yt[b], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        v = summary(y[va], predict(head, X[va]))["eer"]
        hist.append(v)
        if v <= best:
            best, best_state = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, hist


@torch.no_grad()
def predict(head, X):
    head.eval()
    return head(torch.from_numpy(X)).numpy()


def main():
    tr_X, tr_m = load("aradf_train")
    tests = {k: load(k) for k in TESTS}
    tests = {k: v for k, v in tests.items() if v[0] is not None}
    y = tr_m["is_fake"].values
    metrics = {"train_size": int(len(y)), "train_dialects": tr_m["dialect"].value_counts().to_dict()}

    # E1
    head, hist = train_head(tr_X, y)
    (RES / "models").mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), RES / "models" / "head_main.pt")
    metrics["main"] = {"val_history": hist, "layer_weights": head.layer_weights().tolist()}
    for k, (X, m) in tests.items():
        s = predict(head, X)
        metrics["main"][k] = summary(m["is_fake"].values, s)
        pd.DataFrame({"id": m["id"], "dialect": m.get("dialect"), "is_fake": m["is_fake"], "score": s}
                     ).to_csv(RES / f"scores_main_{k}.csv", index=False)
        print(f"E1 main -> {k:13s} EER {metrics['main'][k]['eer']*100:6.2f}%")
    if "aradf_dev" in tests:
        X, m = tests["aradf_dev"]; s = predict(head, X)
        metrics["main"]["aradf_dev_per_dialect"] = {
            d: summary(g["is_fake"].values, s[g.index])["eer"] for d, g in m.groupby("dialect")}

    # E2 leave-one-dialect-out
    if "aradf_dev" in tests:
        X_dev, m_dev = tests["aradf_dev"]
        metrics["unseen_dialect"] = {}
        for d in sorted(tr_m["dialect"].unique()):
            if not (m_dev["dialect"] == d).any():      # no dev data for this dialect in our subset
                continue
            keep = (tr_m["dialect"] != d).values
            h, _ = train_head(tr_X[keep], y[keep])
            sel = (m_dev["dialect"] == d).values
            seen = summary(m_dev["is_fake"].values[sel], predict(head, X_dev[sel]))["eer"]
            unseen = summary(m_dev["is_fake"].values[sel], predict(h, X_dev[sel]))["eer"]
            metrics["unseen_dialect"][d] = {"seen_in_training": seen, "held_out": unseen}
            print(f"E2 dialect {d:5s}: seen {seen*100:5.2f}%   held-out {unseen*100:5.2f}%")

    # E3 layer probe
    if "aradf_dev" in tests:
        X_dev, m_dev = tests["aradf_dev"]
        probe = []
        sub = np.random.default_rng(SEED).choice(len(tr_X), min(4000, len(tr_X)), replace=False)
        for L in range(tr_X.shape[1]):
            sc = StandardScaler().fit(tr_X[sub, L])
            lr = LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced").fit(sc.transform(tr_X[sub, L]), y[sub])
            probe.append(summary(m_dev["is_fake"].values, lr.decision_function(sc.transform(X_dev[:, L])))["eer"])
        metrics["layer_probe_eer"] = probe
        print("E3 layer probe EER:", " ".join(f"{p*100:.1f}" for p in probe))

    # E4 english-only training
    if "asvspoof" in tests:
        X_en, m_en = tests["asvspoof"]
        tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=SEED).split(m_en, groups=m_en["speaker"]))
        h, _ = train_head(X_en[tr_i], m_en["is_fake"].values[tr_i])
        metrics["english_only"] = {"english_test": summary(m_en["is_fake"].values[te_i], predict(h, X_en[te_i]))}
        for k in ("aradf_dev", "arad", "aradf_track2"):
            if k in tests:
                metrics["english_only"][k] = summary(tests[k][1]["is_fake"].values, predict(h, tests[k][0]))
        print("E4 english-only:", {k: round(v["eer"] * 100, 2) for k, v in metrics["english_only"].items()})

    json.dump(metrics, open(RES / "metrics.json", "w"), indent=2)
    print("saved", RES / "metrics.json")


if __name__ == "__main__":
    main()
