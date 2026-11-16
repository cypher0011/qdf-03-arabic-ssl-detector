"""README figures for Model 3."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "assets" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
REAL, FAKE, GREY, GREEN, PURPLE, AMBER = "#2563eb", "#dc2626", "#6b7280", "#16a34a", "#9333ea", "#f59e0b"
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11})
NAMES = {"aradf_dev": "ArA-DF dev\n(Egyptian + Gulf)", "arad": "ArAD\n(RVC voice conv.,\nunseen generator)",
         "aradf_track2": "ArA-DF Track 2\n(unseen channels)", "asvspoof": "ASVspoof 2019\n(English)"}


def save(fig, name):
    fig.savefig(FIG / name, facecolor="white"); plt.close(fig); print("  ", name)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 2.6)); ax.axis("off")
    blocks = [("4 s audio\n16 kHz", GREY, 1.3), ("mHuBERT-147\nFROZEN\n95M params\n147 languages", PURPLE, 1.7),
              ("13 hidden layers\nmean + std\nover time", "#0ea5e9", 1.6), ("learned\nlayer weights\n(softmax)", AMBER, 1.5),
              ("LayerNorm\nMLP 1536-256-1\n~0.4M params", GREEN, 1.6), ("fake score", FAKE, 1.2)]
    x = 0
    for i, (t, c, w) in enumerate(blocks):
        ax.add_patch(FancyBboxPatch((x, 0), w, 1, boxstyle="round,pad=0.05", fc=c, ec="none"))
        ax.text(x + w / 2, 0.5, t, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        if i < len(blocks) - 1:
            ax.annotate("", (x + w + 0.28, 0.5), (x + w + 0.05, 0.5), arrowprops=dict(arrowstyle="->", lw=1.6))
        x += w + 0.35
    ax.text(1.65 + 0.85, -0.25, "trained: 0 params", ha="center", fontsize=8, color=PURPLE)
    ax.text(x - 1.55 - 1.6 - 0.35 - 0.35 + 0.8, -0.25, "trained: everything here (< 0.5M)", ha="left", fontsize=8, color=GREEN)
    ax.set_xlim(-0.1, x); ax.set_ylim(-0.4, 1.1)
    save(fig, "architecture.png")


def fig_layers(m):
    probe = np.array(m["layer_probe_eer"]) * 100
    w = np.array(m["main"]["layer_weights"])
    L = np.arange(len(probe))
    fig, ax = plt.subplots(figsize=(8.5, 3.6)); ax2 = ax.twinx()
    ax2.bar(L, w, color=AMBER, alpha=0.35, label="learned layer weight")
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.plot(L, probe, "-o", color=PURPLE, lw=2, label="linear probe EER (one layer alone)")
    best = int(np.argmin(probe))
    ax.annotate(f"best single layer: {best}\nEER {probe[best]:.1f}%", (best, probe[best]), (best + 2.2, probe[best] + 1.6),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_xticks(L); ax.set_xticklabels(["CNN"] + [str(i) for i in range(1, len(L))])
    ax.set_xlabel("encoder layer (CNN output, then transformer layers 1-12)")
    ax.set_ylabel("EER (%)", color=PURPLE); ax2.set_ylabel("weight", color="#b45309"); ax2.spines["right"].set_visible(True)
    ax.set_title("Which depth of a multilingual speech model 'knows' about deepfakes?", loc="left")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=8)
    save(fig, "layer_analysis.png")


def fig_testsets(m):
    keys = [k for k in NAMES if k in m["main"]]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4))
    b1 = ax.bar(x - w / 2, [m["main"][k]["eer"] * 100 for k in keys], w, color=GREEN, label="trained on Arabic (Model 3)")
    if "english_only" in m:
        b2 = ax.bar(x + w / 2, [m["english_only"].get(k if k != "asvspoof" else "english_test", {"eer": np.nan})["eer"] * 100
                                for k in keys], w, color=REAL, label="same model trained on English only")
    else:
        b2 = []
    for bars in (b1, b2):
        for r in bars:
            if not np.isnan(r.get_height()):
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.8, f"{r.get_height():.1f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([NAMES[k] for k in keys], fontsize=8.5); ax.set_ylabel("EER (%)")
    ax.set_ylim(0, 55); ax.legend(frameon=False, loc="upper left", ncol=2); ax.set_title("Model 3 on four test sets", loc="left")
    save(fig, "test_sets.png")


def fig_dialect(m):
    if "unseen_dialect" not in m:
        return
    d = list(m["unseen_dialect"]); x = np.arange(len(d)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.8))
    seen = [m["unseen_dialect"][k]["seen_in_training"] * 100 for k in d]
    held = [m["unseen_dialect"][k]["held_out"] * 100 for k in d]
    b1 = ax.bar(x - w / 2, seen, w, color=GREEN, label="dialect seen in training")
    b2 = ax.bar(x + w / 2, held, w, color=AMBER, label="dialect NEVER seen in training")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + max(held) * 0.015, f"{r.get_height():.2f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([{"EGY": "Egyptian", "MSA": "MSA (Fusha)", "Gulf": "Gulf (Saudi)"}.get(k, k) for k in d])
    ax.set_ylabel("EER (%)"); ax.legend(frameon=False); ax.set_ylim(0, max(held) * 1.15)
    ax.set_title("Leave-one-dialect-out: does it transfer to an unseen dialect?", loc="left")
    save(fig, "unseen_dialect.png")


def fig_journey(m):
    """Arabic EER of Models 1-3, read from the sibling repositories when present."""
    rows = []
    p1 = ROOT.parent / "qdf-01-spectral-forensics" / "results" / "metrics.json"
    p2 = ROOT.parent / "qdf-02-spectrogram-lcnn" / "results" / "metrics_en.json"
    p2a = ROOT.parent / "qdf-02-spectrogram-lcnn" / "results" / "metrics_ar.json"
    if p1.exists() and "arabic_test" in json.load(open(p1))["forensic_main"]:
        rows.append(("Model 1\nforensic features\n(English-trained)", json.load(open(p1))["forensic_main"]["arabic_test"]["eer"], GREY))
    if p2.exists():
        rows.append(("Model 2\nspectrogram CNN\n(English-trained)", json.load(open(p2))["arabic_test"]["eer"], REAL))
    if p2a.exists():
        rows.append(("Model 2\nspectrogram CNN\n(Arabic-trained)", json.load(open(p2a))["arabic_test"]["eer"], "#60a5fa"))
    if "english_only" in m:
        rows.append(("Model 3 encoder\n(English-trained)", m["english_only"]["aradf_dev"]["eer"], "#a78bfa"))
    rows.append(("Model 3\nmHuBERT-147\n(Arabic-trained)", m["main"]["aradf_dev"]["eer"], GREEN))
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar([r[0] for r in rows], [r[1] * 100 for r in rows], color=[r[2] for r in rows])
    for r, (_, v, _) in zip(bars, rows):
        ax.text(r.get_x() + r.get_width() / 2, v * 100 + 0.8, f"{v*100:.1f}%", ha="center", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8); ax.set_ylabel("EER on Arabic (%)")
    ax.set_title("The journey so far: EER on the same Arabic test set (ArA-DF-2026 dev)", loc="left")
    save(fig, "journey.png")


def fig_det(m):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ticks = [0.001, 0.01, 0.05, 0.2, 0.5]
    ax.set_xticks(norm.ppf(ticks)); ax.set_yticks(norm.ppf(ticks))
    ax.set_xticklabels([f"{t*100:g}" for t in ticks]); ax.set_yticklabels([f"{t*100:g}" for t in ticks])
    ax.set_xlim(norm.ppf(0.0005), norm.ppf(0.6)); ax.set_ylim(norm.ppf(0.0005), norm.ppf(0.6))
    for k, c in zip(NAMES, (GREEN, PURPLE, AMBER, REAL)):
        p = RES / f"scores_main_{k}.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p); fpr, tpr, _ = roc_curve(s.is_fake, s.score); fnr = 1 - tpr; keep = (fpr > 0) & (fnr > 0)
        ax.plot(norm.ppf(fpr[keep]), norm.ppf(fnr[keep]), color=c, lw=2,
                label=f"{NAMES[k].splitlines()[0]}  {m['main'][k]['eer']*100:.1f}%")
    ax.plot([-5, 5], [-5, 5], ":", color=GREY, lw=0.8)
    ax.set_xlabel("False alarm rate (%)"); ax.set_ylabel("Miss rate (%)")
    ax.legend(frameon=False, fontsize=8); ax.set_title("DET curves, Model 3", loc="left")
    save(fig, "det_curves.png")


def fig_scores():
    p = RES / "scores_main_aradf_dev.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    fig, axes = plt.subplots(1, s["dialect"].nunique(), figsize=(12, 3), sharey=True)
    for ax, (d, g) in zip(np.atleast_1d(axes), s.groupby("dialect")):
        bins = np.linspace(s.score.quantile(0.005), s.score.quantile(0.995), 50)
        ax.hist(g[g.is_fake == 0].score, bins, color=REAL, alpha=0.6, density=True, label="real")
        ax.hist(g[g.is_fake == 1].score, bins, color=FAKE, alpha=0.6, density=True, label="fake")
        ax.set_title({"EGY": "Egyptian", "MSA": "MSA", "Gulf": "Gulf"}.get(d, d)); ax.set_yticks([]); ax.set_xlabel("score (logit)")
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.suptitle("Score distributions per dialect (ArA-DF dev)", x=0.01, ha="left", fontweight="bold")
    save(fig, "score_distributions.png")


def main():
    m = json.load(open(RES / "metrics.json"))
    fig_architecture(); fig_layers(m); fig_testsets(m); fig_dialect(m); fig_journey(m); fig_det(m); fig_scores()


if __name__ == "__main__":
    main()
