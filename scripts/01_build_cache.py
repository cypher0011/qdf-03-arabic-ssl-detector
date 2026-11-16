"""Decode, trim silence, and cache fixed-length int16 crops for fast training.

    python scripts/01_build_cache.py --corpus asvspoof
    python scripts/01_build_cache.py --corpus aradf_train      # 2500 per shard (EGY, Gulf, MSA)
    python scripts/01_build_cache.py --corpus aradf_dev
    python scripts/01_build_cache.py --corpus aradf_track2
    python scripts/01_build_cache.py --corpus arad --max-items 2000

Output: ../data/cache/<corpus>.npz with
    wav   int16 [N, SECONDS*16000]   silence-trimmed, peak-normalised, tiled to length
    meta  json list (id, is_fake, attack, speaker, dialect)

Silence is removed *before* caching. Model 1 showed that pause length alone
gives a misleadingly good EER on ASVspoof (the "silence trap").
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arabic_ssl import data  # noqa: E402

CACHE = data.DATA_ROOT.parent / "cache"
SECONDS = 4
N = SECONDS * data.SR


def prep(item: dict):
    try:
        it = data.decode(item)
    except Exception as e:
        print("skip", item.get("id"), e, file=sys.stderr)
        return None
    w = it.pop("wav")
    iv = librosa.effects.split(w, top_db=35, frame_length=400, hop_length=160)
    if len(iv):
        w = np.concatenate([w[s:e] for s, e in iv])
    if len(w) < 1600:
        return None
    w = w / (np.abs(w).max() + 1e-9) * 0.9
    if len(w) < N:
        w = np.tile(w, int(np.ceil(N / len(w))))
    start = (len(w) - N) // 2  # deterministic centre crop
    w = (w[start:start + N] * 32767).astype(np.int16)
    meta = {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in it.items()}
    return w, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["asvspoof", "aradf_train", "aradf_dev", "aradf_track2", "arad"])
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--max-per-shard", type=int, default=2500)
    ap.add_argument("--spoof-keep", type=float, default=0.35)
    a = ap.parse_args()
    k = a.max_per_shard
    src = {"asvspoof": lambda: data.iter_asvspoof(spoof_keep=a.spoof_keep),
           "aradf_train": lambda: data.iter_aradf("train-*.tar", "train", max_per_shard=k),
           "aradf_dev": lambda: data.iter_aradf("dev-*.tar", "dev", max_per_shard=k),
           "aradf_track2": lambda: data.iter_aradf("track2dev-*.tar", "track-2_development_test", max_per_shard=k),
           "arad": lambda: data.iter_arad("test", max_items=a.max_items)}[a.corpus]()
    wavs, metas = [], []
    with Pool(7) as p:
        for r in tqdm(p.imap(prep, islice(src, a.max_items), chunksize=16), desc=a.corpus):
            if r is not None:
                wavs.append(r[0]); metas.append(r[1])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"{a.corpus}.npz", wav=np.stack(wavs), meta=np.array(json.dumps(metas)))
    print(a.corpus, len(wavs), "fake:", sum(m["is_fake"] for m in metas))


if __name__ == "__main__":
    main()
