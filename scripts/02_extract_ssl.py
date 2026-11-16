"""Run the frozen mHuBERT-147 once over every cached corpus and store the
per-layer mean/std embeddings (float16) next to the audio cache.

    python scripts/02_extract_ssl.py aradf_train aradf_dev aradf_track2 arad asvspoof
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arabic_ssl import data  # noqa: E402
from arabic_ssl.backbone import Backbone, int16_to_float  # noqa: E402

CACHE = data.DATA_ROOT.parent / "cache"


def main(names: list[str], bs: int = 16) -> None:
    bb = Backbone()
    for name in names:
        src, dst = CACHE / f"{name}.npz", CACHE / f"ssl_{name}.npy"
        if not src.exists():
            print("missing", src); continue
        if dst.exists():
            print("exists", dst); continue
        wav = np.load(src)["wav"]
        out = np.zeros((len(wav), 13, 1536), dtype=np.float16)
        t0 = time.time()
        for i in tqdm(range(0, len(wav), bs), desc=name):
            out[i:i + bs] = bb.pooled(int16_to_float(wav[i:i + bs])).float().cpu().numpy()
        np.save(dst, out)
        print(f"{name}: {len(wav)} utts in {time.time()-t0:.0f}s -> {dst.name}")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ["aradf_train", "aradf_dev", "aradf_track2", "arad", "asvspoof"])
