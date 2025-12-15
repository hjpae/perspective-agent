# cear_pilot/analysis/figure_ablation.py
# -*- coding: utf-8 -*-
"""
Figure B: Ablation comparison (g_on vs g_off)
- Two panels: PCA scatter colored by zone
- Prints silhouette score if available
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cear_pilot.analysis.metrics import silhouette_by_zone


def load_table(traj_path: Path):
    import pandas as pd
    if traj_path.suffix == ".parquet":
        return pd.read_parquet(traj_path)
    return pd.read_csv(traj_path)


def find_traj(run_dir: Path) -> Path:
    for ext in [".parquet", ".csv"]:
        p = run_dir / f"traj{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No traj.parquet/csv in {run_dir}")


def load_emb(run_dir: Path) -> np.ndarray:
    p = run_dir / "embedding_pca.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run embed_latents.py first.")
    return np.load(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True, help="Ablation root containing g_on/ and g_off/")
    args = ap.parse_args()

    root = Path(args.root_dir)
    on_dir = root / "g_on"
    off_dir = root / "g_off"

    df_on = load_table(find_traj(on_dir))
    df_off = load_table(find_traj(off_dir))
    emb_on = load_emb(on_dir)
    emb_off = load_emb(off_dir)

    zone_on = df_on["zone_id"].to_numpy()
    zone_off = df_off["zone_id"].to_numpy()

    s_on = silhouette_by_zone(emb_on, zone_on)
    s_off = silhouette_by_zone(emb_off, zone_off)

    figdir = root / "figs"
    figdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(emb_on[:, 0], emb_on[:, 1], c=zone_on, s=6)
    plt.title(f"g ON (sil={s_on:.3f})" if s_on is not None else "g ON")
    plt.xlabel("PC1"); plt.ylabel("PC2")

    plt.subplot(1, 2, 2)
    plt.scatter(emb_off[:, 0], emb_off[:, 1], c=zone_off, s=6)
    plt.title(f"g OFF (sil={s_off:.3f})" if s_off is not None else "g OFF")
    plt.xlabel("PC1"); plt.ylabel("PC2")

    out = figdir / "fig_ablation.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
