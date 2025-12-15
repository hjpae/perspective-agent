# cear_pilot/analysis/figure_attractor.py
# -*- coding: utf-8 -*-
"""
Figure A: Attractor-ish visualization
- Scatter of PCA(g) colored by zone_id
- Optional short trajectory lines within each episode

Saves: figs/fig_attractor.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--lines", action="store_true", help="Draw per-episode trajectory lines")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    traj_path = find_traj(run_dir)
    df = load_table(traj_path)

    emb_path = run_dir / "embedding_pca.npy"
    if not emb_path.exists():
        raise FileNotFoundError("Run embed_latents.py first to create embedding_pca.npy")

    emb = np.load(emb_path)  # (N,2)
    zone = df["zone_id"].to_numpy()

    figdir = run_dir / "figs"
    figdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.scatter(emb[:, 0], emb[:, 1], c=zone, s=6)

    if args.lines and "episode" in df.columns:
        # Draw sparse lines to reduce clutter
        for ep in df["episode"].unique():
            sub = df[df["episode"] == ep]
            idx = sub.index.to_numpy()
            if idx.size < 2:
                continue
            e = emb[idx]
            # downsample
            e = e[::3]
            plt.plot(e[:, 0], e[:, 1], linewidth=0.8)

    plt.title("PCA(g): zone-colored")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    out = figdir / "fig_attractor.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
