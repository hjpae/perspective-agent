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


def _parse_eps(eps_str: str):
    # "0,10,20" -> [0,10,20]
    return [int(x.strip()) for x in eps_str.split(",") if x.strip() != ""]

## ver1. pick only selective episodes (ep0, 10, 20)
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--run_dir", type=str, required=True)
#     ap.add_argument("--lines", action="store_true", help="Draw trajectory lines for selected episodes")
#     ap.add_argument(
#         "--eps",
#         type=str,
#         default="0,10,20",
#         help="Comma-separated episode ids to plot trajectories for (default: 0,10,20)",
#     )
#     ap.add_argument("--downsample", type=int, default=3, help="Downsample factor for trajectory plotting")
#     args = ap.parse_args()

#     run_dir = Path(args.run_dir)
#     traj_path = find_traj(run_dir)
#     df = load_table(traj_path)

#     emb_path = run_dir / "embedding_pca.npy"
#     if not emb_path.exists():
#         raise FileNotFoundError("Run embed_latents.py first to create embedding_pca.npy")

#     emb = np.load(emb_path)  # (N,2)

#     # Required columns
#     if "zone_id" not in df.columns:
#         raise KeyError("traj must contain column 'zone_id'")
#     zone = df["zone_id"].to_numpy()

#     figdir = run_dir / "figs"
#     figdir.mkdir(parents=True, exist_ok=True)

#     plt.figure(figsize=(7.2, 6.2))
#     plt.scatter(emb[:, 0], emb[:, 1], c=zone, s=6, alpha=0.9)

#     # Trajectories: only plot selected episodes
#     if args.lines and "episode" in df.columns:
#         all_eps = sorted(df["episode"].unique().tolist())
#         target_eps = _parse_eps(args.eps)

#         # Fallback if requested eps not present:
#         present = [e for e in target_eps if e in all_eps]
#         if len(present) == 0 and len(all_eps) > 0:
#             # pick first / middle / last
#             mid = all_eps[len(all_eps) // 2]
#             present = [all_eps[0], mid, all_eps[-1]]

#         for ep in present:
#             sub = df[df["episode"] == ep]

#             # Use row indices to select the aligned embedding rows
#             idx = sub.index.to_numpy()
#             if idx.size < 2:
#                 continue

#             e = emb[idx]

#             # Downsample to reduce clutter
#             ds = max(1, int(args.downsample))
#             e_plot = e[::ds]

#             # Draw trajectory (neutral color so it doesn't fight zone colors)
#             plt.plot(e_plot[:, 0], e_plot[:, 1], linewidth=1.8, alpha=0.95, color="black")

#             # Mark start/end for readability
#             plt.scatter(e_plot[0, 0], e_plot[0, 1], s=35, marker="o", color="black")
#             plt.scatter(e_plot[-1, 0], e_plot[-1, 1], s=55, marker="*", color="black")

#             # Label episode near the start
#             plt.text(e_plot[0, 0], e_plot[0, 1], f" ep{ep}", fontsize=9, color="black")

#     plt.title("PCA(g): zone-colored + selected episode trajectories")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")

#     out = figdir / "fig_attractor.png"
#     plt.savefig(out, dpi=220, bbox_inches="tight")
#     print(f"Saved: {out}")


## ver2. draw 30 different trajectories per episodes 
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
    #plt.scatter(emb[:, 0], emb[:, 1], c=zone, s=6) 
    plt.scatter(emb[:, 0], emb[:, 1], s=6, color="yellow") # without PCA clustering color

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

    plt.title("PCA(g): different color per episode (attractor well)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    out = figdir / "fig_attractor.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
