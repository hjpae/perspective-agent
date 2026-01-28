# cear_pilot/analysis/pca_z.py
# -*- coding: utf-8 -*-
"""
PCA on z latents from run logs.

Usage:
  python -m cear_pilot.analysis.pca_z --run_dir outputs/runs/20260127_123456 --color zone_id --lines

What it does:
- loads traj.parquet or traj.csv in run_dir
- finds z_* columns
- runs PCA (SVD) after standardization
- saves:
    run_dir/figs/z_pca_scatter.png
    run_dir/figs/z_pca_lines.png (optional)
    run_dir/figs/z_pca_meta.txt (EVR, dims, etc.)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_traj(run_dir: Path) -> pd.DataFrame:
    traj_parq = run_dir / "traj.parquet"
    traj_csv = run_dir / "traj.csv"
    if traj_parq.exists():
        return pd.read_parquet(traj_parq)
    if traj_csv.exists():
        return pd.read_csv(traj_csv)
    raise FileNotFoundError(f"traj.parquet or traj.csv not found under: {run_dir}")


def pca_svd(X: np.ndarray):
    """
    X: (N, D) standardized
    Returns:
      PC scores (N, D), EVR (D,), Vt (D, D)
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    PC = X @ Vt.T
    eigvals = (S ** 2) / max(1, (X.shape[0] - 1))
    evr = eigvals / (eigvals.sum() + 1e-12)
    return PC, evr, Vt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--color", type=str, default="zone_id",
                    help="column name for coloring (default: zone_id). Use '' for no color.")
    ap.add_argument("--lines", action="store_true",
                    help="also draw per-episode line trajectories if episode column exists")
    ap.add_argument("--max_points", type=int, default=50000,
                    help="subsample points for scatter if too many (default 50000)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = load_traj(run_dir)

    z_cols = [c for c in df.columns if c.startswith("z_")]
    if len(z_cols) == 0:
        raise RuntimeError(
            "No z_* columns found in traj. "
            "Check that your collector logs out['z'] into row['z_i']."
        )

    # (optional) subsample for scatter
    if len(df) > int(args.max_points):
        df_plot = df.sample(n=int(args.max_points), random_state=0).sort_index()
    else:
        df_plot = df

    Z = df_plot[z_cols].to_numpy(dtype=np.float32)
    Zm = Z.mean(axis=0, keepdims=True)
    Zs = Z.std(axis=0, keepdims=True) + 1e-8
    Zstd = (Z - Zm) / Zs

    PC, evr, _ = pca_svd(Zstd)
    pc1, pc2 = PC[:, 0], PC[:, 1]

    figs_dir = run_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # -------- scatter --------
    plt.figure()
    color_col = str(args.color).strip()
    if color_col == "":
        plt.scatter(pc1, pc2, s=6)
    else:
        if color_col in df_plot.columns:
            c = df_plot[color_col].to_numpy()
            plt.scatter(pc1, pc2, c=c, s=6)
        else:
            plt.scatter(pc1, pc2, s=6)
            print(f"[warn] color column '{color_col}' not found; plotting without color.")

    plt.xlabel(f"PC1 (EVR={float(evr[0]):.3f})")
    plt.ylabel(f"PC2 (EVR={float(evr[1]):.3f})")
    plt.title("z PCA (scatter)")
    plt.tight_layout()
    out_scatter = figs_dir / "z_pca_scatter.png"
    plt.savefig(out_scatter, dpi=200)
    plt.close()
    print("Saved:", out_scatter)

    # -------- optional per-episode lines --------
    out_lines = None
    if args.lines and ("episode" in df.columns):
        # Use full df (not subsampled) for trajectory continuity, but cap episodes.
        df_full = df.sort_values(["episode", "step"]) if "step" in df.columns else df.sort_values(["episode"])
        # recompute PCA projection using same mean/std + Vt from subsample PCA is ideal,
        # but for simplicity, we recompute on full standardized and reuse Vt requires returning Vt.
        # We'll recompute PCA on full to keep this module self-contained.
        Zf = df_full[z_cols].to_numpy(dtype=np.float32)
        Zfstd = (Zf - Zf.mean(axis=0, keepdims=True)) / (Zf.std(axis=0, keepdims=True) + 1e-8)
        PCf, evrf, _ = pca_svd(Zfstd)

        pc1f, pc2f = PCf[:, 0], PCf[:, 1]
        plt.figure()

        eps = sorted(df_full["episode"].unique().tolist())
        eps = eps[:10]  # first 10 episodes only to avoid clutter
        for ep in eps:
            m = (df_full["episode"].to_numpy() == ep)
            plt.plot(pc1f[m], pc2f[m], linewidth=1.0, alpha=0.8, label=f"ep{ep}")

        plt.xlabel(f"PC1 (EVR={float(evrf[0]):.3f})")
        plt.ylabel(f"PC2 (EVR={float(evrf[1]):.3f})")
        plt.title("z PCA (trajectories; first episodes)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_lines = figs_dir / "z_pca_lines.png"
        plt.savefig(out_lines, dpi=200)
        plt.close()
        print("Saved:", out_lines)

    # -------- meta --------
    meta_txt = figs_dir / "z_pca_meta.txt"
    meta_txt.write_text(
        "\n".join([
            f"run_dir: {run_dir}",
            f"z_dims: {len(z_cols)}",
            f"points_used_for_scatter: {len(df_plot)} / {len(df)}",
            f"EVR_PC1: {float(evr[0]):.6f}",
            f"EVR_PC2: {float(evr[1]):.6f}",
            f"saved_scatter: {out_scatter.name}",
            f"saved_lines: {out_lines.name if out_lines else 'None'}",
        ])
    )
    print("Saved:", meta_txt)


if __name__ == "__main__":
    main()
