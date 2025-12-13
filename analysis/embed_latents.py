# cear_pilot/analysis/embed_latents.py
# -*- coding: utf-8 -*-
"""
Fit PCA on g vectors and save embeddings.

Input:  traj.parquet or traj.csv
Output:
  outputs/.../embedding_pca.npy
  outputs/.../pca_components.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_table(traj_path: Path):
    import pandas as pd
    if traj_path.suffix == ".parquet":
        return pd.read_parquet(traj_path)
    return pd.read_csv(traj_path)


def find_traj(run_dir: Path) -> Path:
    p1 = run_dir / "traj.parquet"
    p2 = run_dir / "traj.csv"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"No traj.parquet or traj.csv in {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--k", type=int, default=2, help="PCA dims")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    traj_path = find_traj(run_dir)
    df = load_table(traj_path)

    g_cols = [c for c in df.columns if c.startswith("g_")]
    G = df[g_cols].to_numpy(dtype=np.float32)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=args.k)
    emb = pca.fit_transform(G).astype(np.float32)

    np.save(run_dir / "embedding_pca.npy", emb)
    np.save(run_dir / "pca_components.npy", pca.components_.astype(np.float32))
    np.save(run_dir / "pca_explained_var.npy", pca.explained_variance_ratio_.astype(np.float32))

    print(f"Saved PCA embedding to: {run_dir / 'embedding_pca.npy'}")
    print(f"Traj: {traj_path}")


if __name__ == "__main__":
    main()
