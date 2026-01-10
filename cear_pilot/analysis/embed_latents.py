# cear_pilot/analysis/embed_latents.py
# -*- coding: utf-8 -*-
"""
Fit PCA on g vectors and save embeddings.

This script supports two modes:

1) Fit mode (default):
   - Fits PCA on the g_* columns of traj.{parquet|csv} under --run_dir
   - Saves:
       embedding_pca.npy          (N, k) PCA coordinates
       pca_components.npy         (k, D) PCA components
       pca_mean.npy               (D,)   PCA mean used for centering
       pca_explained_var.npy      (k,)   explained variance ratio (like sklearn)
   - Output files are saved inside --run_dir

2) Reuse mode (shared coordinate system across conditions):
   - If --pca_fit_dir is provided, this script loads PCA mean/components from that directory
     and only applies transform to the current run's G, without refitting PCA.
   - This makes PC axes comparable across different runs/conditions.

Notes:
- We avoid pickling sklearn objects for robustness; we store mean/components directly.
- Transform in reuse mode is equivalent to sklearn PCA.transform (without whitening):
      X_centered @ components.T
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

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


def extract_G(df) -> Tuple[np.ndarray, list[str]]:
    g_cols = [c for c in df.columns if c.startswith("g_")]
    if len(g_cols) == 0:
        raise ValueError("No g_* columns found. Make sure your collector logs g_i into traj.")
    G = df[g_cols].to_numpy(dtype=np.float32)
    return G, g_cols


def save_pca_artifacts(run_dir: Path, emb: np.ndarray, mean: np.ndarray, components: np.ndarray, evr: np.ndarray):
    np.save(run_dir / "embedding_pca.npy", emb.astype(np.float32))
    np.save(run_dir / "pca_mean.npy", mean.astype(np.float32))
    np.save(run_dir / "pca_components.npy", components.astype(np.float32))
    np.save(run_dir / "pca_explained_var.npy", evr.astype(np.float32))


def load_pca_artifacts(pca_fit_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PCA artifacts from pca_fit_dir:
      - pca_mean.npy
      - pca_components.npy
      - pca_explained_var.npy (optional; if missing, return zeros)
    """
    mean_path = pca_fit_dir / "pca_mean.npy"
    comp_path = pca_fit_dir / "pca_components.npy"
    evr_path = pca_fit_dir / "pca_explained_var.npy"

    if not mean_path.exists():
        raise FileNotFoundError(f"Missing PCA mean: {mean_path}")
    if not comp_path.exists():
        raise FileNotFoundError(f"Missing PCA components: {comp_path}")

    mean = np.load(mean_path).astype(np.float32)          # (D,)
    components = np.load(comp_path).astype(np.float32)    # (K, D)

    if evr_path.exists():
        evr = np.load(evr_path).astype(np.float32)
    else:
        evr = np.zeros((components.shape[0],), dtype=np.float32)

    return mean, components, evr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--k", type=int, default=2, help="PCA dims")
    ap.add_argument(
        "--pca_fit_dir",
        type=str,
        default="",
        help="If provided, reuse PCA (mean/components) from this directory instead of fitting here.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    traj_path = find_traj(run_dir)
    df = load_table(traj_path)

    G, g_cols = extract_G(df)
    D = G.shape[1]

    # Reuse mode: load mean/components and transform only
    if str(args.pca_fit_dir).strip():
        fit_dir = Path(args.pca_fit_dir)
        mean, components_full, evr_full = load_pca_artifacts(fit_dir)

        if mean.shape[0] != D or components_full.shape[1] != D:
            raise ValueError(
                f"PCA dim mismatch: current G has D={D}, "
                f"but fit PCA has mean D={mean.shape[0]} and comp D={components_full.shape[1]}.\n"
                f"Hint: ensure the same agent config / g_dim across runs."
            )

        # Use first k components (or all if k exceeds available)
        k_eff = min(int(args.k), int(components_full.shape[0]))
        components = components_full[:k_eff]          # (k_eff, D)
        evr = evr_full[:k_eff] if evr_full.shape[0] >= k_eff else np.zeros((k_eff,), dtype=np.float32)

        Xc = (G - mean[None, :]).astype(np.float32)   # (N, D)
        emb = (Xc @ components.T).astype(np.float32)  # (N, k_eff)

        # Save embedding in current run_dir, but DO NOT overwrite PCA artifacts
        np.save(run_dir / "embedding_pca.npy", emb)
        np.save(run_dir / "pca_explained_var.npy", evr)

        print(f"[OK] Reused PCA from: {fit_dir}")
        print(f"[OK] Saved embedding to: {run_dir / 'embedding_pca.npy'}  (k={k_eff})")
        print(f"[OK] Traj: {traj_path}")
        return

    # Fit mode: fit PCA here and save full artifacts
    from sklearn.decomposition import PCA

    k = int(args.k)
    pca = PCA(n_components=k)
    emb = pca.fit_transform(G).astype(np.float32)            # (N, k)
    components = pca.components_.astype(np.float32)          # (k, D)
    mean = pca.mean_.astype(np.float32)                      # (D,)
    evr = pca.explained_variance_ratio_.astype(np.float32)   # (k,)

    save_pca_artifacts(run_dir, emb=emb, mean=mean, components=components, evr=evr)

    print(f"[OK] Fit PCA in: {run_dir}")
    print(f"[OK] Saved embedding to: {run_dir / 'embedding_pca.npy'}  (k={k})")
    print(f"[OK] Saved PCA mean/components to: {run_dir}")
    print(f"[OK] Traj: {traj_path}")


if __name__ == "__main__":
    main()
