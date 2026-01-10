# cear_pilot/analysis/overlay_one_traj.py
# -*- coding: utf-8 -*-
"""
Overlay ONE episode trajectory (PCA(g)) from multiple runs into a single figure.

Expected per run_dir:
  - traj.parquet or traj.csv (must include 'episode' and 't')
  - embedding_pca.npy        (N, 2) from embed_latents.py

Usage:
  python -m cear_pilot.analysis.overlay_one_traj \
    --runs A=/path/to/runA flip=/path/to/runFlip flat=/path/to/runFlat \
    --episode 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

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


def parse_runs(kvs: list[str]) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for kv in kvs:
        if "=" not in kv:
            raise ValueError(f"--runs items must be like name=/path, got: {kv}")
        name, p = kv.split("=", 1)
        runs[name.strip()] = Path(p).expanduser().resolve()
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="List of name=run_dir (e.g., A=... flip=... flat=...)",
    )
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", type=str, default="", help="Output PNG path (default: <first_run>/figs/overlay_one_traj.png)")
    ap.add_argument("--connect", action="store_true", help="Draw lines connecting points (trajectory)")
    args = ap.parse_args()

    runs = parse_runs(args.runs)
    if len(runs) < 2:
        raise ValueError("Provide at least two runs to overlay.")

    # Load data per run
    series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # name -> (pc2d, t)
    for name, run_dir in runs.items():
        traj_path = find_traj(run_dir)
        df = load_table(traj_path)

        emb_path = run_dir / "embedding_pca.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embedding_pca.npy in {run_dir}. Run embed_latents first.")

        emb = np.load(emb_path).astype(np.float32)  # (N, 2)
        if emb.shape[0] != len(df):
            raise ValueError(f"Row mismatch: embedding has {emb.shape[0]} rows, traj has {len(df)} rows in {run_dir}")

        # Pick one episode
        ep = int(args.episode)
        mask = (df["episode"].astype(int).to_numpy() == ep)
        if mask.sum() == 0:
            raise ValueError(f"No rows for episode={ep} in {traj_path}")

        pc = emb[mask, :2]
        t = df.loc[mask, "t"].astype(int).to_numpy()
        # Sort by time to draw clean trajectory
        order = np.argsort(t)
        pc = pc[order]
        t = t[order]
        series[name] = (pc, t)

    # Output
    first_run = next(iter(runs.values()))
    out_path = Path(args.out) if str(args.out).strip() else (first_run / "figs" / "overlay_one_traj.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    plt.figure()
    for name, (pc, t) in series.items():
        x = pc[:, 0]
        y = pc[:, 1]
        # Points
        plt.plot(x, y, marker="o", linewidth=1.0 if args.connect else 0.0, markersize=2.0, label=name)

        # Optional: mark start/end
        plt.scatter([x[0]], [y[0]], marker="s", s=30)   # start
        plt.scatter([x[-1]], [y[-1]], marker="X", s=40) # end

    plt.title(f"PCA(g) overlay: episode={int(args.episode)} (shared PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
