# cear_pilot/analysis/figure_perturb.py
# -*- coding: utf-8 -*-
"""
Figure C: Perturbation + recovery
- Plot ||g_t - g_pre_mean|| over time
- Mark perturb time
- Print recovery time estimate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cear_pilot.analysis.metrics import recovery_time


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
    ap.add_argument("--t0", type=int, default=-1, help="Perturb time (if -1, infer from 'perturbed' column)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = load_table(find_traj(run_dir))

    g_cols = [c for c in df.columns if c.startswith("g_")]
    G = df[g_cols].to_numpy(dtype=np.float32)
    t = df["t"].to_numpy(dtype=int)

    if args.t0 >= 0:
        t0 = args.t0
    else:
        if "perturbed" in df.columns and df["perturbed"].sum() > 0:
            t0 = int(df[df["perturbed"] == 1]["t"].iloc[0])
        else:
            t0 = int(t[len(t) // 2])

    # pre mean
    w = 20
    a = max(0, t0 - w)
    b = max(0, t0)
    pre = G[a:b] if b > a else G[:max(1, t0)]
    mu = pre.mean(axis=0)

    dist = np.linalg.norm(G - mu[None, :], axis=-1)

    rt = recovery_time(G, t0=t0, window=20, threshold=0.15)

    figdir = run_dir / "figs"
    figdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(t, dist)
    plt.axvline(t0, linewidth=1.0)
    plt.title(f"Perturb + recovery (recovery_time={rt})")
    plt.xlabel("t")
    plt.ylabel("||g - pre_mean||")

    out = figdir / "fig_perturb.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"t0={t0}, recovery_time={rt}")


if __name__ == "__main__":
    main()
