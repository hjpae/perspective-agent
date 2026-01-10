# cear_pilot/analysis/figure_switch.py
# -*- coding: utf-8 -*-
"""
3-panel figure for regime-switch demo.

Panels:
  1) g-change score d(t)=||g - mean_pre|| with switch marker
  2) policy confidence: pi_max and pi_entropy with switch marker
  3) environment indicator: zone_id (and switched shading) with switch marker

Reads:
  <run_dir>/traj.parquet or traj.csv
  <run_dir>/meta.json (optional)

Writes:
  <run_dir>/figs/fig_switch.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_table(run_dir: Path):
    import pandas as pd

    pq = run_dir / "traj.parquet"
    csv = run_dir / "traj.csv"
    if pq.exists():
        return pd.read_parquet(pq), pq
    if csv.exists():
        return pd.read_csv(csv), csv
    raise FileNotFoundError(f"No traj.parquet or traj.csv found under: {run_dir}")


def _infer_cols(prefix: str, cols: list[str]) -> list[str]:
    out = [c for c in cols if c.startswith(prefix)]
    if not out:
        raise ValueError(f"No columns with prefix='{prefix}' found.")
    out = sorted(out, key=lambda x: int(x.split("_")[1]))
    return out


def _episode_slice(df, episode: int):
    if "episode" not in df.columns:
        return df.copy()
    d = df[df["episode"] == episode].copy()
    if len(d) == 0:
        eps = sorted(df["episode"].unique().tolist())
        raise ValueError(f"Episode {episode} not found. Available: {eps}")
    return d


def _compute_d(g: np.ndarray, t_switch: int, pre_window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      t: (T,)
      d: (T,)
      g_pre_mean: (G,)
    """
    T = g.shape[0]
    t = np.arange(T, dtype=int)

    if t_switch is not None and t_switch > 0:
        pre_end = min(t_switch, T)
    else:
        pre_end = min(pre_window, T)
    pre_end = max(pre_end, 1)

    g_pre = g[:pre_end]
    g_pre_mean = g_pre.mean(axis=0)
    d = np.linalg.norm(g - g_pre_mean[None, :], axis=1)
    return t, d, g_pre_mean


def _read_meta_switch(meta_path: Path) -> int:
    try:
        meta = json.loads(meta_path.read_text())
        return int(meta.get("t_switch", -1))
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--t_switch", type=int, default=-1, help="Override switch time (else use meta.json).")
    ap.add_argument("--pre_window", type=int, default=80, help="Used if t_switch < 0.")
    ap.add_argument("--title", type=str, default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    figs = run_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    df, src = _load_table(run_dir)
    dfe = _episode_slice(df, args.episode)

    # Determine t_switch
    t_switch = int(args.t_switch)
    meta_path = run_dir / "meta.json"
    if t_switch < 0 and meta_path.exists():
        t_switch = _read_meta_switch(meta_path)

    # Extract g matrix
    g_cols = _infer_cols("g_", list(dfe.columns))
    g = dfe[g_cols].to_numpy(dtype=np.float32)

    # Compute d(t)
    t, d, g_pre_mean = _compute_d(g=g, t_switch=t_switch, pre_window=int(args.pre_window))

    # Policy stats (optional but expected in new run_collect)
    has_pi = ("pi_max" in dfe.columns) and ("pi_entropy" in dfe.columns)
    pi_max = dfe["pi_max"].to_numpy(dtype=np.float32) if ("pi_max" in dfe.columns) else None
    pi_entropy = dfe["pi_entropy"].to_numpy(dtype=np.float32) if ("pi_entropy" in dfe.columns) else None

    # Env indicators
    zone_id = dfe["zone_id"].to_numpy(dtype=np.int32) if ("zone_id" in dfe.columns) else None
    switched = dfe["switched"].to_numpy(dtype=np.int32) if ("switched" in dfe.columns) else None

    # Build 3-panel plot
    fig = plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)

    # Panel 1: d(t)
    ax1.plot(t, d)
    if t_switch is not None and t_switch >= 0:
        ax1.axvline(x=t_switch, linestyle="--")
    ax1.set_ylabel("d(t)=||g-mean_pre||")
    ax1.set_title(args.title.strip() if args.title.strip() else f"Regime-switch demo (ep={args.episode}, t_switch={t_switch})")

    # Panel 2: policy confidence
    if has_pi:
        ax2.plot(t, pi_max, label="pi_max")
        ax2.plot(t, pi_entropy, label="pi_entropy")
        if t_switch is not None and t_switch >= 0:
            ax2.axvline(x=t_switch, linestyle="--")
        ax2.set_ylabel("policy")
        ax2.legend(loc="upper right")
    else:
        ax2.text(0.02, 0.8, "Missing policy columns: pi_max/pi_entropy", transform=ax2.transAxes)
        ax2.set_ylabel("policy")

    # Panel 3: env indicator
    if zone_id is not None:
        ax3.plot(t, zone_id)
        ax3.set_yticks([0, 1, 2])
        ax3.set_ylabel("zone_id")
    else:
        ax3.text(0.02, 0.8, "Missing zone_id column", transform=ax3.transAxes)
        ax3.set_ylabel("env")

    # Optional shading after switch
    if switched is not None and np.any(switched > 0):
        # Shade regions where switched==1
        on = np.where(switched > 0)[0]
        if len(on) > 0:
            start = int(on[0])
            ax1.axvspan(start, int(t[-1]), alpha=0.08)
            ax2.axvspan(start, int(t[-1]), alpha=0.08)
            ax3.axvspan(start, int(t[-1]), alpha=0.08)

    if t_switch is not None and t_switch >= 0:
        ax3.axvline(x=t_switch, linestyle="--")

    ax3.set_xlabel("t")

    out_path = figs / "fig_switch.png"
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Read: {src}")
    print(f"[OK] Saved: {out_path}")
    print(f"episode={args.episode} t_switch={t_switch}  (policy_cols={has_pi})")


if __name__ == "__main__":
    main()
