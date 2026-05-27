#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rq1_all_seeds.py
-----------------
Aggregate Φ_r analysis across all seeds and conditions.

For each (seed, condition) pair, computes per-episode Φ_r(z), Φ_r(g),
Φ_r([z,g]), ΔΦ_r. Then aggregates across seeds with proper statistics.

Two key questions answered:
  1. Is Φ_r(g) ≫ Φ_r(z) consistent across all seeds? (the main claim)
  2. Does Φ_r(g) drop after the regime switch consistently? (history sensitivity)

Usage:
  PYTHONPATH=. python rq1_all_seeds.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp, wilcoxon

from cear_pilot.analysis.phi.phi_r import phi_r_zgj

SEEDS = [1, 2, 3, 4, 5]
T_SWITCH = 120
OUTDIR = Path("outputs")


def episode_phir(ep_df, z_cols, g_cols):
    return phi_r_zgj(ep_df[z_cols].values, ep_df[g_cols].values,
                     force_zg_split_in_joint=True)


def process_run(traj_path, t_switch=None):
    """Return per-episode DataFrame with Φ_r values.
       If t_switch is given, splits each episode pre/post."""
    df = pd.read_parquet(traj_path)
    z_cols = sorted([c for c in df.columns if c.startswith("z_")],
                    key=lambda c: int(c.split("_")[1]))
    g_cols = sorted([c for c in df.columns if c.startswith("g_")],
                    key=lambda c: int(c.split("_")[1]))
    rows = []
    for ep_id, ep in df.groupby("episode"):
        if t_switch is None:
            out = episode_phir(ep, z_cols, g_cols)
            rows.append({"episode": int(ep_id), "phase": "full",
                         "T": len(ep), **out})
        else:
            pre = ep[ep.t < t_switch]
            post = ep[ep.t >= t_switch]
            if len(pre) < 30 or len(post) < 30:
                continue
            out_pre = episode_phir(pre, z_cols, g_cols)
            out_post = episode_phir(post, z_cols, g_cols)
            rows.append({"episode": int(ep_id), "phase": "pre",
                         "T": len(pre), **out_pre})
            rows.append({"episode": int(ep_id), "phase": "post",
                         "T": len(post), **out_post})
    return pd.DataFrame(rows)


def main():
    print(f"\n{'='*78}")
    print("RQ1: Φ_r decomposition — all seeds × {clean, p20}")
    print(f"{'='*78}\n")

    # --- collect ---
    all_clean, all_p20 = [], []
    for seed in SEEDS:
        clean_path = OUTDIR / f"replay_seed{seed}_clean" / "traj.parquet"
        p20_path = OUTDIR / f"replay_seed{seed}_p20" / "traj.parquet"

        if clean_path.exists():
            d = process_run(clean_path, t_switch=None)
            d["seed"] = seed
            all_clean.append(d)
            print(f"  seed {seed} clean: {len(d)} episodes")
        else:
            print(f"  [WARN] missing {clean_path}")

        if p20_path.exists():
            d = process_run(p20_path, t_switch=T_SWITCH)
            d["seed"] = seed
            all_p20.append(d)
            print(f"  seed {seed} p20:   {len(d)//2} episodes × pre/post")
        else:
            print(f"  [WARN] missing {p20_path}")

    if not all_clean or not all_p20:
        print("\nNot enough data. Run collect_all_seeds.sh first.")
        return

    clean = pd.concat(all_clean, ignore_index=True)
    p20 = pd.concat(all_p20, ignore_index=True)

    # ===========================================================
    # Result 1: Φ_r(g) ≫ Φ_r(z) — the main claim
    # ===========================================================
    print(f"\n{'='*78}")
    print("RESULT 1: Φ_r concentrates in g, not z   (clean condition)")
    print(f"{'='*78}")
    print(f"\n  Per-seed means (clean, n=10 ep each):\n")
    print(f"  {'seed':>4} | {'Φ_r(z)':>10} | {'Φ_r(g)':>10} | "
          f"{'ratio g/z':>10} | {'p (z<g)':>10}")
    print("  " + "-"*60)
    for seed, sub in clean.groupby("seed"):
        z_mean, g_mean = sub["phi_r_z"].mean(), sub["phi_r_g"].mean()
        # Paired test: Φ_r(g) > Φ_r(z) within each episode?
        _, p = wilcoxon(sub["phi_r_g"], sub["phi_r_z"],
                        alternative="greater")
        ratio = g_mean / max(z_mean, 1e-6)
        print(f"  {seed:>4} | {z_mean:>+10.4f} | {g_mean:>+10.4f} | "
              f"{ratio:>10.1f}x | {p:>10.4f}")

    z_all = clean["phi_r_z"].values
    g_all = clean["phi_r_g"].values
    diff = g_all - z_all
    t_pooled, p_pooled = ttest_1samp(diff, 0)
    print(f"\n  Pooled across seeds (n={len(diff)}):")
    print(f"    mean Φ_r(z) = {z_all.mean():+.4f}")
    print(f"    mean Φ_r(g) = {g_all.mean():+.4f}")
    print(f"    Φ_r(g) − Φ_r(z) = {diff.mean():+.4f}   (paired t={t_pooled:.2f}, "
          f"p={p_pooled:.2e})")

    # ===========================================================
    # Result 2: Φ_r(g) drops across regime switch — history sensitivity
    # ===========================================================
    print(f"\n{'='*78}")
    print("RESULT 2: Φ_r(g) is regime-sensitive   (p20 condition)")
    print(f"{'='*78}")

    # Pivot to wide format for paired comparison
    pre = p20[p20.phase == "pre"].set_index(["seed", "episode"])
    post = p20[p20.phase == "post"].set_index(["seed", "episode"])
    common = pre.index.intersection(post.index)
    pre, post = pre.loc[common], post.loc[common]

    print(f"\n  Per-seed Φ_r(g) change (post − pre):\n")
    print(f"  {'seed':>4} | {'pre':>10} | {'post':>10} | "
          f"{'Δ':>10} | {'p (drop)':>10}")
    print("  " + "-"*60)
    for seed in SEEDS:
        if seed not in pre.index.get_level_values("seed"):
            continue
        s_pre = pre.xs(seed, level="seed")["phi_r_g"]
        s_post = post.xs(seed, level="seed")["phi_r_g"]
        delta = s_post.values - s_pre.values
        _, p = wilcoxon(s_post, s_pre, alternative="less")
        print(f"  {seed:>4} | {s_pre.mean():>+10.4f} | {s_post.mean():>+10.4f} | "
              f"{delta.mean():>+10.4f} | {p:>10.4f}")

    # Pooled across all seeds
    d_g = post["phi_r_g"].values - pre["phi_r_g"].values
    d_z = post["phi_r_z"].values - pre["phi_r_z"].values
    t_g, p_g = ttest_1samp(d_g, 0)
    t_z, p_z = ttest_1samp(d_z, 0)
    cohens_d_g = d_g.mean() / d_g.std()
    cohens_d_z = d_z.mean() / d_z.std() if d_z.std() > 0 else float("nan")
    print(f"\n  Pooled (n={len(d_g)}):")
    print(f"    Φ_r(g): Δ = {d_g.mean():+.4f},   t = {t_g:.2f},   p = {p_g:.2e},   "
          f"Cohen's d = {cohens_d_g:.2f}")
    print(f"    Φ_r(z): Δ = {d_z.mean():+.4f},   t = {t_z:.2f},   p = {p_z:.2e},   "
          f"Cohen's d = {cohens_d_z:.2f}")
    print(f"\n  Asymmetry: |Δ Φ_r(g)| / |Δ Φ_r(z)| = "
          f"{abs(d_g.mean()) / max(abs(d_z.mean()), 1e-6):.1f}x")

    # ===========================================================
    # Save aggregated CSV for plotting
    # ===========================================================
    clean["condition"] = "clean"
    p20["condition"] = "p20"
    all_data = pd.concat([clean, p20], ignore_index=True)
    out_csv = OUTDIR / "rq1_aggregated.csv"
    all_data.to_csv(out_csv, index=False)
    print(f"\n  Saved aggregated CSV → {out_csv}")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
