#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rq1_pre_post.py
---------------
Pre- vs. post-switch Φ_r decomposition on the regime-switching replay.

Splits each episode at t = T_SWITCH and computes Φ_r(z), Φ_r(g),
Φ_r([z,g]), ΔΦ_r on the two halves separately. The key question:
  Does Φ_r(g) shift across the regime boundary?
  - If yes  → g carries history-sensitive slow dynamics  (our claim ✓)
  - If no   → Φ_r(g) is dominated by g's marginal shape  (claim weakens)

Usage:
  PYTHONPATH=. python rq1_pre_post.py
  PYTHONPATH=. python rq1_pre_post.py outputs/replay_seed1_p20/traj.parquet 120
"""

import sys
import numpy as np
import pandas as pd

from cear_pilot.analysis.phi.phi_r import phi_r_zgj


def main(traj_path: str, t_switch: int):
    df = pd.read_parquet(traj_path)
    z_cols = sorted([c for c in df.columns if c.startswith("z_")],
                    key=lambda c: int(c.split("_")[1]))
    g_cols = sorted([c for c in df.columns if c.startswith("g_")],
                    key=lambda c: int(c.split("_")[1]))

    n_eps = df.episode.nunique()
    print(f"\n{'='*72}")
    print(f"Pre/post-switch Φ_r decomposition")
    print(f"{'='*72}")
    print(f"  traj:        {traj_path}")
    print(f"  episodes:    {n_eps}")
    print(f"  t_switch:    {t_switch}")
    print(f"  z_dim={len(z_cols)}, g_dim={len(g_cols)}")
    print(f"{'='*72}\n")

    rows = []
    for ep_id, ep in df.groupby("episode"):
        pre = ep[ep.t < t_switch]
        post = ep[ep.t >= t_switch]
        if len(pre) < 30 or len(post) < 30:
            print(f"  [skip] ep {ep_id}: pre={len(pre)}, post={len(post)}")
            continue
        pre_out = phi_r_zgj(pre[z_cols].values, pre[g_cols].values,
                            force_zg_split_in_joint=True)
        post_out = phi_r_zgj(post[z_cols].values, post[g_cols].values,
                             force_zg_split_in_joint=True)
        rows.append({
            "episode": int(ep_id),
            "T_pre": len(pre), "T_post": len(post),
            "phi_z_pre": pre_out["phi_r_z"], "phi_z_post": post_out["phi_r_z"],
            "phi_g_pre": pre_out["phi_r_g"], "phi_g_post": post_out["phi_r_g"],
            "phi_j_pre": pre_out["phi_r_joint"], "phi_j_post": post_out["phi_r_joint"],
            "d_pre": pre_out["delta_phi_r"], "d_post": post_out["delta_phi_r"],
        })

    res = pd.DataFrame(rows)
    if len(res) == 0:
        print("No usable episodes.")
        return

    # --- Per-episode table ---
    print("Per-episode Φ_r values (pre | post):\n")
    hdr = (f"  {'ep':>3} | {'Φ_r(z)':>15} | {'Φ_r(g)':>15} | "
           f"{'Φ_r([z,g])':>15} | {'ΔΦ_r':>15}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in res.iterrows():
        print(f"  {int(r.episode):>3} | "
              f"{r.phi_z_pre:>+7.3f}|{r.phi_z_post:>+7.3f} | "
              f"{r.phi_g_pre:>+7.3f}|{r.phi_g_post:>+7.3f} | "
              f"{r.phi_j_pre:>+7.3f}|{r.phi_j_post:>+7.3f} | "
              f"{r.d_pre:>+7.3f}|{r.d_post:>+7.3f}")

    # --- Aggregate stats ---
    print("\n" + "="*72)
    print("AGGREGATE: post − pre (mean across episodes)")
    print("="*72)

    metrics = [("Φ_r(z)", "phi_z"), ("Φ_r(g)", "phi_g"),
               ("Φ_r([z,g])", "phi_j"), ("ΔΦ_r", "d")]

    print(f"  {'metric':<12} | {'pre mean':>10} | {'post mean':>10} | "
          f"{'Δ (post-pre)':>14} | {'paired t':>10} | {'p':>8}")
    print("  " + "-"*72)
    from scipy.stats import ttest_rel, wilcoxon
    for label, key in metrics:
        pre_vals = res[f"{key}_pre"].values
        post_vals = res[f"{key}_post"].values
        diff = post_vals - pre_vals
        t_stat, p_val = ttest_rel(post_vals, pre_vals)
        sig = " *" if p_val < 0.05 else ""
        print(f"  {label:<12} | {pre_vals.mean():>+10.4f} | {post_vals.mean():>+10.4f} | "
              f"{diff.mean():>+14.4f} | {t_stat:>+10.3f} | {p_val:>8.4f}{sig}")

    # --- Interpretation guide ---
    print("\n" + "="*72)
    print("READING THIS")
    print("="*72)
    delta_g = res.phi_g_post.mean() - res.phi_g_pre.mean()
    delta_z = res.phi_z_post.mean() - res.phi_z_pre.mean()
    print(f"  Φ_r(g) change across switch: {delta_g:+.4f}")
    print(f"  Φ_r(z) change across switch: {delta_z:+.4f}")
    print()
    if abs(delta_g) > 0.1 and abs(delta_g) > 3 * abs(delta_z):
        print("  → g responds to the regime switch much more than z does.")
        print("    This is consistent with g carrying history-sensitive slow")
        print("    dynamics that perceptual encoding (z) does not.")
    elif abs(delta_g) < 0.05:
        print("  → g shows little response to the regime switch.")
        print("    This weakens the 'history-sensitive' claim — investigate")
        print("    whether Φ_r(g) is dominated by g's marginal shape rather")
        print("    than its temporal structure.")
    else:
        print("  → Mixed signal; both z and g shift. Inspect per-episode")
        print("    pattern above to see if shifts are consistent in direction.")

    # Save for further plotting
    out_csv = traj_path.replace(".parquet", "_pre_post_phir.csv")
    res.to_csv(out_csv, index=False)
    print(f"\n  Saved per-episode CSV → {out_csv}")
    print("="*72)


if __name__ == "__main__":
    traj = sys.argv[1] if len(sys.argv) > 1 else "outputs/replay_seed1_p20/traj.parquet"
    t_switch = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    main(traj, t_switch)
