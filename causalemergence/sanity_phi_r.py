"""
sanity_phi_r.py — run after placing phi_lattice_22.pickle in
cear_pilot/analysis/phi/

Three checks:
  1. Pure noise → Φ_r ≈ 0
  2. Strongly coupled signal → Φ_r > 0
  3. Real AAAI checkpoint replay data → Φ_r(z), Φ_r(g), Φ_r([z,g]), ΔΦ_r

Usage:
  PYTHONPATH=. python sanity_phi_r.py outputs/replay_seed1_clean/traj.parquet
"""

import sys
import numpy as np
import pandas as pd

from cear_pilot.analysis.phi.phi_r import phi_r_from_trajectory, phi_r_zgj


def main(traj_path: str):
    rng = np.random.default_rng(0)
    T = 500

    # --- Check 1: pure noise should give Φ_r ≈ 0 ---
    Z_noise = rng.standard_normal((T, 12))
    phi_noise = phi_r_from_trajectory(Z_noise)
    print(f"[1] pure noise (T={T}, d=12):     Φ_r = {phi_noise:+.4f}   (expect ≈ 0)")

    # --- Check 2: strong shared dynamics should give Φ_r > 0 ---
    shared = np.cumsum(rng.standard_normal(T)) * 0.1
    Z_coupled = np.stack(
        [shared + 0.1 * rng.standard_normal(T) for _ in range(12)],
        axis=1,
    )
    phi_coupled = phi_r_from_trajectory(Z_coupled)
    print(f"[2] coupled signal (T={T}, d=12): Φ_r = {phi_coupled:+.4f}   (expect > 0)")

    # --- Check 3: real replay data ---
    df = pd.read_parquet(traj_path)
    z_cols = sorted([c for c in df.columns if c.startswith("z_")],
                    key=lambda c: int(c.split("_")[1]))
    g_cols = sorted([c for c in df.columns if c.startswith("g_")],
                    key=lambda c: int(c.split("_")[1]))
    print(f"\n[3] real AAAI replay: {traj_path}")
    print(f"    z_dim={len(z_cols)}, g_dim={len(g_cols)}, episodes={df.episode.nunique()}")

    # Per-episode Φ_r decomposition
    print(f"\n    {'episode':>8} | {'Φ_r(z)':>10} | {'Φ_r(g)':>10} | {'Φ_r([z,g])':>12} | {'ΔΦ_r':>10}")
    print(f"    {'-'*8} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*10}")
    deltas = []
    for ep_id, ep in df.groupby("episode"):
        Z = ep[z_cols].values
        G = ep[g_cols].values
        out = phi_r_zgj(Z, G, force_zg_split_in_joint=True)
        deltas.append(out["delta_phi_r"])
        print(f"    {ep_id:>8} | {out['phi_r_z']:>+10.4f} | {out['phi_r_g']:>+10.4f} "
              f"| {out['phi_r_joint']:>+12.4f} | {out['delta_phi_r']:>+10.4f}")
    print(f"    {'mean':>8} |            |            |              | "
          f"{np.mean(deltas):>+10.4f}")


if __name__ == "__main__":
    traj = sys.argv[1] if len(sys.argv) > 1 else "outputs/replay_seed1_clean/traj.parquet"
    main(traj)
