# cear_pilot/analysis/phi/phi_r.py
# -*- coding: utf-8 -*-
"""
Φ_r estimator wrapper around Pigozzi & Levin's ΦID code (information.py).

Pipeline (matches their Methods exactly):
  1. (T, d) → (d, T) transpose
  2. corrected z-score (handles dead units with noise)
  3. lag-1 Gaussian MI matrix (Bonferroni-corrected significance)
  4. minimum information bipartition (Fiedler vector on MI graph)
  5. average within each partition → 2-d reduced system
  6. local ΦID decomposition (Möbius inversion over the lattice)
  7. extract Φ_r (sum of 8 specific atoms in PHIR_ATOMS)
  8. aggregate by median over the local Φ_r trajectory
"""

from typing import Optional, Tuple, Union

import numpy as np

from . import information as info


def phi_r_from_trajectory(
    Z: np.ndarray,
    lag: int = 1,
    return_local: bool = False,
    force_bipartition: Optional[Tuple[list, list]] = None,
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Compute Φ_r for a latent trajectory.

    Parameters
    ----------
    Z : np.ndarray, shape (T, d)
        Latent trajectory: T timesteps, d latent dimensions.
    lag : int
        Time lag for the MI matrix (Pigozzi & Levin use 1).
    return_local : bool
        If True, also return per-timestep local Φ_r trajectory.
    force_bipartition : (list, list), optional
        Override Fiedler bipartition with given (idx1, idx2). Useful for
        Φ_r([z,g]) where we want to force the partition along the z|g
        boundary rather than letting Fiedler choose. Indices refer to
        columns of Z.

    Returns
    -------
    phi_r_median : float
        Higher → more causally emergent.
    phi_r_local : np.ndarray, optional
        (T - lag,) local Φ_r if return_local=True.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D (T, d); got shape {Z.shape}")
    T, d = Z.shape
    if d < 2:
        raise ValueError(f"Need ≥2 latent units; got d={d}")
    if T < 10:
        raise ValueError(f"Trajectory too short; got T={T}")

    # (T, d) → (d, T) for Pigozzi convention (rows=units, cols=time)
    x = Z.T.copy()

    # 1. Standardize
    x = info.corrected_zscore(x, axis=1)

    # 2. Bipartition
    if force_bipartition is not None:
        idx1, idx2 = force_bipartition
        if (set(idx1) | set(idx2)) != set(range(d)) or set(idx1) & set(idx2):
            raise ValueError("force_bipartition must be a disjoint cover of range(d)")
    else:
        mi = info.mutual_information_matrix_fast(x, alpha=0.05, lag=lag, bonferonni=True)
        idx1, idx2 = info.minimum_information_bipartition(mi, noise=True)
        if len(idx1) == 0 or len(idx2) == 0:
            # Degenerate Fiedler split: fall back to halving by index
            idx1 = list(range(d // 2))
            idx2 = list(range(d // 2, d))

    # 3. Reduce to 2-d by averaging within each partition
    x_2d = np.vstack([
        x[idx1].mean(axis=0, keepdims=True),
        x[idx2].mean(axis=0, keepdims=True),
    ])

    # 4. Local ΦID decomposition
    lattice = info.local_phi_id(0, 1, x_2d)

    # 5. Φ_r
    phi_r_local = info.local_phi_r(lattice)
    phi_r_median = float(np.median(phi_r_local))

    if return_local:
        return phi_r_median, phi_r_local
    return phi_r_median


def phi_r_zgj(
    Z: np.ndarray,
    G: np.ndarray,
    lag: int = 1,
    force_zg_split_in_joint: bool = True,
) -> dict:
    """
    Convenience: compute Φ_r for z alone, g alone, and joint [z, g].

    Parameters
    ----------
    Z : (T, d_z)
    G : (T, d_g)
    force_zg_split_in_joint : bool
        If True, force the joint bipartition to be {z-cols | g-cols},
        which is the partition that makes ΔΦ_r interpretable as
        "synergy from z-g coupling". If False, Fiedler is free to choose
        (matches Pigozzi & Levin's method exactly).

    Returns
    -------
    dict with keys 'phi_r_z', 'phi_r_g', 'phi_r_joint', 'delta_phi_r'
    """
    if Z.shape[0] != G.shape[0]:
        raise ValueError(f"Z and G must align in time; got T={Z.shape[0]} vs {G.shape[0]}")
    d_z, d_g = Z.shape[1], G.shape[1]
    joint = np.hstack([Z, G])

    phi_z = phi_r_from_trajectory(Z, lag=lag)
    phi_g = phi_r_from_trajectory(G, lag=lag)

    if force_zg_split_in_joint:
        idx_z = list(range(d_z))
        idx_g = list(range(d_z, d_z + d_g))
        phi_joint = phi_r_from_trajectory(joint, lag=lag, force_bipartition=(idx_z, idx_g))
    else:
        phi_joint = phi_r_from_trajectory(joint, lag=lag)

    delta = phi_joint - max(phi_z, phi_g)
    return {
        "phi_r_z": phi_z,
        "phi_r_g": phi_g,
        "phi_r_joint": phi_joint,
        "delta_phi_r": delta,
    }
