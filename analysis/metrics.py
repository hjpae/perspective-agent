# cear_pilot/analysis/metrics.py
# -*- coding: utf-8 -*-
"""
Metrics for "order parameter" behavior:
- drift in g
- recovery time after perturbation
- silhouette score by zone (in embedding space)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def g_columns(df) -> list[str]:
    return [c for c in df.columns if c.startswith("g_")]


def s_columns(df) -> list[str]:
    return [c for c in df.columns if c.startswith("s_")]


def obs_columns(df) -> list[str]:
    return [c for c in df.columns if c.startswith("obs_")]


def drift_norm(G: np.ndarray) -> np.ndarray:
    """
    G: (T, D)
    returns stepwise ||g_t - g_{t-1}||, length T (with 0 at t=0)
    """
    d = np.zeros((G.shape[0],), dtype=np.float32)
    if G.shape[0] <= 1:
        return d
    d[1:] = np.linalg.norm(G[1:] - G[:-1], axis=-1)
    return d


def recovery_time(
    G: np.ndarray,
    t0: int,
    window: int = 20,
    threshold: float = 0.15,
) -> Optional[int]:
    """
    Define recovery as: distance to pre-perturb mean <= threshold * pre-perturb std
    using a pre window [t0-window, t0).
    Returns number of steps after t0 until recovered, or None.
    """
    T = G.shape[0]
    a = max(0, t0 - window)
    b = max(0, t0)

    if b - a < 5:
        return None

    pre = G[a:b]
    mu = pre.mean(axis=0)
    sig = pre.std(axis=0) + 1e-6

    # distance in standardized space
    def dist(g):
        return np.linalg.norm((g - mu) / sig)

    for t in range(t0, T):
        if dist(G[t]) <= threshold:
            return t - t0
    return None


def silhouette_by_zone(emb: np.ndarray, zone: np.ndarray) -> Optional[float]:
    """
    emb: (N, k), zone: (N,)
    """
    try:
        from sklearn.metrics import silhouette_score
        # Need at least 2 labels
        if len(np.unique(zone)) < 2:
            return None
        return float(silhouette_score(emb, zone))
    except Exception:
        return None
