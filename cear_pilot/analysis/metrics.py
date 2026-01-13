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
from typing import Optional, Tuple, Dict, List, Any

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


def detect_delay_quantile(
    score: np.ndarray,
    switch_t: int,
    pre_window: int = 80,
    alpha: float = 0.05,
    consec: int = 3,
) -> Optional[int]:
    """
    Change-point detection delay.
    threshold = (1-alpha) quantile of pre-window score.
    delay = first t>=switch_t where score[t:t+consec] all exceed threshold.

    Returns delay steps, or None if not detected.
    """
    T = len(score)
    a = max(0, switch_t - pre_window)
    b = max(0, switch_t)
    if b - a < 10:
        return None

    thr = float(np.quantile(score[a:b], 1.0 - alpha))
    for t in range(switch_t, T - consec + 1):
        if np.all(score[t:t+consec] > thr):
            return int(t - switch_t)
    return None


def hysteresis_area(
    score: np.ndarray,
    regime: np.ndarray,
    switches: np.ndarray,
    L: int = 60,
) -> Dict[str, Any]:
    """
    Compute hysteresis (A->B vs B->A) after warmup using local windows.

    - regime[t] in {0,1}
    - switches[t]=1 at the switch time (same length as score)
    - For each switch time t0, take window [t0, t0+L)
      and collect score segments separately for A->B and B->A.
    - Mean trajectories m_up, m_dn; area = mean(|m_up - m_dn|)

    Returns dict with area and mean curves.
    """
    T = len(score)
    idx = np.where(switches.astype(int) == 1)[0].tolist()
    seg_up = []  # 0->1
    seg_dn = []  # 1->0

    for t0 in idx:
        if t0 + L > T:
            continue
        r0 = int(regime[t0-1]) if t0 - 1 >= 0 else int(regime[t0])
        r1 = int(regime[t0])
        seg = score[t0:t0+L].astype(np.float32)

        if r0 == 0 and r1 == 1:
            seg_up.append(seg)
        elif r0 == 1 and r1 == 0:
            seg_dn.append(seg)

    def mean_or_none(segs: List[np.ndarray]) -> Optional[np.ndarray]:
        if len(segs) == 0:
            return None
        return np.stack(segs, axis=0).mean(axis=0)

    m_up = mean_or_none(seg_up)
    m_dn = mean_or_none(seg_dn)

    out: Dict[str, Any] = {
        "n_up": len(seg_up),
        "n_dn": len(seg_dn),
        "m_up": m_up,
        "m_dn": m_dn,
        "area": None,
    }
    if m_up is not None and m_dn is not None:
        out["area"] = float(np.mean(np.abs(m_up - m_dn)))
    return out
