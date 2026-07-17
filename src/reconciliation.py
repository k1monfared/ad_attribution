"""Hierarchical forecast reconciliation for multi-region attribution.

The additive quantity across the hierarchy is the cumulative extra streams
caused by the campaign. The total (all treated regions) sits at the top and the
regions sit at the bottom, linked by a summing matrix S that maps the bottom
level to every node:

    all_levels = S @ bottom_level

Estimating each node on its own gives incoherent numbers: the independent
per-region synthetic-control effects do not sum to the independent aggregate
synthetic-control effect, because they are separate estimates at different
levels with different fit quality. Reconciliation projects the incoherent base
estimates onto the space of coherent ones.

Optimal (MinT) reconciliation, Wickramasuriya, Athanasopoulos and Hyndman
(2019):

    reconciled_bottom = (S' W^-1 S)^-1 S' W^-1 * base_all
    reconciled_all    = S @ reconciled_bottom

where W is the covariance of the base-forecast errors. Here W is diagonal, built
from the placebo error variance of each node. A full (non-diagonal) W that
captures cross-region error correlation is a natural extension.

Bottom-up (sum the per-region estimates) and aggregate-only (take the aggregate
estimate and split it by base proportions) are provided as comparison baselines.
"""

from __future__ import annotations

import numpy as np


def summing_matrix(n_regions: int) -> np.ndarray:
    """S maps [bottom] to [total, bottom_1, ..., bottom_R]."""
    S = np.zeros((n_regions + 1, n_regions))
    S[0, :] = 1.0
    S[1:, :] = np.eye(n_regions)
    return S


def mint_reconcile(base_all: np.ndarray, var_all: np.ndarray, S: np.ndarray):
    """MinT reconciliation with a diagonal error covariance W = diag(var_all)."""
    var_all = np.asarray(var_all, dtype=float)
    # Floor variances so a perfect-looking fit does not dominate the weighting.
    floor = 1e-6 * (np.nanmax(var_all) if np.nanmax(var_all) > 0 else 1.0)
    var_all = np.clip(var_all, floor, None)

    Winv = np.diag(1.0 / var_all)
    M = S.T @ Winv @ S
    G = np.linalg.solve(M, S.T @ Winv)  # (R, R+1)
    reconciled_bottom = G @ base_all
    reconciled_all = S @ reconciled_bottom
    return reconciled_bottom, reconciled_all


def bottom_up(base_bottom: np.ndarray, S: np.ndarray):
    base_bottom = np.asarray(base_bottom, dtype=float)
    return base_bottom, S @ base_bottom


def aggregate_only(base_top: float, base_bottom: np.ndarray, S: np.ndarray):
    """Take the aggregate estimate and split it by base-forecast proportions."""
    base_bottom = np.asarray(base_bottom, dtype=float)
    total = base_bottom.sum()
    prop = base_bottom / total if total != 0 else np.full(len(base_bottom), 1.0 / len(base_bottom))
    bottom = base_top * prop
    return bottom, S @ bottom


def incoherence_gap(base_top: float, base_bottom: np.ndarray) -> float:
    """Bottom-up sum minus the aggregate estimate. Zero means already coherent."""
    return float(np.sum(base_bottom) - base_top)
