"""Donor-pool construction: feature matching plus parallel-trends validation.

A synthetic control is only as good as its donor pool. Two failure modes matter
most here:

1. The fanbase pitfall. Campaigns target a specific region or audience. A
   tempting control is the SAME artist in other, untargeted regions. But if the
   campaign targets the artist's existing fanbase, those other regions are the
   loyal base: roughly flat, and not a model for how the TARGET audience would
   have moved without the ad. Using them biases the estimate.

2. Donors that simply do not co-move. Even among other artists, many will not
   track the treated unit organically, and including them lets the optimizer
   fit noise in the pre-period.

The pipeline here is the one that scales:

* feature match: rank candidate artists or songs by similarity in a feature or
  embedding space (genre, tempo, audience demographics, catalog age, momentum),
  and keep the closest ones. At catalog scale this is a nearest-neighbor lookup
  over embeddings, cheap even for large catalogs.
* parallel-trends validation: of the feature-matched candidates, keep only
  those whose PRE-period series actually moved together with the treated unit
  through organic changes, measured by correlation on levels and on daily
  changes. Drop the rest. This is the explicit, visible check that the
  parallel-trends assumption holds before any effect is estimated.

The fanbase series fails this check by construction: it is flat, so it does not
co-move with a trending target audience.
"""

from __future__ import annotations

import numpy as np


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 0 else 0.0


def comovement(treated_pre: np.ndarray, series_pre: np.ndarray) -> dict:
    """Pre-period co-movement of one series with the treated unit."""
    level_corr = _corr(treated_pre, series_pre)
    diff_corr = _corr(np.diff(treated_pre), np.diff(series_pre))
    return {"level_corr": level_corr, "diff_corr": diff_corr}


def select_donors(
    treated_pre: np.ndarray,
    candidates_pre: np.ndarray,
    candidate_features: np.ndarray,
    treated_features: np.ndarray,
    candidate_names: list[str],
    feature_top_k: int = 12,
    level_corr_threshold: float = 0.8,
    diff_corr_threshold: float = 0.2,
):
    """Two-stage donor selection: feature pre-filter, then parallel-trends check.

    Returns selected column indices plus a per-candidate report.
    """
    n_cand = candidates_pre.shape[1]

    # Stage 1: feature-space nearest neighbors (Euclidean distance).
    dists = np.linalg.norm(candidate_features - treated_features[None, :], axis=1)
    feature_rank = np.argsort(dists)
    feature_keep = set(feature_rank[:feature_top_k].tolist())

    # Stage 2: parallel-trends validation on the feature-matched candidates.
    report = []
    selected = []
    for j in range(n_cand):
        cm = comovement(treated_pre, candidates_pre[:, j])
        passed_feature = j in feature_keep
        passed_trends = (
            cm["level_corr"] >= level_corr_threshold
            and cm["diff_corr"] >= diff_corr_threshold
        )
        keep = passed_feature and passed_trends
        if keep:
            selected.append(j)
        report.append(
            {
                "name": candidate_names[j],
                "feature_distance": float(dists[j]),
                "passed_feature_filter": bool(passed_feature),
                "level_corr": cm["level_corr"],
                "diff_corr": cm["diff_corr"],
                "selected": bool(keep),
            }
        )

    return np.array(selected, dtype=int), report


def summarize_selection(report: list[dict], candidate_is_good: np.ndarray) -> dict:
    """Score the selection against the known good/bad labels (demo only)."""
    selected_flags = np.array([r["selected"] for r in report])
    good = np.asarray(candidate_is_good, dtype=bool)
    n_selected = int(selected_flags.sum())
    n_good_selected = int((selected_flags & good).sum())
    n_bad_selected = int((selected_flags & ~good).sum())
    selected_level_corrs = [r["level_corr"] for r in report if r["selected"]]
    return {
        "n_candidates": len(report),
        "n_selected": n_selected,
        "n_good_selected": n_good_selected,
        "n_bad_selected": n_bad_selected,
        "min_selected_level_corr": float(min(selected_level_corrs)) if selected_level_corrs else float("nan"),
        "mean_selected_level_corr": float(np.mean(selected_level_corrs)) if selected_level_corrs else float("nan"),
    }
