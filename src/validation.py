"""Correctness and robustness checks for the Bayesian counterfactual.

* coverage_study: simulate many campaigns with known injected effects, fit the
  model, and measure how often the 95% credible interval contains the true
  cumulative effect (should be about 95%) and how well the point estimate
  recovers the truth. Runs the seasonality-AWARE and the seasonality-BLIND
  model on the same campaigns so their bias and calibration can be compared.

* prior_sensitivity: refit one scenario under different donor-coefficient priors
  (normal, laplace, horseshoe) to show the estimate is not an artifact of the
  prior.

These are the checks a skeptical reader asks for: calibration across many
campaigns, recovery of the truth, and robustness to modeling choices.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import diagnostics, donor_selection
from .bayesian_model import BayesConfig, compare_to_truth, fit_counterfactual
from .data_gen import ScenarioConfig, generate_scenario


def _select(scn, sel_cfg, n_pre):
    treated = scn["treated"]["streams"].to_numpy()
    candidates = scn["candidates"].to_numpy()
    cand_names = list(scn["candidates"].columns)
    idx, _ = donor_selection.select_donors(
        treated[:n_pre], candidates[:n_pre], scn["candidate_features"],
        scn["treated_features"], cand_names,
        feature_top_k=sel_cfg["feature_top_k"],
        level_corr_threshold=sel_cfg["level_corr_threshold"],
        diff_corr_threshold=sel_cfg["diff_corr_threshold"],
    )
    if idx.size == 0:  # fall back to feature top-k if the filter is too strict
        idx = np.arange(min(8, candidates.shape[1]))
    return treated, candidates[:, idx]


def coverage_study(cfg, fit_cfg: BayesConfig, n_campaigns: int, seed: int) -> dict:
    """Coverage and recovery across simulated campaigns, aware vs blind."""
    common = cfg["common"]
    sel_cfg = cfg["selection"]
    val = cfg["validation"]
    rng = np.random.default_rng(seed)

    rec = {"aware": [], "blind": []}
    worst_rhat = 0.0
    total_div = 0

    for i in range(n_campaigns):
        peak_lift = float(rng.uniform(*val["lift_range"]))
        trend_total = float(rng.uniform(*val["trend_range"]))
        noise = float(rng.uniform(*val["noise_range"]))
        base = float(rng.uniform(*val["base_range"]))
        scn_cfg = ScenarioConfig(
            seed=int(rng.integers(1, 2**31 - 1)),
            name=f"cov_{i}",
            n_days=common["n_days"],
            campaign_start=common["campaign_start"],
            n_good=val["n_good"],
            n_bad=val["n_bad"],
            trend_total=trend_total,
            peak_lift=peak_lift,
            ramp_days=common["ramp_days"],
            noise_scale=noise,
            base_level=base,
        )
        scn = generate_scenario(scn_cfg)
        n_pre = scn["campaign_start"]
        dates = scn["dates"]
        treated, donors = _select(scn, sel_cfg, n_pre)

        for label, include_weekly in (("aware", True), ("blind", False)):
            fit = fit_counterfactual(
                treated, donors, n_pre, fit_cfg, dates=dates,
                include_weekly=include_weekly, include_annual=False,
            )
            truth = compare_to_truth(
                fit["summary"], scn["true_effect"], scn["true_counterfactual"], n_pre
            )
            conv = diagnostics.convergence(fit["idata"])
            worst_rhat = max(worst_rhat, conv["max_rhat"])
            total_div += conv["divergences"]
            width = fit["summary"]["total_effect_hi"] - fit["summary"]["total_effect_lo"]
            rec[label].append({
                "covered": truth["true_effect_covered"],
                "signed_pct_error": (truth["estimated_cumulative_effect"] - truth["true_cumulative_effect"])
                / abs(truth["true_cumulative_effect"]),
                "abs_pct_error": truth["cumulative_pct_error"],
                "rel_lift_abs_error": truth["relative_lift_abs_error"],
                "interval_width": float(width),
                "true_cum": truth["true_cumulative_effect"],
                "est_cum": truth["estimated_cumulative_effect"],
            })

    def agg(records):
        cov = float(np.mean([r["covered"] for r in records]))
        return {
            "coverage_95": cov,
            "n": len(records),
            "mean_abs_pct_error": float(np.mean([r["abs_pct_error"] for r in records])),
            "median_abs_pct_error": float(np.median([r["abs_pct_error"] for r in records])),
            "mean_signed_pct_error": float(np.mean([r["signed_pct_error"] for r in records])),
            "mean_rel_lift_abs_error": float(np.mean([r["rel_lift_abs_error"] for r in records])),
            "mean_interval_width": float(np.mean([r["interval_width"] for r in records])),
            "true_cum": [r["true_cum"] for r in records],
            "est_cum": [r["est_cum"] for r in records],
            "covered_flags": [bool(r["covered"]) for r in records],
        }

    return {
        "n_campaigns": n_campaigns,
        "aware": agg(rec["aware"]),
        "blind": agg(rec["blind"]),
        "worst_rhat": float(worst_rhat),
        "total_divergences": int(total_div),
    }


def prior_sensitivity(scn, cfg, fit_cfg: BayesConfig, priors=("normal", "laplace", "horseshoe")) -> dict:
    """Refit one scenario under different donor-coefficient priors."""
    n_pre = scn["campaign_start"]
    sel_cfg = cfg["selection"]
    treated, donors = _select(scn, sel_cfg, n_pre)
    dates = scn["dates"]

    out = {}
    for prior in priors:
        pcfg = replace(fit_cfg, donor_prior=prior)
        fit = fit_counterfactual(treated, donors, n_pre, pcfg, dates=dates, include_annual=False)
        truth = compare_to_truth(fit["summary"], scn["true_effect"], scn["true_counterfactual"], n_pre)
        conv = diagnostics.convergence(fit["idata"])
        s = fit["summary"]
        out[prior] = {
            "cumulative_effect_mean": s["total_effect_mean"],
            "cumulative_effect_lo": s["total_effect_lo"],
            "cumulative_effect_hi": s["total_effect_hi"],
            "relative_lift_mean": s["relative_lift_mean"],
            "true_cumulative_effect": truth["true_cumulative_effect"],
            "covered": truth["true_effect_covered"],
            "max_rhat": conv["max_rhat"],
            "divergences": conv["divergences"],
        }
    return out
