"""Multi-region attribution: Bayesian base estimates, reconciliation, Monte Carlo.

Each node of the hierarchy (every region, and the national aggregate) gets its
own Bayesian structural-time-series counterfactual. The additive quantity is the
cumulative extra streams over the post-period. Estimating each node
independently is incoherent: the per-region posterior means do not sum to the
aggregate posterior mean, because they come from different donor pools with
different fit quality.

We keep the MinT reconciliation and feed it the Bayesian estimates. This is the
cleaner of the two options: a single joint hierarchical Bayesian model over five
regions plus the aggregate, each with its own local level, donor regression and
seasonality, is far heavier to fit and to check, and it would not change the
headline message. Reusing the validated MinT step is also more honest about the
uncertainty, because the reconciliation weight matrix W is now built from the
POSTERIOR variance of each node's effect (a principled model-based variance)
rather than from a placebo proxy. MinT then downweights the noisy donor-poor
regions and returns per-region effects that sum exactly to the reconciled total.
"""

from __future__ import annotations

import numpy as np

from . import reconciliation
from .bayesian_model import BayesConfig, fit_counterfactual
from .data_gen import RegionsConfig, generate_region_campaign


def _aggregate_series(campaign):
    """Aggregate treated series (sum of regions) and the national donor pool."""
    names = campaign["names"]
    per = campaign["per_region"]
    treated_agg = np.sum([np.asarray(per[n]["treated"], dtype=float) for n in names], axis=0)
    donor_agg = campaign["national_donors"].to_numpy()
    return treated_agg, donor_agg


def base_estimates(campaign, cfg: BayesConfig, dates=None) -> dict:
    """Independent Bayesian effect and posterior variance for every node."""
    n_pre = campaign["campaign_start"]
    names = campaign["names"]
    per = campaign["per_region"]

    bottom = []
    bottom_var = []
    for name in names:
        treated = np.asarray(per[name]["treated"], dtype=float)
        donors = per[name]["donors"].to_numpy()
        d = per[name].get("dates", dates)
        fit = fit_counterfactual(treated, donors, n_pre, cfg, dates=d, include_annual=False)
        draws = fit["summary"]["total_effect_draws"]
        bottom.append(float(draws.mean()))
        bottom_var.append(float(draws.var()))

    treated_agg, donor_agg = _aggregate_series(campaign)
    fit_a = fit_counterfactual(treated_agg, donor_agg, n_pre, cfg, dates=dates, include_annual=False)
    draws_a = fit_a["summary"]["total_effect_draws"]
    top = float(draws_a.mean())
    top_var = float(draws_a.var())

    bottom = np.array(bottom)
    bottom_var = np.array(bottom_var)
    base_all = np.concatenate([[top], bottom])
    var_all = np.concatenate([[top_var], bottom_var])

    true_bottom = np.array([campaign["true_effects"][n] for n in names])
    return {
        "names": names,
        "base_top": float(top),
        "base_bottom": bottom,
        "base_all": base_all,
        "top_var": float(top_var),
        "bottom_var": bottom_var,
        "var_all": var_all,
        "true_bottom": true_bottom,
        "true_top": float(campaign["true_total"]),
    }


def reconcile(base: dict) -> dict:
    """Apply MinT, bottom-up, and aggregate-only to one set of base estimates."""
    n = len(base["names"])
    S = reconciliation.summing_matrix(n)

    mint_bottom, mint_all = reconciliation.mint_reconcile(base["base_all"], base["var_all"], S)
    bu_bottom, bu_all = reconciliation.bottom_up(base["base_bottom"], S)
    ao_bottom, ao_all = reconciliation.aggregate_only(base["base_top"], base["base_bottom"], S)
    gap = reconciliation.incoherence_gap(base["base_top"], base["base_bottom"])

    true_top = base["true_top"]
    return {
        "names": base["names"],
        "summing_matrix": S.tolist(),
        "incoherence_gap": gap,
        "true_top": true_top,
        "true_bottom": base["true_bottom"].tolist(),
        "base": {
            "top": base["base_top"],
            "bottom": base["base_bottom"].tolist(),
            "bottom_sum": float(base["base_bottom"].sum()),
        },
        "mint": {
            "bottom": mint_bottom.tolist(),
            "top": float(mint_all[0]),
            "error_vs_true_top": float(mint_all[0] - true_top),
        },
        "bottom_up": {
            "bottom": bu_bottom.tolist(),
            "top": float(bu_all[0]),
            "error_vs_true_top": float(bu_all[0] - true_top),
        },
        "aggregate_only": {
            "bottom": ao_bottom.tolist(),
            "top": float(ao_all[0]),
            "error_vs_true_top": float(ao_all[0] - true_top),
        },
    }


def run_illustrative(regions_cfg: RegionsConfig, cfg: BayesConfig) -> dict:
    """One campaign at the master seed, for the figures and the report."""
    campaign = generate_region_campaign(regions_cfg, regions_cfg.seed)
    dates = campaign["per_region"][campaign["names"][0]]["dates"]
    base = base_estimates(campaign, cfg, dates=dates)
    result = reconcile(base)
    result["var_all"] = base["var_all"].tolist()
    return result


def monte_carlo(regions_cfg: RegionsConfig, n_runs: int, cfg: BayesConfig) -> dict:
    """Repeat over many simulated campaigns and summarize each method."""
    rng = np.random.default_rng(regions_cfg.seed)
    seeds = rng.integers(1, 2**31 - 1, size=n_runs)

    gaps = []
    err = {"bottom_up": [], "aggregate_only": [], "mint": []}
    abserr = {"bottom_up": [], "aggregate_only": [], "mint": []}
    true_top = None

    for s in seeds:
        campaign = generate_region_campaign(regions_cfg, int(s))
        dates = campaign["per_region"][campaign["names"][0]]["dates"]
        base = base_estimates(campaign, cfg, dates=dates)
        res = reconcile(base)
        true_top = res["true_top"]
        gaps.append(res["incoherence_gap"])
        for m in err:
            e = res[m]["error_vs_true_top"]
            err[m].append(e)
            abserr[m].append(abs(e))

    def stats(x):
        x = np.asarray(x, dtype=float)
        return {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)),
            "median": float(np.median(x)),
            "p05": float(np.quantile(x, 0.05)),
            "p95": float(np.quantile(x, 0.95)),
        }

    return {
        "n_runs": int(n_runs),
        "true_top": float(true_top),
        "incoherence_gap": stats(gaps),
        "signed_error": {m: stats(err[m]) for m in err},
        "abs_error": {m: stats(abserr[m]) for m in abserr},
        "abs_error_mean": {m: float(np.mean(abserr[m])) for m in abserr},
        "raw": {
            "gaps": list(map(float, gaps)),
            "abs_error": {m: list(map(float, abserr[m])) for m in abserr},
        },
    }
