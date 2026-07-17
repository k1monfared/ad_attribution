"""Bayesian convergence and fit diagnostics with ArviZ.

Everything a skeptical reader asks for after seeing a posterior:

* convergence: R-hat, effective sample size (bulk and tail), divergences, and
  the energy-based BFMI,
* fit: prior predictive and posterior predictive checks on the pre-period.

The numeric diagnostics use ArviZ (`az.rhat`, `az.ess`, `az.summary`). BFMI is
computed directly from the sampler energy for a stable scalar per chain.
"""

from __future__ import annotations

import arviz as az
import numpy as np

# Deterministics we do not treat as sampled parameters for convergence. `mu`
# and `level` include the post-period extrapolation, which is unconstrained by
# data by construction, so their post-period R-hat is not a sampling problem.
_DETERMINISTIC = {"mu", "level", "weekly", "annual", "beta"}


def _param_names(idata):
    names = list(idata.posterior.data_vars)
    keep = [n for n in names if n not in _DETERMINISTIC or n == "beta"]
    # beta is a Deterministic only under the horseshoe; keep it either way since
    # it is the quantity of interest and is low-dimensional.
    return keep


def convergence(idata) -> dict:
    """R-hat, ESS, divergences, and BFMI over the sampled parameters."""
    var_names = _param_names(idata)

    rhat = az.rhat(idata, var_names=var_names)
    ess_bulk = az.ess(idata, var_names=var_names, method="bulk")
    ess_tail = az.ess(idata, var_names=var_names, method="tail")

    def _reduce(ds, op):
        vals = [getattr(ds[v], op)() for v in ds.data_vars if ds[v].values.size > 0]
        vals = [float(x) for x in vals]
        return (max(vals) if op == "max" else min(vals)) if vals else float("nan")

    max_rhat = _reduce(rhat, "max")
    min_ess_bulk = _reduce(ess_bulk, "min")
    min_ess_tail = _reduce(ess_tail, "min")

    diverging = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else 0
    n_samples = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])

    energy = idata.sample_stats["energy"].values  # (chain, draw)
    bfmi = [
        float(np.sum(np.diff(energy[c]) ** 2) / np.sum((energy[c] - energy[c].mean()) ** 2))
        for c in range(energy.shape[0])
    ]

    return {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "divergences": diverging,
        "n_samples": n_samples,
        "bfmi_per_chain": bfmi,
        "min_bfmi": float(min(bfmi)),
        "n_chains": int(idata.posterior.sizes["chain"]),
        "n_draws": int(idata.posterior.sizes["draw"]),
    }


def summary_table(idata, var_names=None):
    """ArviZ summary (mean, sd, 95% ETI, ess, r_hat) for reporting."""
    if var_names is None:
        var_names = [v for v in ("beta", "sigma", "sigma_level", "tau") if v in idata.posterior.data_vars]
    return az.summary(idata, var_names=var_names, ci_prob=0.95, ci_kind="eti")
