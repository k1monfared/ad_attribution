"""Bayesian structural time-series counterfactual (PyMC).

The counterfactual for the treated unit is a forecast. We fit a structural
time-series model to the PRE-campaign window and project it through the
post-campaign window. Because the donor series are unaffected controls that
keep moving through the post-period, the projection is driven by the donors
plus the estimated seasonal and level structure. The causal effect is the
observed series minus this predicted counterfactual, with full posterior
credible intervals.

This is the Bayesian-structural-time-series (BSTS) counterfactual of Brodersen
et al. (2015), "Inferring causal impact using Bayesian structural time-series
models", implemented from scratch in PyMC.

Model (fit in standardized space, then transformed back to streams)
------------------------------------------------------------------
    y_t = level_t + donors_t . beta + weekly_t + annual_t + noise_t

* level_t : a local level, a Gaussian random walk. Non-centered:
  level_t = level_0 + sigma_level * cumsum(innovations). The market trend is
  carried mostly by the donors; the local level absorbs the residual smooth
  drift. Its post-period forecast is flat from the last level with uncertainty
  that grows like sqrt(horizon), which is where most of the counterfactual
  uncertainty comes from.
* donors_t . beta : the donor/control series enter as EXTRA REGRESSORS. There
  is no convex sum-to-one constraint (this is a regression, not a synthetic
  control). A regularizing prior shrinks the coefficients (see `donor_prior`).
* weekly_t : weekly seasonality via Fourier terms (period 7).
* annual_t : annual / day-of-year seasonality via Fourier terms (period
  365.25), included only when the pre-period is long enough (see
  `should_include_annual`).

Donor coefficient prior
-----------------------
The default is an adaptive-ridge Normal: beta ~ Normal(0, tau),
tau ~ HalfNormal. This is a good match for a pool of PRE-SELECTED, co-moving,
similar-scale donors, where we expect several of them to contribute moderately
(a dense, not sparse, signal). The hierarchical scale tau lets the data set the
amount of shrinkage, it is numerically stable for NUTS, and it keeps the
coefficients from chasing pre-period noise. Laplace (sparse) and the
regularized horseshoe are available for the prior-sensitivity check.

Similar-scale donors are still required even though the weights are relaxed.
Every series is standardized before the regression, which removes the raw level
difference, but scale is not only a level. A donor at a very different volume
typically has a different signal-to-noise ratio and different dynamics, so after
standardizing it contributes mostly noise, and its coefficient is unstable when
projected into the post-period. Keeping donors of similar scale and co-movement
keeps the regressors informative and the forecast trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


# Annual seasonality needs roughly a full year of pre-period to identify its
# amplitude and phase. With less, the annual cycle is confounded with the trend
# and the local level and would be an unsupported extrapolation, so we omit it.
MIN_ANNUAL_PRE_DAYS = 365


@dataclass
class BayesConfig:
    donor_prior: str = "normal"     # "normal" (adaptive ridge), "laplace", "horseshoe"
    donor_prior_scale: float = 1.0  # scale of the hyperprior on the coefficient spread
    weekly_harmonics: int = 2
    annual_harmonics: int = 3
    level_sigma_scale: float = 0.05  # HalfNormal scale of the random-walk innovation sd
    obs_sigma_scale: float = 0.5
    seasonal_scale: float = 1.0
    draws: int = 1000
    tune: int = 1000
    chains: int = 2
    cores: int = 2
    target_accept: float = 0.95
    seed: int = 20260707


def fourier_design(t: np.ndarray, period: float, n_harmonics: int) -> np.ndarray:
    """Fourier feature matrix for a given period: [sin, cos] for each harmonic."""
    cols = []
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(cols) if cols else np.zeros((len(t), 0))


def should_include_annual(n_pre: int, min_days: int = MIN_ANNUAL_PRE_DAYS) -> bool:
    """Include the annual component only with enough pre-period history."""
    return n_pre >= min_days


def _standardize(y_pre, X, n_pre):
    y_mean = float(y_pre.mean())
    y_std = float(y_pre.std())
    y_std = y_std if y_std > 0 else 1.0
    X_mean = X[:n_pre].mean(axis=0)
    X_std = X[:n_pre].std(axis=0)
    X_std = np.where(X_std > 0, X_std, 1.0)
    Xz = (X - X_mean) / X_std
    return y_mean, y_std, Xz


def build_model(
    treated: np.ndarray,
    donors: np.ndarray,
    n_pre: int,
    cfg: BayesConfig,
    include_weekly: bool = True,
    include_annual: bool | None = None,
    dow: np.ndarray | None = None,
):
    """Assemble the PyMC model. Returns (model, meta).

    treated : (T,) observed treated series (streams)
    donors  : (T, D) donor series (streams)
    n_pre   : number of pre-campaign days
    dow     : optional day-of-week array (T,) for the weekly Fourier terms
    """
    treated = np.asarray(treated, dtype=float)
    donors = np.asarray(donors, dtype=float)
    T, D = donors.shape
    t = np.arange(T)

    if include_annual is None:
        include_annual = should_include_annual(n_pre)

    y_mean, y_std, Xz = _standardize(treated[:n_pre], donors, n_pre)
    ys_pre = (treated[:n_pre] - y_mean) / y_std

    week_t = t if dow is None else np.asarray(dow, dtype=float)
    Fw = fourier_design(week_t, 7.0, cfg.weekly_harmonics) if include_weekly else np.zeros((T, 0))
    Fa = fourier_design(t, 365.25, cfg.annual_harmonics) if include_annual else np.zeros((T, 0))

    with pm.Model() as model:
        # Donor regressors with a regularizing prior (relaxed, not convex).
        if cfg.donor_prior == "normal":
            tau = pm.HalfNormal("tau", cfg.donor_prior_scale)
            beta = pm.Normal("beta", 0.0, tau, shape=D)
        elif cfg.donor_prior == "laplace":
            b = pm.HalfNormal("b", cfg.donor_prior_scale)
            beta = pm.Laplace("beta", 0.0, b, shape=D)
        elif cfg.donor_prior == "horseshoe":
            tau0 = pm.HalfNormal("tau0", cfg.donor_prior_scale)
            lam = pm.HalfCauchy("lam", 1.0, shape=D)
            z = pm.Normal("z", 0.0, 1.0, shape=D)
            beta = pm.Deterministic("beta", z * lam * tau0)
        else:
            raise ValueError(f"unknown donor_prior {cfg.donor_prior!r}")

        contrib = pt.dot(Xz, beta)

        # Local level: non-centered Gaussian random walk.
        sigma_level = pm.HalfNormal("sigma_level", cfg.level_sigma_scale)
        level0 = pm.Normal("level0", 0.0, 1.0)
        innov = pm.Normal("innov", 0.0, 1.0, shape=T)
        level = pm.Deterministic("level", level0 + sigma_level * pt.cumsum(innov))
        mu = level + contrib

        if include_weekly:
            gamma_w = pm.Normal("gamma_weekly", 0.0, cfg.seasonal_scale, shape=Fw.shape[1])
            weekly = pm.Deterministic("weekly", pt.dot(Fw, gamma_w))
            mu = mu + weekly
        if include_annual:
            gamma_a = pm.Normal("gamma_annual", 0.0, cfg.seasonal_scale, shape=Fa.shape[1])
            annual = pm.Deterministic("annual", pt.dot(Fa, gamma_a))
            mu = mu + annual

        mu = pm.Deterministic("mu", mu)
        sigma = pm.HalfNormal("sigma", cfg.obs_sigma_scale)
        pm.Normal("y_obs", mu[:n_pre], sigma, observed=ys_pre)

    meta = {
        "T": T,
        "D": D,
        "n_pre": n_pre,
        "y_mean": y_mean,
        "y_std": y_std,
        "include_weekly": bool(include_weekly),
        "include_annual": bool(include_annual),
        "Fw_cols": Fw.shape[1],
        "Fa_cols": Fa.shape[1],
    }
    return model, meta


def _stack(posterior, name):
    """Return draws of a variable as (S, ...) with the sample dim first."""
    da = posterior[name].stack(sample=("chain", "draw"))
    # move sample to axis 0
    return np.moveaxis(da.values, -1, 0)


def counterfactual_draws(idata, meta, seed: int = 0):
    """Posterior counterfactual streams over the full horizon: (S, T)."""
    post = idata.posterior
    mu = _stack(post, "mu")                # (S, T) standardized
    sigma = _stack(post, "sigma")          # (S,)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=mu.shape) * sigma[:, None]
    cf_std = mu + noise
    cf = cf_std * meta["y_std"] + meta["y_mean"]
    mu_real = mu * meta["y_std"] + meta["y_mean"]
    return cf, mu_real


def effect_summary(observed, cf_draws, n_pre, ci: float = 0.95):
    """Pointwise, cumulative, and relative-lift effect with credible intervals.

    observed : (T,) observed streams
    cf_draws : (S, T) posterior counterfactual streams
    """
    observed = np.asarray(observed, dtype=float)
    T = observed.shape[0]
    post = slice(n_pre, T)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2

    effect = observed[None, :] - cf_draws                       # (S, T)
    cum_post = np.cumsum(effect[:, post], axis=1)               # (S, n_post)
    total = effect[:, post].sum(axis=1)                         # (S,)
    cf_total = cf_draws[:, post].sum(axis=1)                    # (S,)
    rel = total / cf_total

    def band(a, axis=0):
        return (
            a.mean(axis=axis),
            np.quantile(a, lo_q, axis=axis),
            np.quantile(a, hi_q, axis=axis),
        )

    eff_mean, eff_lo, eff_hi = band(effect)
    cum_mean, cum_lo, cum_hi = band(cum_post)
    cf_mean, cf_lo, cf_hi = band(cf_draws)

    pre_daily_mean = float(observed[:n_pre].mean())
    obs_post_daily = float(observed[post].mean())
    cf_post_daily = float(cf_mean[post].mean())

    return {
        "ci": ci,
        "pointwise_effect_mean": eff_mean,
        "pointwise_effect_lo": eff_lo,
        "pointwise_effect_hi": eff_hi,
        "cumulative_effect_mean": cum_mean,
        "cumulative_effect_lo": cum_lo,
        "cumulative_effect_hi": cum_hi,
        "counterfactual_mean": cf_mean,
        "counterfactual_lo": cf_lo,
        "counterfactual_hi": cf_hi,
        "total_effect_mean": float(total.mean()),
        "total_effect_lo": float(np.quantile(total, lo_q)),
        "total_effect_hi": float(np.quantile(total, hi_q)),
        "total_effect_draws": total,
        "relative_lift_mean": float(rel.mean()),
        "relative_lift_lo": float(np.quantile(rel, lo_q)),
        "relative_lift_hi": float(np.quantile(rel, hi_q)),
        "counterfactual_post_total_mean": float(cf_total.mean()),
        "n_pre": int(n_pre),
        "n_post": int(T - n_pre),
        "observed_post_total": float(observed[post].sum()),
        "pre_daily_mean": pre_daily_mean,
        "observed_post_daily_mean": obs_post_daily,
        "counterfactual_post_daily_mean": cf_post_daily,
        "mom_change_observed": obs_post_daily / pre_daily_mean - 1.0,
        "mom_change_counterfactual": cf_post_daily / pre_daily_mean - 1.0,
        "streams_retained": float(total.mean()),
    }


def fit_counterfactual(
    treated,
    donors,
    n_pre,
    cfg: BayesConfig,
    dates: pd.DatetimeIndex | None = None,
    include_weekly: bool = True,
    include_annual: bool | None = None,
    do_prior_predictive: bool = False,
    do_posterior_predictive: bool = False,
    ci: float = 0.95,
):
    """Fit the model, sample, and return the counterfactual and effect summary."""
    dow = dates.dayofweek.to_numpy() if dates is not None else None
    model, meta = build_model(
        treated, donors, n_pre, cfg,
        include_weekly=include_weekly, include_annual=include_annual, dow=dow,
    )

    with model:
        prior_pred = None
        if do_prior_predictive:
            prior_pred = pm.sample_prior_predictive(draws=400, random_seed=cfg.seed)
        idata = pm.sample(
            draws=cfg.draws, tune=cfg.tune, chains=cfg.chains, cores=cfg.cores,
            target_accept=cfg.target_accept, random_seed=cfg.seed, progressbar=False,
        )
        if do_posterior_predictive:
            pm.sample_posterior_predictive(
                idata, var_names=["y_obs"], random_seed=cfg.seed + 1,
                progressbar=False, extend_inferencedata=True,
            )

    cf, mu_real = counterfactual_draws(idata, meta, seed=cfg.seed + 2)
    summary = effect_summary(treated, cf, n_pre, ci=ci)

    return {
        "idata": idata,
        "meta": meta,
        "counterfactual_draws": cf,
        "mu_real": mu_real,
        "summary": summary,
        "prior_predictive": prior_pred,
        "include_annual": meta["include_annual"],
        "include_weekly": meta["include_weekly"],
        "y_std": meta["y_std"],
        "y_mean": meta["y_mean"],
    }


def compare_to_truth(summary, true_effect, true_counterfactual, n_pre):
    """Score the posterior effect against the injected ground truth."""
    true_effect = np.asarray(true_effect, dtype=float)
    true_counterfactual = np.asarray(true_counterfactual, dtype=float)
    post = slice(n_pre, len(true_effect))

    true_cum = float(true_effect[post].sum())
    true_rel = float(true_effect[post].sum() / true_counterfactual[post].sum())
    est_cum = summary["total_effect_mean"]
    est_rel = summary["relative_lift_mean"]
    covered = bool(summary["total_effect_lo"] <= true_cum <= summary["total_effect_hi"])
    return {
        "true_cumulative_effect": true_cum,
        "true_relative_lift": true_rel,
        "estimated_cumulative_effect": est_cum,
        "estimated_relative_lift": est_rel,
        "cumulative_effect_lo": summary["total_effect_lo"],
        "cumulative_effect_hi": summary["total_effect_hi"],
        "cumulative_abs_error": abs(est_cum - true_cum),
        "cumulative_pct_error": abs(est_cum - true_cum) / abs(true_cum) if true_cum != 0 else float("nan"),
        "relative_lift_abs_error": abs(est_rel - true_rel),
        "true_effect_covered": covered,
    }
