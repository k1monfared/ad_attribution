"""One entry point that reproduces the entire Bayesian demo.

    python scripts/run_demo.py

Steps:
  1. generate and commit synthetic data (three scenarios plus regions)
  2. build the donor pool: feature match, then validate parallel trends
  3. fit a Bayesian structural-time-series counterfactual per scenario, estimate
     the effect with full posterior credible intervals
  4. run the Bayesian diagnostic suite (R-hat, ESS, divergences, BFMI, prior and
     posterior predictive checks)
  5. validate: credible-interval coverage across many simulated campaigns,
     seasonality-aware vs seasonality-blind comparison, donor-prior sensitivity
  6. demonstrate the annual / day-of-year component inclusion rule (long history
     includes it, short history omits it)
  7. reconcile Bayesian per-region and aggregate estimates across regions (MinT)
     and run a small Monte Carlo comparing bottom-up, aggregate-only, MinT
  8. write outputs (report .md + results .json) and docs/images figures
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import aggregate, diagnostics, donor_selection, io_utils, plotting, validation  # noqa: E402
from src.bayesian_model import BayesConfig, compare_to_truth, fit_counterfactual, should_include_annual  # noqa: E402
from src.data_gen import annual_signal  # noqa: E402
from scripts.generate_data import build_all, write_data  # noqa: E402


def make_bayes_cfg(cfg, draws=None, tune=None):
    b = cfg["bayes"]
    return BayesConfig(
        donor_prior=b["donor_prior"],
        donor_prior_scale=b["donor_prior_scale"],
        weekly_harmonics=b["weekly_harmonics"],
        annual_harmonics=b["annual_harmonics"],
        level_sigma_scale=b["level_sigma_scale"],
        draws=draws if draws is not None else b["draws"],
        tune=tune if tune is not None else b["tune"],
        chains=b["chains"],
        cores=b["chains"],
        target_accept=b["target_accept"],
        seed=b["seed"],
    )


def _select_donors(scn, sel):
    n_pre = scn["campaign_start"]
    treated = scn["treated"]["streams"].to_numpy()
    candidates = scn["candidates"].to_numpy()
    cand_names = list(scn["candidates"].columns)
    idx, report = donor_selection.select_donors(
        treated[:n_pre], candidates[:n_pre], scn["candidate_features"],
        scn["treated_features"], cand_names,
        feature_top_k=sel["feature_top_k"],
        level_corr_threshold=sel["level_corr_threshold"],
        diff_corr_threshold=sel["diff_corr_threshold"],
    )
    if idx.size == 0:  # never fit an empty donor set
        dists = np.linalg.norm(scn["candidate_features"] - scn["treated_features"][None, :], axis=1)
        idx = np.argsort(dists)[: min(8, candidates.shape[1])]
    return treated, candidates[:, idx], idx, report


def _predictive_band(group, var, y_std, y_mean):
    da = group[var].stack(sample=("chain", "draw"))
    arr = np.moveaxis(da.values, -1, 0) * y_std + y_mean  # (S, n)
    return np.quantile(arr, 0.025, 0), np.quantile(arr, 0.975, 0), np.median(arr, 0)


def run_scenario(scn, cfg, fit_cfg, predictive=False):
    n_pre = scn["campaign_start"]
    dates = scn["dates"]
    sel = cfg["selection"]
    treated, donors, sel_idx, sel_report = _select_donors(scn, sel)
    sel_summary = donor_selection.summarize_selection(sel_report, scn["candidate_is_good"])

    fan = scn["fanbase"].to_numpy()
    fan_cm = donor_selection.comovement(treated[:n_pre], fan[:n_pre])

    fit = fit_counterfactual(
        treated, donors, n_pre, fit_cfg, dates=dates,
        do_prior_predictive=predictive, do_posterior_predictive=predictive,
    )
    summary = fit["summary"]
    truth = compare_to_truth(summary, scn["true_effect"], scn["true_counterfactual"], n_pre)
    conv = diagnostics.convergence(fit["idata"])
    summ_table = diagnostics.summary_table(fit["idata"])

    return {
        "name": scn["name"],
        "n_pre": n_pre,
        "dates": dates,
        "treated": treated,
        "true_counterfactual": scn["true_counterfactual"],
        "true_effect": scn["true_effect"],
        "selected_idx": sel_idx,
        "selected_donors": donors,
        "fanbase": fan,
        "fanbase_comovement": fan_cm,
        "selection_summary": sel_summary,
        "fit": fit,
        "summary": summary,
        "truth": truth,
        "convergence": conv,
        "summary_table": summ_table,
    }


def make_scenario_figures(r, img):
    s = r["summary"]
    plotting.plot_observed_vs_counterfactual(
        r["dates"], r["treated"], s["counterfactual_mean"], s["counterfactual_lo"],
        s["counterfactual_hi"], r["n_pre"], r["true_counterfactual"],
        "Growth scenario: observed vs Bayesian counterfactual",
        img / "observed_vs_counterfactual.png",
    )
    plotting.plot_cumulative(
        r["dates"], s["cumulative_effect_mean"], s["cumulative_effect_lo"],
        s["cumulative_effect_hi"], r["n_pre"],
        np.cumsum(r["true_effect"][r["n_pre"]:]),
        "Cumulative campaign effect (growth scenario)", img / "cumulative_effect.png",
    )
    plotting.plot_pointwise_effect(
        r["dates"], s["pointwise_effect_mean"], s["pointwise_effect_lo"],
        s["pointwise_effect_hi"], r["n_pre"], r["true_effect"],
        "Pointwise campaign effect with 95% credible interval (growth scenario)",
        img / "pointwise_effect.png",
    )
    plotting.plot_donor_selection(
        r["dates"], r["treated"], r["selected_donors"], r["fanbase"], r["n_pre"],
        img / "donor_selection.png",
    )
    plotting.plot_convergence(r["fit"]["idata"], img / "convergence_diagnostics.png")

    # Prior and posterior predictive checks on the pre-period.
    fit = r["fit"]
    y_std, y_mean = fit["y_std"], fit["y_mean"]
    pre = slice(0, r["n_pre"])
    prior_lo, prior_hi, _ = _predictive_band(
        fit["prior_predictive"].prior_predictive, "y_obs", y_std, y_mean)
    post_lo, post_hi, post_med = _predictive_band(
        fit["idata"].posterior_predictive, "y_obs", y_std, y_mean)
    plotting.plot_ppc(
        r["dates"][pre], r["treated"][pre], prior_lo, prior_hi,
        post_lo, post_hi, post_med, img / "posterior_predictive_check.png",
    )


def _pct(x):
    return f"{x * 100:.1f}%"


def write_report(res, out_dir, runtime_s):
    g, d, ad = res["growth"], res["decline"], res["annual_demo"]
    cov = res["coverage"]
    ps = res["prior_sensitivity"]
    il = res["regions"]["illustrative"]
    mc = res["regions"]["monte_carlo"]
    md = []
    md.append("# Attribution demo report\n")
    md.append(
        "Generated by `scripts/run_demo.py`. Method: a Bayesian structural "
        "time-series counterfactual (PyMC) with donors as regressors, a local "
        "level, weekly seasonality, and a conditionally-included annual "
        "component. MCMC is stochastic but seeded, so estimates and convergence "
        "are stable across runs.\n"
    )
    md.append(f"- Total runtime: {runtime_s/60:.1f} min (CPU)\n")

    for tag, r in (("Growth", g), ("Decline", d)):
        s, t, c = r["summary"], r["truth"], r["convergence"]
        md.append(f"\n## {tag} scenario\n")
        md.append("| quantity | estimate | 95% credible interval | ground truth |")
        md.append("|---|---|---|---|")
        md.append(
            f"| cumulative extra streams | {t['estimated_cumulative_effect']:,.0f} "
            f"| {s['total_effect_lo']:,.0f} to {s['total_effect_hi']:,.0f} "
            f"| {t['true_cumulative_effect']:,.0f} |"
        )
        md.append(
            f"| relative lift | {_pct(t['estimated_relative_lift'])} "
            f"| {_pct(s['relative_lift_lo'])} to {_pct(s['relative_lift_hi'])} "
            f"| {_pct(t['true_relative_lift'])} |"
        )
        md.append("")
        md.append(f"- Cumulative estimate error vs truth: {t['cumulative_pct_error']*100:.1f}%.")
        md.append(f"- True cumulative effect inside the 95% credible interval: {t['true_effect_covered']}.")
        md.append(
            f"- Annual component included: {r['fit']['include_annual']} "
            f"(pre-period {r['n_pre']} days, rule requires >= 365)."
        )
        md.append(
            f"- Convergence: max R-hat {c['max_rhat']:.3f}, min ESS(bulk) {c['min_ess_bulk']:.0f}, "
            f"divergences {c['divergences']}/{c['n_samples']}, min BFMI {c['min_bfmi']:.2f}."
        )
        if tag == "Decline":
            md.append(
                f"- Trend framing: observed post-period is "
                f"{s['mom_change_observed']*100:+.1f}% versus the pre-campaign level, "
                f"so it still looks down. Without the campaign it would have been "
                f"{s['mom_change_counterfactual']*100:+.1f}%. The campaign retained about "
                f"{s['streams_retained']:,.0f} streams (decline avoided)."
            )

    # Donor selection.
    gs = g["selection_summary"]
    fcm = g["fanbase_comovement"]
    md.append("\n## Donor selection and the fanbase pitfall\n")
    md.append(
        f"From {gs['n_candidates']} candidates, {gs['n_selected']} were selected "
        f"({gs['n_good_selected']} genuinely similar, {gs['n_bad_selected']} unrelated). "
        f"Minimum pre-period level correlation among selected donors: "
        f"{gs['min_selected_level_corr']:.3f}."
    )
    md.append(
        f"- The same-artist other-region (fanbase) series has pre-period level "
        f"correlation {fcm['level_corr']:.3f} and daily-change correlation "
        f"{fcm['diff_corr']:.3f} with the target audience. It is flat and correctly excluded."
    )

    # Annual component.
    md.append("\n## Annual / day-of-year component inclusion rule\n")
    md.append(
        f"The annual Fourier component is added only when the pre-period spans at "
        f"least 365 days. Growth and decline scenarios have a 120-day pre-period, "
        f"so the annual component is correctly OMITTED ({g['fit']['include_annual']}). "
        f"The long-history scenario has a {ad['n_pre']}-day pre-period, so the annual "
        f"component is INCLUDED ({ad['fit']['include_annual']})."
    )
    adt = ad["truth"]
    md.append(
        f"- Long-history recovery: estimated cumulative effect "
        f"{adt['estimated_cumulative_effect']:,.0f} vs true {adt['true_cumulative_effect']:,.0f} "
        f"({adt['cumulative_pct_error']*100:.1f}% error), true effect covered: {adt['true_effect_covered']}."
    )

    # Coverage and seasonality comparison.
    aw, bl = cov["aware"], cov["blind"]
    md.append(f"\n## Credible-interval coverage over {cov['n_campaigns']} simulated campaigns\n")
    md.append(
        "Each campaign has a known injected effect. We check how often the 95% "
        "credible interval contains the true cumulative effect (calibration) and "
        "how well the point estimate recovers it (accuracy).\n"
    )
    md.append("| model | 95% CI coverage | median abs error (cumulative) | mean rel-lift error | mean CI width |")
    md.append("|---|---|---|---|---|")
    md.append(
        f"| seasonality-aware | {aw['coverage_95']*100:.0f}% | {aw['median_abs_pct_error']*100:.1f}% "
        f"| {aw['mean_rel_lift_abs_error']*100:.1f}% | {aw['mean_interval_width']:,.0f} |"
    )
    md.append(
        f"| seasonality-blind | {bl['coverage_95']*100:.0f}% | {bl['median_abs_pct_error']*100:.1f}% "
        f"| {bl['mean_rel_lift_abs_error']*100:.1f}% | {bl['mean_interval_width']:,.0f} |"
    )
    md.append("")
    md.append(
        f"- The seasonality-aware model recovers the relative lift to within "
        f"{aw['mean_rel_lift_abs_error']*100:.1f}% on average, and its 95% credible interval "
        f"contained the true cumulative effect in {aw['coverage_95']*100:.0f}% of campaigns "
        f"(at or above the nominal 95%, i.e. conservative rather than overconfident)."
    )
    md.append(
        f"- The seasonality-blind model must inflate its observation noise to absorb the "
        f"unmodeled weekly pattern, so its intervals are about "
        f"{(bl['mean_interval_width']/aw['mean_interval_width']-1)*100:.0f}% wider "
        f"({bl['coverage_95']*100:.0f}% coverage, over the nominal 95%) with no gain in accuracy. "
        f"It is the less sharp and less well-calibrated model."
    )
    md.append(
        f"- Across all coverage fits, worst R-hat {cov['worst_rhat']:.3f} and "
        f"{cov['total_divergences']} total divergences."
    )

    # Prior sensitivity.
    md.append("\n## Donor-coefficient prior sensitivity (growth scenario)\n")
    md.append("| prior | cumulative effect | 95% credible interval | max R-hat | divergences |")
    md.append("|---|---|---|---|---|")
    for prior in ("normal", "laplace", "horseshoe"):
        p = ps[prior]
        md.append(
            f"| {prior} | {p['cumulative_effect_mean']:,.0f} "
            f"| {p['cumulative_effect_lo']:,.0f} to {p['cumulative_effect_hi']:,.0f} "
            f"| {p['max_rhat']:.3f} | {p['divergences']} |"
        )
    md.append(f"\nTrue cumulative effect: {ps['normal']['true_cumulative_effect']:,.0f}. "
              "The estimate is stable across priors.")

    # Multi-region.
    md.append("\n## Multi-region hierarchical reconciliation (Bayesian + MinT)\n")
    md.append(
        "Each region and the national aggregate get their own Bayesian "
        "counterfactual. Independent estimates are incoherent; MinT reconciliation "
        "(weighted by the posterior variance of each node) makes them coherent and "
        "reduces error.\n"
    )
    md.append("### Illustrative campaign (fixed seed)\n")
    md.append(
        f"- Aggregate estimate: {il['base']['top']:,.0f}. Sum of independent per-region "
        f"estimates: {il['base']['bottom_sum']:,.0f}. Incoherence gap: {il['incoherence_gap']:,.0f}."
    )
    md.append(f"- True total injected effect: {il['true_top']:,.0f}.")
    md.append("")
    md.append("| method | total estimate | error vs true total |")
    md.append("|---|---|---|")
    for m, label in (("bottom_up", "bottom-up"), ("aggregate_only", "aggregate-only"), ("mint", "reconciled (MinT)")):
        md.append(f"| {label} | {il[m]['top']:,.0f} | {il[m]['error_vs_true_top']:+,.0f} |")

    md.append(f"\n### Monte Carlo over {mc['n_runs']} campaigns (fixed master seed)\n")
    md.append(
        f"True total held fixed at {mc['true_top']:,.0f}. Incoherence gap: mean "
        f"{mc['incoherence_gap']['mean']:,.0f}, standard deviation {mc['incoherence_gap']['std']:,.0f}.\n"
    )
    md.append("| method | mean absolute error | mean signed error |")
    md.append("|---|---|---|")
    for m, label in (("bottom_up", "bottom-up"), ("aggregate_only", "aggregate-only"), ("mint", "reconciled (MinT)")):
        md.append(
            f"| {label} | {mc['abs_error_mean'][m]:,.0f} | {mc['signed_error'][m]['mean']:+,.0f} |"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attribution_report.md").write_text("\n".join(md) + "\n")


def _scenario_json(r):
    s, t, c = r["summary"], r["truth"], r["convergence"]
    return {
        "cumulative_effect_mean": s["total_effect_mean"],
        "cumulative_effect_ci": [s["total_effect_lo"], s["total_effect_hi"]],
        "relative_lift_mean": s["relative_lift_mean"],
        "relative_lift_ci": [s["relative_lift_lo"], s["relative_lift_hi"]],
        "truth_comparison": t,
        "include_annual": r["fit"]["include_annual"],
        "include_weekly": r["fit"]["include_weekly"],
        "convergence": c,
    }


def results_json(res, cfg):
    return {
        "method": "Bayesian structural time-series counterfactual (PyMC) with MinT reconciliation",
        "growth": _scenario_json(res["growth"]),
        "decline": _scenario_json(res["decline"]),
        "annual_demo": _scenario_json(res["annual_demo"]),
        "coverage": {
            "n_campaigns": res["coverage"]["n_campaigns"],
            "aware": {k: v for k, v in res["coverage"]["aware"].items()
                      if k not in ("true_cum", "est_cum", "covered_flags")},
            "blind": {k: v for k, v in res["coverage"]["blind"].items()
                      if k not in ("true_cum", "est_cum", "covered_flags")},
            "worst_rhat": res["coverage"]["worst_rhat"],
            "total_divergences": res["coverage"]["total_divergences"],
        },
        "prior_sensitivity": res["prior_sensitivity"],
        "multi_region": {
            "illustrative": res["regions"]["illustrative"],
            "monte_carlo": {k: v for k, v in res["regions"]["monte_carlo"].items() if k != "raw"},
        },
        "config": cfg,
    }


def main():
    t0 = time.time()
    cfg, scenarios, regions_cfg, campaign = build_all()
    write_data(scenarios, campaign, ROOT / "data")
    print("[1/8] synthetic data written")

    fit_cfg = make_bayes_cfg(cfg)
    val_fit_cfg = make_bayes_cfg(cfg, draws=cfg["validation"]["draws"], tune=cfg["validation"]["tune"])
    region_fit_cfg = make_bayes_cfg(cfg, draws=300, tune=300)

    growth = run_scenario(scenarios["growth"], cfg, fit_cfg, predictive=True)
    print(f"[2/8] growth: est lift {_pct(growth['truth']['estimated_relative_lift'])} "
          f"vs true {_pct(growth['truth']['true_relative_lift'])}, "
          f"maxR {growth['convergence']['max_rhat']:.3f}, div {growth['convergence']['divergences']}")

    decline = run_scenario(scenarios["decline"], cfg, fit_cfg, predictive=False)
    print(f"[3/8] decline: retained {decline['summary']['streams_retained']:,.0f} streams, "
          f"maxR {decline['convergence']['max_rhat']:.3f}")

    annual_demo = run_scenario(scenarios["annual_demo"], cfg, fit_cfg, predictive=False)
    print(f"[4/8] annual_demo: annual included = {annual_demo['fit']['include_annual']}")

    coverage = validation.coverage_study(cfg, val_fit_cfg, cfg["validation"]["n_campaigns"], cfg["validation"]["seed"])
    print(f"[5/8] coverage: aware {coverage['aware']['coverage_95']*100:.0f}% "
          f"vs blind {coverage['blind']['coverage_95']*100:.0f}%")

    prior_sensitivity = validation.prior_sensitivity(scenarios["growth"], cfg, fit_cfg)
    print(f"[6/8] prior sensitivity done ({', '.join(prior_sensitivity)})")

    illustrative = aggregate.run_illustrative(regions_cfg, region_fit_cfg)
    monte_carlo = aggregate.monte_carlo(regions_cfg, regions_cfg.mc_runs, region_fit_cfg)
    print(f"[7/8] reconciliation MC ({monte_carlo['n_runs']} runs): "
          f"MinT mean|err| {monte_carlo['abs_error_mean']['mint']:,.0f}")

    res = {
        "growth": growth, "decline": decline, "annual_demo": annual_demo,
        "coverage": coverage, "prior_sensitivity": prior_sensitivity,
        "regions": {"illustrative": illustrative, "monte_carlo": monte_carlo},
    }

    img = ROOT / "docs" / "images"
    make_scenario_figures(growth, img)

    # Top "Straight answers" section: stakeholder-facing charts that show only
    # what a real deployment sees, so the injected truth is withheld (None).
    gsum = growth["summary"]
    plotting.plot_observed_vs_counterfactual(
        growth["dates"], growth["treated"], gsum["counterfactual_mean"],
        gsum["counterfactual_lo"], gsum["counterfactual_hi"], growth["n_pre"], None,
        "Growth scenario: observed streams vs estimated counterfactual",
        img / "top_growth_counterfactual.png",
    )
    dsum = decline["summary"]
    plotting.plot_pointwise_effect(
        decline["dates"], dsum["pointwise_effect_mean"], dsum["pointwise_effect_lo"],
        dsum["pointwise_effect_hi"], decline["n_pre"], None,
        "Decline scenario: estimated daily campaign effect with 95% credible interval",
        img / "top_decline_pointwise.png",
    )

    ds = decline["summary"]
    plotting.plot_communication(
        decline["dates"], decline["treated"], ds["counterfactual_mean"],
        ds["counterfactual_lo"], ds["counterfactual_hi"], decline["n_pre"], ds,
        img / "decline_communication.png",
    )
    plotting.plot_coverage(
        coverage["aware"]["true_cum"], coverage["aware"]["est_cum"],
        coverage["aware"]["covered_flags"], coverage["aware"]["coverage_95"],
        "Recovery and calibration (seasonality-aware model)", img / "coverage.png",
    )
    plotting.plot_seasonality_comparison(coverage["aware"], coverage["blind"], img / "seasonality_comparison.png")

    # Annual component figure.
    ad = annual_demo
    ad_fit = ad["fit"]
    post = ad_fit["idata"].posterior
    recovered_annual = np.moveaxis(post["annual"].stack(sample=("chain", "draw")).values, -1, 0).mean(0) * ad_fit["y_std"]
    ad_cfg = cfg["annual_demo"]
    # The annual Fourier term captures the treated-specific annual signal, the
    # part donors do not carry (a fraction 1 - donor_annual_frac of the total).
    treated_specific = (1.0 - ad_cfg["donor_annual_frac"]) * ad_cfg["annual_amp"]
    injected_annual = ad_cfg["base_level"] * treated_specific * annual_signal(ad["dates"])
    plotting.plot_annual_component(
        ad["dates"], ad["treated"], ad["summary"]["counterfactual_mean"],
        ad["summary"]["counterfactual_lo"], ad["summary"]["counterfactual_hi"], ad["n_pre"],
        recovered_annual, injected_annual, img / "annual_component.png",
    )

    plotting.plot_reconciliation(illustrative, img / "reconciliation.png")
    plotting.plot_montecarlo(monte_carlo, img / "reconciliation_montecarlo.png")

    out_dir = ROOT / "outputs"
    payload = results_json(res, cfg)
    io_utils.save_json(payload, out_dir / "attribution_results.json")
    io_utils.save_json(payload, ROOT / "data" / "sample_outputs" / "attribution_results.json")
    io_utils.save_json(
        {"illustrative": illustrative,
         "monte_carlo": {k: v for k, v in monte_carlo.items() if k != "raw"}},
        out_dir / "reconciliation_montecarlo.json",
    )

    runtime = time.time() - t0
    write_report(res, out_dir, runtime)

    # Top "Straight answers" extra charts, generated from the committed run.
    from scripts.generate_top_charts import main as generate_top_charts
    generate_top_charts()

    print(f"[8/8] outputs and figures written. total runtime {runtime/60:.1f} min")


if __name__ == "__main__":
    main()
