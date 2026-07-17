"""Regenerate the figures in docs/images.

This reruns the Bayesian fits (MCMC is required to produce the counterfactual
posteriors) and redraws every figure. Outputs and the report are left
untouched. A lighter sampler and a smaller coverage study are used here so the
figure refresh is quicker than the full `run_demo.py`.

    python scripts/generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import aggregate, plotting, validation  # noqa: E402
from src.data_gen import annual_signal  # noqa: E402
from scripts.generate_data import build_all  # noqa: E402
from scripts.run_demo import make_bayes_cfg, make_scenario_figures, run_scenario  # noqa: E402


def make_top_section_figures(growth, decline, img):
    """Stakeholder-facing charts for the top "Straight answers" section.

    These show only what a real deployment would see, so the injected true
    counterfactual and true effect are deliberately withheld (passed as None).
    The ground-truth overlays stay in the technical-section figures.
    """
    gs = growth["summary"]
    plotting.plot_observed_vs_counterfactual(
        growth["dates"], growth["treated"], gs["counterfactual_mean"],
        gs["counterfactual_lo"], gs["counterfactual_hi"], growth["n_pre"],
        None,
        "Growth scenario: observed streams vs estimated counterfactual",
        img / "top_growth_counterfactual.png",
    )
    dsm = decline["summary"]
    plotting.plot_pointwise_effect(
        decline["dates"], dsm["pointwise_effect_mean"], dsm["pointwise_effect_lo"],
        dsm["pointwise_effect_hi"], decline["n_pre"], None,
        "Decline scenario: estimated daily campaign effect with 95% credible interval",
        img / "top_decline_pointwise.png",
    )


def main():
    cfg, scenarios, regions_cfg, campaign = build_all()
    fit_cfg = make_bayes_cfg(cfg, draws=600, tune=600)
    val_fit_cfg = make_bayes_cfg(cfg, draws=400, tune=400)
    img = ROOT / "docs" / "images"

    growth = run_scenario(scenarios["growth"], cfg, fit_cfg, predictive=True)
    make_scenario_figures(growth, img)

    decline = run_scenario(scenarios["decline"], cfg, fit_cfg, predictive=False)
    make_top_section_figures(growth, decline, img)
    ds = decline["summary"]
    plotting.plot_communication(
        decline["dates"], decline["treated"], ds["counterfactual_mean"],
        ds["counterfactual_lo"], ds["counterfactual_hi"], decline["n_pre"], ds,
        img / "decline_communication.png",
    )

    annual_demo = run_scenario(scenarios["annual_demo"], cfg, fit_cfg, predictive=False)
    post = annual_demo["fit"]["idata"].posterior
    recovered_annual = np.moveaxis(
        post["annual"].stack(sample=("chain", "draw")).values, -1, 0
    ).mean(0) * annual_demo["fit"]["y_std"]
    _adc = cfg["annual_demo"]
    injected_annual = _adc["base_level"] * (1.0 - _adc["donor_annual_frac"]) * _adc["annual_amp"] * annual_signal(annual_demo["dates"])
    plotting.plot_annual_component(
        annual_demo["dates"], annual_demo["treated"], annual_demo["summary"]["counterfactual_mean"],
        annual_demo["summary"]["counterfactual_lo"], annual_demo["summary"]["counterfactual_hi"],
        annual_demo["n_pre"], recovered_annual, injected_annual, img / "annual_component.png",
    )

    coverage = validation.coverage_study(cfg, val_fit_cfg, 16, cfg["validation"]["seed"])
    plotting.plot_coverage(
        coverage["aware"]["true_cum"], coverage["aware"]["est_cum"],
        coverage["aware"]["covered_flags"], coverage["aware"]["coverage_95"],
        "Recovery and calibration (seasonality-aware model)", img / "coverage.png",
    )
    plotting.plot_seasonality_comparison(coverage["aware"], coverage["blind"], img / "seasonality_comparison.png")

    illustrative = aggregate.run_illustrative(regions_cfg, val_fit_cfg)
    monte_carlo = aggregate.monte_carlo(regions_cfg, min(regions_cfg.mc_runs, 12), val_fit_cfg)
    plotting.plot_reconciliation(illustrative, img / "reconciliation.png")
    plotting.plot_montecarlo(monte_carlo, img / "reconciliation_montecarlo.png")

    # Top "Straight answers" extra charts read the committed outputs and data.
    from scripts.generate_top_charts import main as generate_top_charts
    generate_top_charts()

    print(f"Figures regenerated in {img}")


if __name__ == "__main__":
    main()
