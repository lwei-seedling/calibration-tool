"""Interactive end-to-end validation script for the Calibration Tool.

Runs a structured 6-step validation of the full Project → Vehicle → Portfolio
pipeline using the mock portfolio dataset (examples/mock_portfolio/).

Designed for junior developers and analysts: single command, clear pass/fail,
human-readable output with economic interpretation of each result.

Usage
-----
    python validate_e2e.py                     # default: 1 000 sims, seed 42
    python validate_e2e.py --sims 500          # fewer sims for quick runs
    python validate_e2e.py --sims 2000 --seed 7
    python validate_e2e.py --charts            # also produce matplotlib charts

What it checks
--------------
  Step 1 — Data loading:       3 vehicles × 3 projects loaded and validated
  Step 2 — Deterministic:      cashflow integrity, waterfall ordering, edge cases
  Step 3 — Monte Carlo:        IRR stats, loss rates, no overflow warnings
  Step 4 — Sensitivity:        higher guarantee / vol / lower CFs → expected direction
  Step 5 — Diagnostics:        NaN ratios, IRR percentiles, loss statistics
  Step 6 — Summary output:     per-vehicle table, optional charts
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import run_e2e as _e2e  # noqa: E402

from calibration.portfolio.optimizer import PortfolioOptimizer
from calibration.project.models import ProjectInputs
from calibration.project.simulation import ProjectSimulator
from calibration.utils.irr import batch_irr, clean_irr
from calibration.vehicle.capital_stack import CapitalStack
from calibration.vehicle.risk_mitigants import CoverageType, Guarantee, GrantReserve

_MOCK_DIR = _ROOT / "examples" / "mock_portfolio"

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_PASS = "✓ PASS"
_FAIL = "✗ FAIL"
_WARN = "⚠ WARN"


def _header(text: str) -> None:
    bar = "═" * 70
    print(f"\n{bar}")
    print(f"  {text}")
    print(bar)


def _sub(text: str) -> None:
    print(f"\n  {text}")
    print("  " + "─" * 60)


def _check(label: str, passed: bool, detail: str = "") -> bool:
    status = _PASS if passed else _FAIL
    line = f"  {status}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return passed


def _run_silent(inputs):
    """Run PortfolioOptimizer suppressing expected RuntimeWarnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return PortfolioOptimizer(inputs).run()


# ---------------------------------------------------------------------------
# Sensitivity helpers
# ---------------------------------------------------------------------------

def _rebuild_portfolio(base_inputs, vehicle_overrides: dict[int, dict]):
    """Return a new PortfolioInputs with per-vehicle field overrides."""
    new_vehicles = []
    for i, v in enumerate(base_inputs.vehicles):
        ov = vehicle_overrides.get(i, {})
        new_vehicles.append(v.model_copy(update=ov) if ov else v)
    return base_inputs.model_copy(update={"vehicles": new_vehicles})


def _scale_cashflows(base_inputs, vehicle_idx: int, scale: float):
    """Scale positive cashflows for all projects in one vehicle."""
    v = base_inputs.vehicles[vehicle_idx]
    new_projects = [
        p.model_copy(update={
            "base_cashflows": [cf * scale if cf > 0 else cf for cf in p.base_cashflows]
        })
        for p in v.projects
    ]
    new_v = v.model_copy(update={"projects": new_projects})
    new_vehicles = list(base_inputs.vehicles)
    new_vehicles[vehicle_idx] = new_v
    return base_inputs.model_copy(update={"vehicles": new_vehicles})


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------

def validate(n_sims: int = 1000, seed: int = 42, charts: bool = False) -> bool:
    """Run all 6 validation steps. Returns True if all checks pass."""
    checks: list[bool] = []
    t_start = time.time()

    # -----------------------------------------------------------------------
    # STEP 1 — Data Loading
    # -----------------------------------------------------------------------
    _header("STEP 1: DATA LOADING")

    inputs, names = _e2e._load_folder_inputs(_MOCK_DIR, n_sims=n_sims, seed=seed)

    checks.append(_check("3 vehicles loaded", len(inputs.vehicles) == 3,
                         f"got {len(inputs.vehicles)}"))
    checks.append(_check("9 projects loaded (3 per vehicle)",
                         all(len(v.projects) == 3 for v in inputs.vehicles)))
    checks.append(_check("All projects have 16 cashflow periods (years 0–15)",
                         all(len(p.base_cashflows) == 16
                             for v in inputs.vehicles for p in v.projects)))
    checks.append(_check("Year-0 cashflow is negative (capex outflow) for all projects",
                         all(p.base_cashflows[0] < 0
                             for v in inputs.vehicles for p in v.projects)))

    print()
    for v, n in zip(inputs.vehicles, names):
        total_capex = sum(abs(p.base_cashflows[0]) for p in v.projects)
        print(f"    {n}: {len(v.projects)} projects | "
              f"total capex ${total_capex/1e6:.1f}M | "
              f"vehicle capital ${v.total_capital/1e6:.1f}M")

    # -----------------------------------------------------------------------
    # STEP 2 — Deterministic Validation
    # -----------------------------------------------------------------------
    _header("STEP 2: DETERMINISTIC VALIDATION  (sims=100, seed=1)")

    det_inputs, _ = _e2e._load_folder_inputs(_MOCK_DIR, n_sims=100, seed=1)

    # A. Cashflow integrity — capex must be unchanged across paths
    capex_ok = True
    for v in det_inputs.vehicles:
        for p in v.projects:
            sim = ProjectSimulator(p)
            res = sim.run(n_sims=100, seed=1)
            if not np.allclose(res.cashflows[:, 0], res.cashflows[0, 0]):
                capex_ok = False
    checks.append(_check(
        "Year-0 capex is deterministic (identical across all simulation paths)",
        capex_ok,
    ))

    # B. Multi-year construction — forestry years 1 & 2 are deterministic outflows
    v0 = det_inputs.vehicles[0]
    p_arr = v0.projects[0]  # forestry_arr: year 1 = -500k, year 2 = -500k
    sim_arr = ProjectSimulator(p_arr)
    res_arr = sim_arr.run(n_sims=100, seed=1)
    multi_yr_ok = (
        np.allclose(res_arr.cashflows[:, 1], -500_000) and
        np.allclose(res_arr.cashflows[:, 2], -500_000)
    )
    checks.append(_check("Multi-year construction cashflows preserved (forestry yr 1 & 2)", multi_yr_ok))

    # C. Negative-NPV project flows through (not filtered)
    p_risky = v0.projects[2]  # forestry_risky: highest capex, delayed revenue
    sim_risky = ProjectSimulator(p_risky)
    res_risky = sim_risky.run(n_sims=100, seed=1)
    checks.append(_check(
        "Negative-NPV project (forestry_risky) flows through without filtering",
        res_risky.cashflows.shape[0] == 100,
        f"loss_prob={res_risky.loss_probability:.1%}",
    ))

    # D. Waterfall ordering — first-loss absorbs before senior
    n_det = det_inputs.n_sims
    v_det = det_inputs.vehicles[0]
    vehicle_cfs_det = np.zeros((n_det, 16))
    for p in v_det.projects:
        sim_p = ProjectSimulator(p)
        res_p = sim_p.run(n_sims=n_det, seed=1)
        L = min(vehicle_cfs_det.shape[1], res_p.cashflows.shape[1])
        vehicle_cfs_det[:, :L] += res_p.cashflows[:, :L]

    T_det = max(p.lifetime_years for p in v_det.projects)
    stack_det = CapitalStack(
        total_capital=v_det.total_capital,
        grant_reserve=GrantReserve(v_det.grant_reserve),
        guarantee=Guarantee(v_det.guarantee_coverage, CoverageType.PERCENTAGE),
        senior_coupon=v_det.senior_coupon,
        mezzanine_coupon=v_det.mezzanine_coupon,
        mezzanine_fraction=v_det.mezzanine_fraction,
        lifetime_years=T_det,
        discount_rate=v_det.discount_rate,
    )
    tranches = stack_det.waterfall(vehicle_cfs_det, alpha=0.40)
    senior_loss = tranches["senior"].loss_distribution
    fl_loss = tranches["first_loss"].loss_distribution
    waterfall_ok = bool(np.all(senior_loss <= fl_loss + 1e-8))
    checks.append(_check(
        "Waterfall ordering: first-loss absorbs before senior on all paths",
        waterfall_ok,
    ))

    # -----------------------------------------------------------------------
    # STEP 3 — Monte Carlo Validation
    # -----------------------------------------------------------------------
    _header(f"STEP 3: MONTE CARLO VALIDATION  (sims={n_sims}, seed={seed})")

    print(f"\n  Running calibration and optimisation... ", end="", flush=True)
    t0 = time.time()

    overflow_warnings: list[str] = []
    original_showwarning = warnings.showwarning

    def _capture_overflow(message, category, filename, lineno, file=None, line=None):
        msg = str(message)
        if issubclass(category, RuntimeWarning) and (
            "overflow" in msg.lower() or "power" in msg.lower()
        ):
            overflow_warnings.append(msg)
        else:
            original_showwarning(message, category, filename, lineno, file, line)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = _capture_overflow
        result = PortfolioOptimizer(inputs).run()

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s")

    checks.append(_check("LP solver status: optimal", result.status == "optimal",
                         f"status={result.status!r}"))
    checks.append(_check("No overflow/power RuntimeWarnings", len(overflow_warnings) == 0,
                         f"{len(overflow_warnings)} warnings" if overflow_warnings else ""))
    checks.append(_check("Portfolio loss distribution is non-negative",
                         bool(np.all(result.portfolio_loss_distribution >= 0))))
    checks.append(_check("Leverage ratio is positive", result.leverage_ratio > 0,
                         f"{result.leverage_ratio:.2f}x"))

    irr = result.portfolio_irr_distribution
    p5 = float(np.nanpercentile(irr, 5))
    p50 = float(np.nanpercentile(irr, 50))
    p95 = float(np.nanpercentile(irr, 95))
    nan_pct = float(np.mean(np.isnan(irr))) * 100
    irr_ok = (p5 >= -1.0) and (p50 > -0.5) and (p95 <= 10.0) and (p5 <= p50 <= p95)
    checks.append(_check(
        "IRR percentiles within economic bounds",
        irr_ok,
        f"p5={p5:.1%}  p50={p50:.1%}  p95={p95:.1%}",
    ))

    # Per-vehicle summary table
    _sub("Per-Vehicle Results")
    print(f"  {'Vehicle':<28} {'Alpha':>6} {'Med IRR':>8} {'Loss%':>7} "
          f"{'NaN%':>6} {'Leverage':>9} {'Marg Eff':>9}")
    print("  " + "-" * 73)
    for v_idx, name in enumerate(names):
        alpha = result.catalytic_fractions[v_idx]
        # Compute vehicle-level senior IRR from the actual tranche via a
        # quick waterfall call at the calibrated alpha
        v = inputs.vehicles[v_idx]
        n_q = 200
        v_cfs = np.zeros((n_q, 16))
        for p in v.projects:
            sim_p = ProjectSimulator(p)
            res_p = sim_p.run(n_sims=n_q, seed=seed + v_idx)
            L = min(v_cfs.shape[1], res_p.cashflows.shape[1])
            v_cfs[:, :L] += res_p.cashflows[:, :L]
        T_v = max(p.lifetime_years for p in v.projects)
        stack_v = CapitalStack(
            total_capital=v.total_capital,
            grant_reserve=GrantReserve(v.grant_reserve),
            guarantee=Guarantee(v.guarantee_coverage, CoverageType.PERCENTAGE),
            senior_coupon=v.senior_coupon,
            mezzanine_coupon=v.mezzanine_coupon,
            mezzanine_fraction=v.mezzanine_fraction,
            lifetime_years=T_v,
            discount_rate=v.discount_rate,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tr = stack_v.waterfall(v_cfs, alpha)
        senior_irr = tr["senior"].irr_distribution
        med_irr = float(np.nanmedian(senior_irr))
        loss_pct = float(tr["senior"].loss_probability) * 100
        nan_r = float(np.mean(np.isnan(senior_irr))) * 100
        lev = result.marginal_catalytic_efficiency.get(v_idx, 0.0)

        print(f"  {name:<28} {alpha:>5.1%}  {med_irr:>7.1%}  {loss_pct:>6.1f}%  "
              f"{nan_r:>5.1f}%  {lev:>8.2f}x  {lev:>8.2f}x")

    print(f"\n  Portfolio leverage:  {result.leverage_ratio:.2f}x")
    print(f"  Portfolio CVaR 95%: {result.cvar_95:.1%}")
    print(f"  Portfolio NaN%:     {nan_pct:.1f}%")
    if nan_pct == 0.0:
        print(f"  ⚠  NaN% = 0 — consider whether edge cases are being silently handled")
    elif nan_pct > 40.0:
        print(f"  ⚠  NaN% > 40% — assumptions may be unrealistic")

    # -----------------------------------------------------------------------
    # STEP 4 — Sensitivity Checks
    # -----------------------------------------------------------------------
    _header("STEP 4: SENSITIVITY CHECKS  (sims=200)")

    sens_inputs, _ = _e2e._load_folder_inputs(_MOCK_DIR, n_sims=200, seed=seed)
    base_sens = _run_silent(sens_inputs)
    base_mean_alpha = np.mean(list(base_sens.catalytic_fractions.values()))
    base_irr_std = float(np.nanstd(base_sens.portfolio_irr_distribution))

    _sub("Test A — Increase guarantee coverage: 25% → 50%")
    ov = {i: {"guarantee_coverage": 0.50} for i in range(3)}
    high_guar_r = _run_silent(_rebuild_portfolio(sens_inputs, ov))
    high_alpha = np.mean(list(high_guar_r.catalytic_fractions.values()))
    delta_a = high_alpha - base_mean_alpha
    a_ok = high_alpha <= base_mean_alpha + 0.05
    checks.append(_check(
        f"Higher guarantee → alpha decreases or stays flat",
        a_ok,
        f"base={base_mean_alpha:.1%} → high_guar={high_alpha:.1%} (Δ{delta_a:+.1%})",
    ))

    _sub("Test B — Increase price volatility: 0.15 → 0.40")
    high_vol_i = sens_inputs
    for vi in range(3):
        v = high_vol_i.vehicles[vi]
        new_ps = [p.model_copy(update={"price_vol": 0.40}) for p in v.projects]
        new_v = v.model_copy(update={"projects": new_ps})
        nvs = list(high_vol_i.vehicles)
        nvs[vi] = new_v
        high_vol_i = high_vol_i.model_copy(update={"vehicles": nvs})
    high_vol_r = _run_silent(high_vol_i)
    high_std = float(np.nanstd(high_vol_r.portfolio_irr_distribution))
    delta_std = high_std - base_irr_std
    b_ok = high_std >= base_irr_std * 0.9
    checks.append(_check(
        "Higher price vol → IRR dispersion increases",
        b_ok,
        f"base std={base_irr_std:.4f} → high vol std={high_std:.4f} (Δ{delta_std:+.4f})",
    ))

    _sub("Test C — Reduce vehicle-0 cashflows by 20%")
    low_cf_i = _scale_cashflows(sens_inputs, vehicle_idx=0, scale=0.80)
    low_cf_r = _run_silent(low_cf_i)
    low_alpha_v0 = low_cf_r.catalytic_fractions[0]
    base_alpha_v0 = base_sens.catalytic_fractions[0]
    delta_c = low_alpha_v0 - base_alpha_v0
    c_ok = low_alpha_v0 >= base_alpha_v0 - 0.05
    checks.append(_check(
        "Lower cashflows (×0.80) → alpha increases or stays flat",
        c_ok,
        f"base={base_alpha_v0:.1%} → low CF={low_alpha_v0:.1%} (Δ{delta_c:+.1%})",
    ))

    # -----------------------------------------------------------------------
    # STEP 5 — Diagnostics
    # -----------------------------------------------------------------------
    _header("STEP 5: DIAGNOSTICS")

    nan_ratio = float(np.mean(np.isnan(result.portfolio_irr_distribution)))
    loss = result.portfolio_loss_distribution
    mean_loss = float(np.mean(loss))
    median_loss = float(np.median(loss))
    p95_loss = float(np.percentile(loss, 95))
    p99_loss = float(np.percentile(loss, 99))

    print(f"\n  IRR NaN ratio:    {nan_ratio:.1%}")
    nan_status = (
        _WARN + " ratio = 0% (check edge-case handling)" if nan_ratio == 0.0
        else _WARN + f" ratio > 40% — may indicate unrealistic assumptions" if nan_ratio > 0.40
        else _PASS + " within expected range [0%, 40%]"
    )
    print(f"  {nan_status}")

    print(f"\n  IRR distribution:  p5={p5:.1%}  p50={p50:.1%}  p95={p95:.1%}")

    print(f"\n  Loss rate statistics:")
    print(f"    Mean   : {mean_loss:.2%}")
    print(f"    Median : {median_loss:.2%}")
    print(f"    p95    : {p95_loss:.2%}")
    print(f"    p99    : {p99_loss:.2%}")

    loss_ok = (mean_loss >= 0 and median_loss >= 0 and p95_loss >= 0
               and p99_loss <= 1.0 + 1e-9)
    checks.append(_check("Loss statistics are non-negative and at most 100%", loss_ok))

    # IrrDiagnostics API check
    p0 = inputs.vehicles[0].projects[0]
    sim_diag = ProjectSimulator(p0)
    res_diag = sim_diag.run(n_sims=200, seed=seed)
    _, diag = batch_irr(res_diag.cashflows, return_diagnostics=True)
    checks.append(_check(
        "IrrDiagnostics API: n_computed correct",
        diag.n_computed == 200,
        f"n_computed={diag.n_computed}, no_sign_change={diag.n_no_sign_change}, failures={diag.n_failures}",
    ))

    # -----------------------------------------------------------------------
    # STEP 6 — Optional Charts
    # -----------------------------------------------------------------------
    if charts:
        _header("STEP 6: CHARTS")
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # IRR histogram
            ax = axes[0]
            irr_clean = clean_irr(result.portfolio_irr_distribution)
            irr_plot = irr_clean[irr_clean > -1.0]
            ax.hist(irr_plot * 100, bins=40, edgecolor="white", color="steelblue")
            ax.axvline(np.nanmedian(irr_plot) * 100, color="red", linestyle="--",
                       label=f"Median {np.nanmedian(irr_plot):.1%}")
            ax.set_xlabel("IRR (%)")
            ax.set_ylabel("Frequency")
            ax.set_title("Portfolio IRR Distribution")
            ax.legend()

            # Loss distribution
            ax2 = axes[1]
            ax2.hist(loss * 100, bins=40, edgecolor="white", color="coral")
            ax2.axvline(float(np.percentile(loss, 95)) * 100, color="darkred",
                        linestyle="--", label=f"CVaR95 {result.cvar_95:.1%}")
            ax2.set_xlabel("Loss Rate (%)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Portfolio Loss Distribution")
            ax2.legend()

            plt.tight_layout()
            chart_path = _ROOT / "validation_charts.png"
            plt.savefig(chart_path, dpi=120)
            print(f"\n  Charts saved to: {chart_path}")
            plt.close()
        except ImportError:
            print("\n  matplotlib not installed — install with: pip install matplotlib")
            print("  Skipping charts.")
    else:
        _header("STEP 6: CHARTS  (skipped — pass --charts to enable)")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - t_start
    n_pass = sum(checks)
    n_total = len(checks)

    bar = "═" * 70
    print(f"\n{bar}")
    if n_pass == n_total:
        print(f"  ✓ ALL {n_total} CHECKS PASSED  ({total_elapsed:.0f}s)")
    else:
        print(f"  ✗ {n_total - n_pass} of {n_total} CHECKS FAILED  ({total_elapsed:.0f}s)")
    print(bar)

    return n_pass == n_total


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end validation of the Calibration Tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sims", type=int, default=1000,
                   help="Number of Monte Carlo simulations for main run (default 1000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default 42)")
    p.add_argument("--charts", action="store_true",
                   help="Produce matplotlib charts (requires matplotlib)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ok = validate(n_sims=args.sims, seed=args.seed, charts=args.charts)
    sys.exit(0 if ok else 1)
