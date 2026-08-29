"""Reproductions for the findings in docs/BUILD_REVIEW.md.

Run: python scripts/review_repro.py

Each block is self-contained and prints the evidence quoted in the review, so a
fix can be validated by re-running this file and watching the relevant block flip.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.simplefilter("ignore")


def banner(tag: str, title: str) -> None:
    print(f"\n{'=' * 74}\n{tag} — {title}\n{'=' * 74}")


# ---------------------------------------------------------------------------
# F1 — excess cash leaks to equity; senior principal is never repaid
# ---------------------------------------------------------------------------
def f1_cash_leak() -> None:
    from calibration.vehicle.capital_stack import CapitalStack
    from calibration.vehicle.risk_mitigants import CoverageType, GrantReserve, Guarantee

    banner("F1", "Excess cash leaks to equity; senior principal is never repaid")

    # Deterministic, cash-rich: -$10M at t=0, +$3M/yr for 10 yrs (3x money back).
    cfs = np.zeros((1, 11))
    cfs[0, 0] = -10_000_000
    cfs[0, 1:] = 3_000_000

    stack = CapitalStack(
        total_capital=10_000_000,
        grant_reserve=GrantReserve(0),
        guarantee=Guarantee(0.0, CoverageType.PERCENTAGE),
        senior_coupon=0.08,
        mezzanine_coupon=0.12,
        mezzanine_fraction=0.0,
        lifetime_years=10,
    )
    tr = stack._cashflow_waterfall(cfs, 0.20)
    fl, _mz, sr = stack._build_tranches(0.20)

    print(f"Vehicle: -$10.0M at t=0, +$3.0M/yr for 10 yrs (total inflow $30.0M)")
    print(f"Senior ${sr.notional:,.0f} @ 8% coupon; first-loss ${fl.notional:,.0f}\n")
    print(f"{'t':>3} {'vehicle CF':>12} {'-> senior':>12} {'-> first_loss':>14}")
    for t in range(11):
        print(f"{t:>3} {cfs[0, t]:>12,.0f} {tr['senior'][0, t]:>12,.0f} "
              f"{tr['first_loss'][0, t]:>14,.0f}")

    sr_total = tr["senior"][0, 1:].sum()
    coupons = sr.notional * 0.08 * 10
    shortfall = sr.notional - (sr_total - coupons)
    print(f"\nSenior received      ${sr_total:>12,.0f}  on ${sr.notional:,.0f} of principal")
    print(f"Principal shortfall  ${shortfall:>12,.0f}  (unpaid at maturity)")
    print(f"First-loss received  ${tr['first_loss'][0, 1:].sum():>12,.0f}")
    print("\nEXPECTED AFTER FIX: principal shortfall = $0.")


# ---------------------------------------------------------------------------
# F2 — the correlation matrix has no effect
# ---------------------------------------------------------------------------
def f2_correlation_noop() -> None:
    from calibration.utils.stats import cholesky_correlated_draws

    banner("F2", "The correlation matrix has no effect")

    n = 20_000
    X = [np.random.default_rng(11).normal(size=n),
         np.random.default_rng(22).normal(size=n)]

    for rho in (0.0, 0.5, 0.9):
        corr = np.array([[1.0, rho], [rho, 1.0]])
        draws = cholesky_correlated_draws(n, corr, np.random.default_rng(7))

        as_built, fixed = [], []
        for j in range(2):
            target_ranks = np.argsort(np.argsort(draws[:, j]))

            # --- exactly what optimizer._simulate_vehicle does today ---
            reorder_idx = np.empty(n, dtype=int)
            reorder_idx[target_ranks] = np.arange(n)
            as_built.append(X[j][reorder_idx])

            # --- correct Iman-Conover reordering ---
            sorted_idx = np.argsort(X[j])
            fixed.append(X[j][sorted_idx[target_ranks]])

        print(f"target rho={rho:.2f} | "
              f"as-built={np.corrcoef(*as_built)[0, 1]:+.4f} | "
              f"iman-conover={np.corrcoef(*fixed)[0, 1]:+.4f}")

    print("\nEXPECTED AFTER FIX: the as-built column tracks the target column.")


def f2_end_to_end() -> None:
    from calibration.portfolio.models import PortfolioInputs
    from calibration.portfolio.optimizer import PortfolioOptimizer
    from calibration.project.models import ProjectInputs
    from calibration.vehicle.models import VehicleInputs

    banner("F2", "End to end: the UI's off-diagonal correlation slider")

    def make(rho: float) -> PortfolioInputs:
        projs = [
            ProjectInputs(base_cashflows=[-2_000_000.0] + [420_000.0] * 10,
                          price_vol=v, lifetime_years=10)
            for v in (0.30, 0.45, 0.25)
        ]
        J = len(projs)
        corr = [[1.0 if i == j else rho for j in range(J)] for i in range(J)]
        veh = VehicleInputs(projects=projs, correlation_matrix=corr,
                            total_capital=7_200_000, guarantee_coverage=0.25,
                            grant_reserve=360_000, mezzanine_fraction=0.20)
        return PortfolioInputs(vehicles=[veh], total_budget=7_200_000,
                               n_sims=400, seed=42)

    print(f"{'rho':>6} | {'alpha*':>9} | {'catalytic $':>13} | {'leverage':>9}")
    for rho in (0.0, 0.3, 0.6, 0.8):
        res = PortfolioOptimizer(make(rho)).run()
        print(f"{rho:>6.2f} | {res.catalytic_fractions[0]:>9.6f} | "
              f"{res.catalytic_allocations[0]:>13,.0f} | {res.leverage_ratio:>9.4f}")

    print("\nEXPECTED AFTER FIX: alpha* rises with rho (less diversification).")


# ---------------------------------------------------------------------------
# F3 — IRRs above 1000% are booked as total loss
# ---------------------------------------------------------------------------
def f3_irr_bracket_miss() -> None:
    import numpy_financial as npf

    from calibration.utils.irr import _R_HI, _R_LO, _irr_single, _npv_stable, clean_irr

    banner("F3", "Any IRR above 1000% is recorded as a total loss")

    # The cashflow from tests/test_project.py::test_profitable_project
    cf = np.array([-100_000.0] + [50_000 * 100 - 5_000] * 10)
    val = _irr_single(cf)

    print(f"true IRR (numpy_financial): {npf.irr(cf):.6f}   ({npf.irr(cf) * 100:.0f}%)")
    print(f"_irr_single              : {val}")
    print(f"clean_irr                : {clean_irr(np.array([val]))}        <- booked as total loss\n")

    t = np.arange(len(cf), dtype=float)
    print(f"NPV(r={_R_LO}) = {_npv_stable(_R_LO, cf, t):.3e}")
    print(f"NPV(r={_R_HI})   = {_npv_stable(_R_HI, cf, t):.3e}    <- still positive: no root in bracket")
    print(f"\nclean_irr([nan, inf, -inf, 0.5]) = "
          f"{clean_irr(np.array([np.nan, np.inf, -np.inf, 0.5]))}")
    print("The documented 10.0 cap only catches +inf, which brentq never returns.")
    print("\nEXPECTED AFTER FIX: clean_irr returns 10.0 (the cap), not -1.0.")


# ---------------------------------------------------------------------------
# F4 — CVaR collapses when losses are mostly zero
# ---------------------------------------------------------------------------
def f4_cvar_collapse() -> None:
    from calibration.utils.stats import cvar, var

    banner("F4", "CVaR collapses when losses are mostly zero")

    losses = np.zeros(1000)
    losses[:20] = np.linspace(1e6, 5e6, 20)   # senior loses in 2% of scenarios

    threshold = var(losses, 0.95)
    got = cvar(losses, 0.95)
    truth = np.mean(np.sort(losses)[-50:])    # true mean of the worst 5%

    print(f"P(loss>0)             = {np.mean(losses > 0):.3f}")
    print(f"VaR(95%)              = {threshold:,.0f}")
    print(f"cvar() as implemented = {got:>10,.0f}   <- tail set = "
          f"{int(np.sum(losses >= threshold))} of {len(losses)} paths")
    print(f"true mean of worst 5% = {truth:>10,.0f}   <- understated by "
          f"{truth / max(got, 1e-9):.0f}x")
    print("\nEXPECTED AFTER FIX: cvar() matches the true mean of the worst 5%.")


# ---------------------------------------------------------------------------
# F5 — hurdle >= senior coupon yields a degenerate alpha
# ---------------------------------------------------------------------------
def f5_degenerate_alpha() -> None:
    from calibration.vehicle.calibration import CalibratorConfig, CatalyticCalibrator
    from calibration.vehicle.capital_stack import CapitalStack
    from calibration.vehicle.risk_mitigants import CoverageType, GrantReserve, Guarantee

    banner("F5", "Hurdle at or above the senior coupon returns a degenerate alpha")

    rng = np.random.default_rng(5)
    cfs = np.empty((400, 11))
    cfs[:, 0] = -10_000_000
    cfs[:, 1:] = 3_000_000 * np.exp(rng.normal(0, 0.25, (400, 10)))

    stack = CapitalStack(
        total_capital=10_000_000,
        grant_reserve=GrantReserve(500_000),
        guarantee=Guarantee(0.25, CoverageType.PERCENTAGE),
        senior_coupon=0.08,
        mezzanine_coupon=0.12,
        mezzanine_fraction=0.10,
        lifetime_years=10,
    )

    print("Senior median IRR vs alpha (senior_coupon = 8%):")
    for a in (0.0, 0.1, 0.3, 0.5):
        r = stack.waterfall(cfs, a)["senior"]
        print(f"  alpha={a:<5} median={r.median_irr:.6f}  max over paths={r.irr_distribution.max():.6f}")
    print("  -> senior IRR is structurally capped at the coupon.\n")

    print("Consequence — the UI allows a hurdle up to 15% against an 8% coupon:")
    for hurdle in (0.06, 0.08, 0.10, 0.12):
        cal = CatalyticCalibrator(stack, cfs, CalibratorConfig(investor_hurdle_irr=hurdle))
        try:
            a = cal.calibrate()
            _fl, _mz, sr = stack._build_tranches(a)
            flag = "  <- DEGENERATE: senior tranche ~gone" if sr.notional < 0.02 * 10_000_000 else ""
            print(f"  hurdle={hurdle:>4.0%}  ->  alpha*={a:.4f}  "
                  f"senior notional=${sr.notional:>11,.0f}{flag}")
        except ValueError as exc:
            print(f"  hurdle={hurdle:>4.0%}  ->  ValueError: {exc}")

    print("\nEXPECTED AFTER FIX: hurdle >= coupon is rejected at config time.")


# ---------------------------------------------------------------------------
# F6 — mitigants cannot affect senior IRR
# ---------------------------------------------------------------------------
def f6_mitigants_inert() -> None:
    from calibration.vehicle.capital_stack import CapitalStack
    from calibration.vehicle.risk_mitigants import CoverageType, GrantReserve, Guarantee

    banner("F6", "The guarantee and grant reserve cannot affect senior IRR")

    rng = np.random.default_rng(1)
    cfs = np.empty((500, 11))
    cfs[:, 0] = -10_000_000
    cfs[:, 1:] = 1_400_000 * np.exp(rng.normal(0, 0.45, (500, 10)))

    def build(guar: float, reserve: float) -> CapitalStack:
        return CapitalStack(10_000_000, GrantReserve(reserve),
                            Guarantee(guar, CoverageType.PERCENTAGE),
                            0.08, 0.12, 0.20, 10)

    print("alpha held at 0.30:")
    print(f"{'guarantee':>10} {'reserve':>10} | {'median IRR':>11} {'loss prob':>10}")
    for guar, res in [(0.0, 0), (0.5, 0), (1.0, 0), (0.0, 2_000_000)]:
        r = build(guar, res).waterfall(cfs, 0.30)["senior"]
        print(f"{guar:>10.2f} {res:>10,} | {r.median_irr:>11.4f} {r.loss_probability:>10.3f}")

    print("\nEXPECTED AFTER FIX: median IRR rises with guarantee coverage.")


if __name__ == "__main__":
    f1_cash_leak()
    f2_correlation_noop()
    f2_end_to_end()
    f3_irr_bracket_miss()
    f4_cvar_collapse()
    f5_degenerate_alpha()
    f6_mitigants_inert()
    print(f"\n{'=' * 74}\nSee docs/BUILD_REVIEW.md for analysis and suggested fixes.\n")
