"""Verification for the findings in docs/BUILD_REVIEW.md.

Run: python scripts/review_repro.py

Each block prints the evidence for one finding and asserts the fixed behaviour,
so a regression shows up as an AssertionError rather than a plausible number.
The "was" line in each block records what the pre-fix engine produced.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.simplefilter("ignore")

FAILURES: list[str] = []


def banner(tag: str, title: str) -> None:
    print(f"\n{'=' * 74}\n{tag} — {title}\n{'=' * 74}")


def check(label: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# F1 — a solvent vehicle repays senior principal
# ---------------------------------------------------------------------------
def f1_cash_retention() -> None:
    from calibration.vehicle.capital_stack import CapitalStack
    from calibration.vehicle.risk_mitigants import CoverageType, GrantReserve, Guarantee

    banner("F1", "Cash is retained against the bullet, so senior principal is repaid")

    cfs = np.zeros((1, 11))
    cfs[0, 0] = -10_000_000
    cfs[0, 1:] = 3_000_000          # 3x money back, deterministic

    stack = CapitalStack(10_000_000, GrantReserve(0),
                         Guarantee(0.0, CoverageType.PERCENTAGE),
                         0.08, 0.12, 0.0, 10)
    tr = stack._cashflow_waterfall(cfs, 0.20)
    _fl, _mz, sr = stack._build_tranches(0.20)

    print("Vehicle: -$10.0M at t=0, +$3.0M/yr for 10 yrs (total inflow $30.0M)\n")
    print(f"{'t':>3} {'vehicle CF':>12} {'-> senior':>12} {'-> first_loss':>14}")
    for t in (0, 1, 2, 9, 10):
        print(f"{t:>3} {cfs[0, t]:>12,.0f} {tr['senior'][0, t]:>12,.0f} "
              f"{tr['first_loss'][0, t]:>14,.0f}")

    received = tr["senior"][0, 1:].sum()
    shortfall = sr.notional - (received - sr.notional * 0.08 * 10)
    irr = stack.waterfall(cfs, 0.20)["senior"].median_irr
    print(f"\n  senior principal shortfall : ${shortfall:>12,.0f}   (was $5,640,000)")
    print(f"  senior median IRR          : {irr:>13.4f}   (was 0.0008, coupon is 0.08)")
    check("senior principal fully repaid", abs(shortfall) < 1.0)
    check("senior earns its coupon", abs(irr - 0.08) < 1e-4)

    # Cash conservation: nothing may be distributed that the vehicle did not make.
    distributed = sum(tr[k][:, 1:].sum(axis=1) for k in tr)
    generated = np.maximum(0.0, cfs[:, 1:]).sum(axis=1)
    check("no money created", bool(np.all(distributed <= generated + 1e-6)))


# ---------------------------------------------------------------------------
# F2 — the correlation matrix is applied
# ---------------------------------------------------------------------------
def f2_correlation() -> None:
    from scipy.stats import spearmanr

    from calibration.utils.stats import cholesky_correlated_draws

    banner("F2", "The correlation matrix reaches the simulation")

    n = 20_000
    X = [np.random.default_rng(11).normal(size=n),
         np.random.default_rng(22).normal(size=n)]

    print("Rank correlation induced by the reordering (the statistic it controls):")
    print(f"{'target':>7} {'old code':>10} {'fixed':>10}")
    achieved = {}
    for rho in (0.0, 0.5, 0.9):
        corr = np.array([[1.0, rho], [rho, 1.0]])
        draw_corr = 2.0 * np.sin(np.pi * corr / 6.0)
        np.fill_diagonal(draw_corr, 1.0)
        draws = cholesky_correlated_draws(n, draw_corr, np.random.default_rng(7))

        old, fixed = [], []
        for j in range(2):
            tr = np.argsort(np.argsort(draws[:, j]))
            # Old code: permutation built from the draws alone.
            inv = np.empty(n, dtype=int)
            inv[tr] = np.arange(n)
            old.append(X[j][inv])
            # Fixed: map the project's own CF ordering onto the target ordering.
            fixed.append(X[j][np.argsort(X[j])[tr]])

        a_old = spearmanr(*old).statistic
        a_new = spearmanr(*fixed).statistic
        achieved[rho] = a_new
        print(f"{rho:>7.2f} {a_old:>10.4f} {a_new:>10.4f}")

    for rho, got in achieved.items():
        check(f"rank correlation {rho:.1f} achieved (got {got:.4f})", abs(got - rho) < 0.04)


def f2_end_to_end() -> None:
    from calibration.portfolio.models import PortfolioInputs
    from calibration.portfolio.optimizer import PortfolioOptimizer
    from calibration.project.models import ProjectInputs
    from calibration.vehicle.calibration import CalibratorConfig, CatalyticCalibrator
    from calibration.vehicle.models import VehicleInputs

    banner("F2", "End to end: correlation now moves the answer")

    def run(rho: float) -> tuple[float, float]:
        projs = [
            ProjectInputs(base_cashflows=[-2_000_000.0] + [300_000.0] * 10,
                          price_vol=v, lifetime_years=10)
            for v in (0.30, 0.45, 0.25)
        ]
        corr = [[1.0 if i == j else rho for j in range(3)] for i in range(3)]
        veh = VehicleInputs(projects=projs, correlation_matrix=corr,
                            total_capital=7_200_000, guarantee_coverage=0.25,
                            grant_reserve=360_000, mezzanine_fraction=0.10,
                            senior_coupon=0.08)
        pi = PortfolioInputs(vehicles=[veh], total_budget=7_200_000, n_sims=600,
                             seed=42,
                             calibrator_config=CalibratorConfig(investor_hurdle_irr=0.07))
        opt = PortfolioOptimizer(pi)
        cfs = opt._simulate_vehicle(0, 0)
        alpha = CatalyticCalibrator(
            opt._build_capital_stack(0), cfs, pi.calibrator_config
        ).calibrate()
        return alpha, float(cfs.sum(axis=1).std())

    print(f"{'rho':>6} | {'alpha*':>9} | {'vehicle CF spread':>18}")
    alphas = []
    for rho in (0.0, 0.3, 0.6, 0.8):
        a, sd = run(rho)
        alphas.append(a)
        print(f"{rho:>6.2f} | {a:>9.4f} | {sd:>18,.0f}")
    print("  (was: 0.7999 at every rho — the slider changed nothing)")
    check("alpha increases with correlation",
          all(b >= a - 1e-6 for a, b in zip(alphas, alphas[1:])) and alphas[-1] > alphas[0])


# ---------------------------------------------------------------------------
# F3 — IRR sentinels
# ---------------------------------------------------------------------------
def f3_irr() -> None:
    import numpy_financial as npf

    from calibration.utils.irr import batch_irr, clean_irr

    banner("F3", "A very high IRR is capped, not booked as a total loss")

    cf = np.array([[-100_000.0] + [4_995_000.0] * 10])
    got = clean_irr(batch_irr(cf))[0]
    print(f"  true IRR (numpy_financial): {npf.irr(cf[0]):.4f}   ({npf.irr(cf[0]) * 100:.0f}%)")
    print(f"  clean_irr                 : {got:.4f}   (was -1.0, i.e. total loss)")
    check("high IRR reports the 10.0 cap", abs(got - 10.0) < 1e-9)

    known = clean_irr(batch_irr(np.array([[-1000.0, 100, 100, 100, 100, 1100.0]])))[0]
    print(f"  analytic 10% bond         : {known:.6f}")
    check("known IRR recovered", abs(known - 0.10) < 1e-6)

    loss = clean_irr(batch_irr(np.array([[-1000.0, 0.0, 0.0]])))[0]
    check("genuine total loss still -1.0", abs(loss + 1.0) < 1e-9)


# ---------------------------------------------------------------------------
# F4 — CVaR describes the tail
# ---------------------------------------------------------------------------
def f4_cvar() -> None:
    from calibration.utils.stats import cvar, var

    banner("F4", "CVaR describes the tail even when losses are mostly zero")

    losses = np.zeros(1000)
    losses[:20] = np.linspace(1e6, 5e6, 20)
    got = cvar(losses, 0.95)
    truth = float(np.mean(np.sort(losses)[-50:]))

    print(f"  P(loss>0)             = {np.mean(losses > 0):.3f}")
    print(f"  VaR(95%)              = {var(losses, 0.95):,.0f}")
    print(f"  cvar()                = {got:>10,.0f}   (was 60,000 — understated 20x)")
    print(f"  true mean of worst 5% = {truth:>10,.0f}")
    check("CVaR matches the true tail mean", abs(got - truth) < 1e-6)


# ---------------------------------------------------------------------------
# F5 — degenerate calibrations are refused
# ---------------------------------------------------------------------------
def f5_guards() -> None:
    from calibration.vehicle.calibration import CalibratorConfig, CatalyticCalibrator
    from calibration.vehicle.capital_stack import CapitalStack
    from calibration.vehicle.risk_mitigants import CoverageType, GrantReserve, Guarantee

    banner("F5", "Unsatisfiable configurations raise instead of reporting nonsense")

    rng = np.random.default_rng(5)
    cfs = np.empty((400, 11))
    cfs[:, 0] = -10_000_000
    cfs[:, 1:] = 1_100_000 * np.exp(rng.normal(0, 0.35, (400, 10)))
    stack = CapitalStack(10_000_000, GrantReserve(500_000),
                         Guarantee(0.25, CoverageType.PERCENTAGE),
                         0.08, 0.12, 0.10, 10)

    print("  senior coupon is 8%; senior IRR cannot exceed it at any alpha\n")
    outcomes = {}
    for hurdle in (0.06, 0.07, 0.08, 0.10):
        try:
            a = CatalyticCalibrator(
                stack, cfs, CalibratorConfig(investor_hurdle_irr=hurdle)
            ).calibrate()
            _fl, _mz, sr = stack._build_tranches(a)
            outcomes[hurdle] = a
            print(f"  hurdle={hurdle:>4.0%} -> alpha*={a:.4f}  senior notional=${sr.notional:>11,.0f}")
        except ValueError as exc:
            outcomes[hurdle] = None
            print(f"  hurdle={hurdle:>4.0%} -> refused: {str(exc)[:64]}...")
    print("  (was: 0.8999 / 0.9000 with a $150-$1,398 senior tranche, reported as a solution)")

    check("hurdle below coupon still solves", outcomes[0.06] is not None)
    check("hurdle at coupon refused", outcomes[0.08] is None)
    check("hurdle above coupon refused", outcomes[0.10] is None)

    # A vehicle that returns nothing must raise, not report a degenerate alpha.
    dead = np.tile(np.array([-10_000_000.0] + [0.0] * 10), (200, 1))
    try:
        CatalyticCalibrator(stack, dead, CalibratorConfig(investor_hurdle_irr=0.07)).calibrate()
        check("total write-off vehicle refused", False)
    except ValueError:
        check("total write-off vehicle refused", True)


# ---------------------------------------------------------------------------
# F6 — mitigants reach the senior tranche
# ---------------------------------------------------------------------------
def f6_mitigants() -> None:
    from calibration.vehicle.capital_stack import CapitalStack
    from calibration.vehicle.risk_mitigants import CoverageType, GrantReserve, Guarantee

    banner("F6", "The guarantee and grant reserve reach the senior tranche")

    rng = np.random.default_rng(1)
    cfs = np.empty((500, 11))
    cfs[:, 0] = -10_000_000
    cfs[:, 1:] = 700_000 * np.exp(rng.normal(0, 0.45, (500, 10)))

    def build(guar: float, res: float) -> CapitalStack:
        return CapitalStack(10_000_000, GrantReserve(res),
                            Guarantee(guar, CoverageType.PERCENTAGE),
                            0.08, 0.12, 0.10, 10)

    print("  Cash-starved vehicle, alpha held at 0.30:")
    print(f"  {'guarantee':>10} {'reserve':>10} | {'median IRR':>11} {'loss prob':>10}")
    rows = {}
    for guar, res in [(0.0, 0), (0.5, 0), (1.0, 0), (0.0, 2_000_000)]:
        r = build(guar, res).waterfall(cfs, 0.30)["senior"]
        rows[(guar, res)] = r.median_irr
        print(f"  {guar:>10.2f} {res:>10,} | {r.median_irr:>11.4f} {r.loss_probability:>10.3f}")
    print("  (was: 0.0006 in every row — mitigants could not touch senior IRR)")

    check("guarantee raises senior IRR", rows[(0.5, 0)] > rows[(0.0, 0)])
    check("more guarantee helps more", rows[(1.0, 0)] >= rows[(0.5, 0)])
    check("grant reserve raises senior IRR", rows[(0.0, 2_000_000)] > rows[(0.0, 0)])


if __name__ == "__main__":
    f1_cash_retention()
    f2_correlation()
    f2_end_to_end()
    f3_irr()
    f4_cvar()
    f5_guards()
    f6_mitigants()

    print(f"\n{'=' * 74}")
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("All checks passed. See docs/BUILD_REVIEW.md for analysis.")
    print(f"{'=' * 74}\n")
