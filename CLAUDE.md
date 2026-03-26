# CLAUDE.md — Developer Guide for Claude Code

## Project Overview

**Calibration Tool** — a Python system that solves for the minimum catalytic capital
(first-loss, grants, guarantees) required to make a blended-finance vehicle investable
for commercial capital. The hierarchy is:

```
PROJECT → VEHICLE → PORTFOLIO
```

Catalytic capital is **never an input** — it is solved by calibration.

---

## Architecture

```
calibration/
├── project/           # Monte Carlo simulation of individual projects
│   ├── models.py      # ProjectInputs (Pydantic), ProjectResult (dataclass)
│   └── simulation.py  # ProjectSimulator: draws shocks → cashflows → IRR
├── vehicle/           # Blended-finance vehicle: waterfall + calibration
│   ├── models.py      # VehicleInputs (Pydantic), VehicleResult, TrancheResult
│   ├── risk_mitigants.py  # Guarantee (wraps senior), GrantReserve
│   ├── capital_stack.py   # Dual waterfall (cashflow + loss)
│   └── calibration.py    # CatalyticCalibrator: Brent's method / grid fallback
├── portfolio/         # LP optimizer across vehicles
│   ├── models.py      # PortfolioInputs (Pydantic), PortfolioResult (dataclass)
│   └── optimizer.py   # PortfolioOptimizer: full pipeline + cvxpy LP
└── utils/
    ├── irr.py         # batch_irr(), clean_irr() with edge-case sentinels
    ├── stats.py       # var(), cvar(), cholesky_correlated_draws()
    └── loaders.py     # load_project_from_excel(), load_vehicle_from_folder()
```

---

## Setup

```bash
pip install -e ".[dev]"
```

Dependencies: `numpy`, `scipy`, `numpy-financial`, `cvxpy`, `pydantic`, `pandas`, `pytest`.

---

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=calibration --cov-report=term-missing

# Single layer
pytest tests/test_project.py -v
pytest tests/test_vehicle.py -v
pytest tests/test_portfolio.py -v
```

All tests should pass. A warning about negative NPV on a stress-test project is expected.

---

## End-to-End Runner

```bash
python run_e2e.py                              # built-in sample data
python run_e2e.py --csv examples/             # load from CSV files
python run_e2e.py --folder examples/          # folder-per-vehicle Excel/CSV mode
python run_e2e.py --json examples/portfolio.json
python run_e2e.py --sims 5000 --seed 42
```

See `README.md` for CSV/JSON format specifications.

---

## Key Design Decisions

### 1. Loss waterfall order
Junior → Senior, following DFI practice (MIGA, DFC, GuarantCo):
```
Grant Reserve → First-Loss → Mezzanine → Guarantee (wraps Senior) → Senior
```
The guarantee is **not** a sequential layer before first-loss. It is applied to
**senior tranche losses after all subordination layers are exhausted**:
`senior_loss_gross → apply guarantee → senior_loss_net`. This matches DFI
practice (MIGA, DFC, GuarantCo guarantee senior noteholders, not catalytic capital).

### 2. Dual waterfall basis
- **Loss waterfall** operates on **NPV-based terminal loss**
  `L[s] = max(0, -NPV(CF[s], discount_rate))` where `discount_rate` defaults
  to 0.0 (undiscounted, backward-compatible). Set `discount_rate > 0` on
  `VehicleInputs` to enable time-value-of-money discounting in loss calculations.
- **Cashflow waterfall** operates **per-period** — investors receive distributions as generated.

### 3. Correlation at vehicle level
Post-simulation correlation is applied at project aggregation (vehicle level).
Rank-based reordering (Iman-Conover style) reorders each project's simulation
paths to match a target Cholesky-decomposed correlation structure. The
`corr_matrix` in `VehicleInputs` is a J×J project-level correlation matrix.

**Note:** This is an approximation — it correlates total lifetime cashflow
rankings, not the underlying stochastic drivers. A future extension would
correlate underlying risk factors (price shocks, yield shocks) at draw time.
`ProjectSimulator` does not hardcode independence assumptions and can accept
pre-correlated shocks.

### 4. Calibration algorithm
`CatalyticCalibrator.calibrate()`:
1. Monotonicity check at 20 evenly-spaced α points
2. If monotone → `scipy.optimize.brentq` (10–15 evaluations, xtol=1e-4)
3. If not monotone → two-phase grid search (50 coarse + 50 fine points)
4. Common-random-numbers: paths are pre-generated once, reused across all α evaluations

**Calibration objective note:** `_h(alpha) = min(g1, g2)` where g1 is the
IRR constraint slack and g2 is the loss-probability constraint slack. The
`min()` creates a kink where g1==g2, which is non-smooth but still valid
for Brent's method (sign change is sufficient). If the kink causes issues,
consider sequential constraint evaluation or a smooth LogSumExp surrogate.

### 5. Project cashflow simulation modes

Two mutually exclusive modes; **never mix them for the same project**:

| Mode | Trigger | Shock application |
|---|---|---|
| **Revenue/cost** | `base_revenue` provided | GBM price path × `base_revenue` per period |
| **Cashflow** | only `base_cashflows` provided | lognormal multiplier applied to positive net-CF periods |

**GBM price path** (revenue/cost mode):
```
price_index[s, t] = exp( cumsum( (μ − σ²/2) + σ·ε_t ) )
revenue[s, t] = base_revenue[t] × price_index[s, t]
```
where `μ = price_drift` (or estimated from `price_series`) and `σ = price_vol`.
This produces a proper log-normal random walk rather than a single i.i.d. scalar multiplier.

**Multi-year capex**: supply `base_cashflows` as a full array including t=0 outflow(s):
```python
base_cashflows=[-500_000, -300_000, 0, 180_000, …]  # t=0 and t=1 are construction years
```
The simulator uses the array as CF[0..T]; no scalar scaling is applied to negative periods.

### 5a. IRR sentinels
- `-1.0` → total loss (no positive inflows, capex outflow present)
- `NaN` → undefined (no investment outflow at t=0) — treated as `-1.0` after cleaning
- `10.0` → capped (outlier path with >1000% return)

### 6. Portfolio LP (Rockafellar-Uryasev)
Variables: `w_v` (allocations), `zeta` (VaR threshold), `u_s` (excess loss auxiliaries).
Objective: **maximise `∑ (1 − c_v) * w_v`** (commercial capital mobilised).
Budget constraint: `∑ c_v * w_v ≤ B_cat` (total catalytic budget).
Optional stability constraint: `∑ w_v ≥ min_deployment` (prevents degenerate all-zero solution).
Solver preference: CLARABEL → ECOS → SCS (cvxpy auto-selects).

**`marginal_catalytic_efficiency`** on `VehicleResult` reports `(1 − α*) / α*` — the commercial
capital mobilised per catalytic dollar at the minimum binding point. This equals `leverage_ratio`
and is the vehicle-level shadow price of the catalytic budget constraint.

---

## Adding a New Project Type

1. Define risk factors in `ProjectInputs` (add Pydantic fields).
2. Update `ProjectSimulator.run()` to draw and apply new shocks.
3. No changes needed in vehicle or portfolio layers — they consume `cashflows: ndarray`.

## Adding a New Risk Mitigant

1. Add a class to `calibration/vehicle/risk_mitigants.py` with an `absorb(loss)` method.
2. Insert it as a new layer in `CapitalStack._loss_waterfall()`.
3. Update `CapitalStack.waterfall()` if it also affects cashflow distribution.

## Changing the Optimization Objective

Edit `PortfolioOptimizer._solve_lp()` in `calibration/portfolio/optimizer.py`.
The current objective is `max ∑(1-c_v)*w_v` (commercial capital mobilised subject to catalytic
budget `∑c_v*w_v ≤ B_cat`). Alternative: minimise total catalytic cost `min ∑c_v*w_v`, or
minimise weighted-average catalytic fraction.

---

## Common Pitfalls

- **Bullet maturity model**: principal returns entirely at `t=T`. For short-life projects
  with steady cashflows, ensure terminal cashflow is large enough to repay senior principal.
  If your project has back-loaded revenue, this works naturally. Otherwise, increase
  `total_capital / T * multiplier` to ensure terminal cash is sufficient.

- **Infeasible calibration**: if `CatalyticCalibrator` raises `ValueError: Constraints infeasible`,
  either (a) relax `investor_hurdle_irr` / `max_loss_probability` in `CalibratorConfig`,
  or (b) add more protective mitigants (higher `guarantee_coverage` or `grant_reserve`).

- **CVaR constraint too tight**: if the portfolio LP returns status `infeasible`, increase
  `cvar_max` in `PortfolioInputs` or reduce `cvar_confidence`. Also check that
  `min_deployment` is not set too high relative to `total_budget`.

- **All-zero LP allocation**: if the optimizer returns zero weights for all vehicles, set
  `min_deployment > 0` in `PortfolioInputs` to enforce a minimum total deployment.

- **Correlation matrix not PD**: `cholesky_correlated_draws()` automatically applies Higham's
  nearest-PD projection. Check the warning log if correlations were modified.

- **Mixed cashflow/revenue modes**: do not provide both `base_cashflows` and `base_revenue` for
  the same project. If `base_revenue` is set, the simulator uses GBM price paths and ignores
  `base_cashflows` multiplier logic entirely (mode consistency guard).

- **Excel ingestion with revenue/cost columns**: if your Excel sheet has `revenue` and `cost`
  columns, `load_project_from_excel()` populates `base_revenue` and `base_costs` directly
  (not `base_cashflows`). Ensure `price_vol` is set in the sheet or passed as a kwarg.
