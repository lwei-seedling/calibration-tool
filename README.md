# Calibration Tool

**Decision engine for allocating catalytic capital across blended-finance portfolios.**

It solves the core question in blended finance: *how much concessional capital is the minimum needed to make a vehicle commercially investable?* — and then optimises its allocation across a portfolio of vehicles to maximise leverage (commercial capital mobilised per catalytic dollar).

The system models the full hierarchy: **PROJECT → VEHICLE → PORTFOLIO**.
- **Calibration** solves the minimum catalytic fraction per vehicle using Brent's method.
- **Portfolio optimisation** allocates capital across vehicles via a CVaR-constrained LP.
- **Catalytic capital is never an input** — it is always a solved output.

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Run with built-in sample data

```bash
python run_e2e.py
```

This runs a two-vehicle, six-project portfolio (East Africa Nature Fund + West Africa Clean Energy Fund) and prints the calibrated catalytic fractions, leverage ratios, and CVaR statistics.

### 3. Run the test suite

```bash
pytest
```

All 23 tests should pass in under 5 seconds.

---

## Loading Your Own Data

### Option A — CSV files

Two CSV sub-formats are supported and **auto-detected** per file.

#### A1. Cashflow-based format (recommended for detailed forecasts)

Each file represents one project. Columns: `year` and `cashflow`.

| Column | Description |
|---|---|
| `year` | Period index (0 = capex outflow, 1..T = operating periods) |
| `cashflow` | Net cashflow for that period (negative at year 0 = capex) |

**Example** (`examples/cashflow_vehicle.csv`):

```csv
year,cashflow
0,-1500000
1,180000
2,210000
3,240000
4,260000
5,270000
10,1650000
```

Run:
```bash
python run_e2e.py --csv examples/
```

The simulator scales the supplied cashflows per path using a lognormal multiplier
(sigma = `price_vol`, default 0.15). Set `price_vol` via JSON for tighter/looser
volatility. This mode is ideal when you have a financial model that produces an
explicit year-by-year cashflow forecast.

#### A2. Parametric format (for factor-driven Monte Carlo)

Each file can contain multiple projects (one row per project). Columns below.

Create one CSV file per vehicle named `projects_vehicle_1.csv`, `projects_vehicle_2.csv`, etc. Place them in a directory (e.g. `examples/`).

**Required columns:**

| Column | Type | Description |
|---|---|---|
| `capex` | float | Capital expenditure (positive, e.g. `1500000`) |
| `opex_annual` | float | Annual operating cost (positive, e.g. `60000`) |
| `price` | float | Base output price per unit |
| `yield_` | float | Base annual output quantity |
| `lifetime_years` | int | Project lifetime (1–50) |

**Optional columns** (defaults used if omitted):

| Column | Default | Description |
|---|---|---|
| `price_vol` | 0.15 | Annual price volatility (lognormal sigma) |
| `yield_vol` | 0.10 | Annual yield volatility |
| `inflation_rate` | 0.03 | Annual opex inflation |
| `fx_vol` | 0.05 | Annual FX volatility on revenue |
| `delay_prob` | 0.05 | Per-period probability of production delay |

**Example** (`examples/projects_vehicle_1.csv`):

```csv
name,capex,opex_annual,price,yield_,lifetime_years,price_vol,yield_vol,inflation_rate,fx_vol,delay_prob
Solar Farm Kenya,2000000,80000,45.0,60000,15,0.12,0.08,0.04,0.06,0.03
Agroforestry Tanzania,800000,35000,12.0,80000,10,0.18,0.15,0.05,0.08,0.07
Blue Carbon Mozambique,1200000,50000,25.0,55000,12,0.20,0.12,0.04,0.10,0.05
```

Run:

```bash
python run_e2e.py --csv examples/
```

> **Note:** When loading from CSV, vehicle-level settings (correlation matrix, total capital, guarantee coverage, grant reserve) are set to conservative defaults. For full control, use the JSON format.

---

### Option B — JSON portfolio spec

A single JSON file specifies the entire portfolio including vehicle-level parameters.

**Top-level fields:**

| Field | Type | Description |
|---|---|---|
| `total_budget` | float | Total portfolio capital |
| `n_sims` | int | Monte Carlo paths (default 1000; use 5000+ for production) |
| `seed` | int or null | Random seed |
| `cvar_confidence` | float | CVaR confidence level (default 0.95) |
| `cvar_max` | float | Maximum portfolio CVaR as loss rate (default 0.30) |
| `min_expected_return` | float | Minimum expected portfolio return (default 0.0) |
| `max_allocation_fraction` | float | Max fraction of budget to one vehicle (default 1.0) |
| `calibrator_config` | object | Calibration constraints (see below) |
| `vehicles` | array | List of vehicle objects |

**`calibrator_config` fields:**

| Field | Default | Description |
|---|---|---|
| `investor_hurdle_irr` | 0.08 | Minimum median senior tranche IRR |
| `max_loss_probability` | 0.05 | Maximum senior tranche loss probability |
| `alpha_lo` | 0.0 | Lower bound for catalytic fraction search |
| `alpha_hi` | 0.99 | Upper bound for catalytic fraction search |

**Each vehicle object:**

| Field | Type | Description |
|---|---|---|
| `name` | string | Display name (optional) |
| `total_capital` | float | Total vehicle capital |
| `guarantee_coverage` | float | Guarantee as fraction of senior notional (0–1) |
| `grant_reserve` | float | Grant reserve amount (absolute $) |
| `mezzanine_fraction` | float | Fraction of total capital in mezzanine tranche |
| `senior_coupon` | float | Senior annual coupon rate (default 0.08) |
| `mezzanine_coupon` | float | Mezzanine annual coupon rate (default 0.12) |
| `discount_rate` | float | Discount rate for NPV-based loss (default 0.0 = sum-based) |
| `correlation_matrix` | J×J array | Project-level correlation matrix |
| `projects` | array | List of project objects (same fields as parametric CSV, plus `base_cashflows`) |

**Example** (`examples/portfolio.json`):

```json
{
  "total_budget": 10000000,
  "n_sims": 1000,
  "seed": 42,
  "calibrator_config": {
    "investor_hurdle_irr": 0.07,
    "max_loss_probability": 0.08
  },
  "vehicles": [
    {
      "name": "East Africa Nature Fund",
      "total_capital": 4000000,
      "guarantee_coverage": 0.30,
      "grant_reserve": 200000,
      "mezzanine_fraction": 0.10,
      "senior_coupon": 0.08,
      "mezzanine_coupon": 0.13,
      "correlation_matrix": [[1.0, 0.35, 0.20], [0.35, 1.0, 0.40], [0.20, 0.40, 1.0]],
      "projects": [
        {"capex": 2000000, "opex_annual": 80000, "price": 45.0, "yield_": 60000,
         "lifetime_years": 15, "price_vol": 0.12, "yield_vol": 0.08,
         "inflation_rate": 0.04, "fx_vol": 0.06, "delay_prob": 0.03}
      ]
    }
  ]
}
```

Run:

```bash
python run_e2e.py --json examples/portfolio.json

# Override simulations and seed at runtime
python run_e2e.py --json examples/portfolio.json --sims 5000 --seed 123
```

---

## Runner Options

```
usage: run_e2e.py [-h] [--csv DIR | --json FILE] [--sims SIMS] [--seed SEED]

optional arguments:
  --csv DIR      Directory with projects_vehicle_N.csv files
  --json FILE    JSON file with full portfolio specification
  --sims SIMS    Number of Monte Carlo simulations (default: 1000)
  --seed SEED    Random seed for reproducibility
```

---

## Understanding the Output

```
══════════════════════════════════════════════════════════════════════
  CATALYTIC CAPITAL CALIBRATION — RESULTS
══════════════════════════════════════════════════════════════════════

  Portfolio Summary
──────────────────────────────────────────────────────────────────────
  Solver status          : optimal
  Total budget           :  $10,000,000
  Total catalytic capital:   $3,200,000  (32.0%)
  Total commercial capital:  $6,800,000  (68.0%)
  Portfolio leverage ratio:        2.13x  (commercial per catalytic $)
  Portfolio CVaR (95%)   :       18.4%  of deployed capital

  Per-Vehicle Breakdown
──────────────────────────────────────────────────────────────────────
                            Allocation $  Alpha  Catalytic $  Commercial $  Leverage
  East Africa Nature Fund    4,000,000   28.5%    1,140,000     2,860,000     2.5x
  West Africa Clean Energy   6,000,000   34.3%    2,058,000     3,942,000     1.9x
```

**Key metrics:**

- **Alpha (cat %)** — the calibrated catalytic fraction: minimum share of vehicle capital that must be subordinated (first-loss + grant reserve) to meet the investor hurdle IRR and loss probability constraints.
- **Leverage** — commercial capital mobilized per dollar of catalytic capital. Higher is better.
- **CVaR (95%)** — expected portfolio loss rate in the worst 5% of scenarios. Must be below `cvar_max`.
- **Solver status** — `optimal` means the LP found a feasible allocation. `optimal_inaccurate` is acceptable. `infeasible` means the constraints cannot be jointly satisfied — relax `cvar_max` or `investor_hurdle_irr`.

---

## Loss Waterfall

Losses are absorbed in this order (most-junior first):

```
1. Grant Reserve       (donor-funded cash buffer, zero return)
2. First-Loss Tranche  (catalytic equity — accepts high loss risk)
3. Mezzanine Tranche   (subordinated commercial debt)
4. Guarantee           (wraps the senior tranche; e.g. DFC/MIGA-style)
5. Senior Tranche      (commercial lenders — protected by all layers above)
```

The guarantee **wraps the senior tranche specifically**, not the first-loss tranche. This matches real DFI structures (MIGA, DFC, GuarantCo protect senior noteholders, not subordinated catalytic capital).

---

## Modifying Constraints

To use different investor constraints, edit `calibrator_config` in your JSON or pass a `CalibratorConfig` object in Python:

```python
from calibration.vehicle.calibration import CalibratorConfig

cfg = CalibratorConfig(
    investor_hurdle_irr=0.10,   # require 10% median senior IRR
    max_loss_probability=0.03,  # allow at most 3% chance of senior loss
)
```

To change portfolio-level risk limits, adjust `PortfolioInputs`:

```python
from calibration.portfolio.models import PortfolioInputs

inputs = PortfolioInputs(
    ...,
    cvar_confidence=0.95,
    cvar_max=0.25,             # tighter: max 25% CVaR loss rate
    min_expected_return=0.05,  # require at least 5% expected return
)
```

---

## Using the API Directly

```python
from calibration.project.models import ProjectInputs
from calibration.project.simulation import ProjectSimulator
from calibration.vehicle.calibration import CalibratorConfig, CatalyticCalibrator
from calibration.vehicle.capital_stack import CapitalStack
from calibration.vehicle.risk_mitigants import Guarantee, GrantReserve, CoverageType

# 1. Simulate a single project
project = ProjectInputs(
    capex=1_000_000, opex_annual=50_000,
    price=40.0, yield_=50_000, lifetime_years=10
)
sim_result = ProjectSimulator(project).run(n_sims=2000, seed=42)
print(f"Loss probability: {sim_result.loss_probability:.1%}")

# 2. Build a capital stack and calibrate
stack = CapitalStack(
    total_capital=1_000_000,
    grant_reserve=GrantReserve(50_000),
    guarantee=Guarantee(0.30, CoverageType.PERCENTAGE),
    senior_coupon=0.08, mezzanine_coupon=0.12,
    mezzanine_fraction=0.10, lifetime_years=10,
)
calibrator = CatalyticCalibrator(
    capital_stack=stack,
    vehicle_cashflows=sim_result.cashflows,
    config=CalibratorConfig(investor_hurdle_irr=0.07, max_loss_probability=0.08),
)
alpha_star = calibrator.calibrate()
print(f"Minimum catalytic fraction: {alpha_star:.1%}")
print(f"Catalytic capital: ${alpha_star * 1_000_000:,.0f}")

# 3. Run the full portfolio pipeline
from calibration.portfolio.models import PortfolioInputs
from calibration.portfolio.optimizer import PortfolioOptimizer
from calibration.vehicle.models import VehicleInputs

vehicle = VehicleInputs(
    projects=[project],
    correlation_matrix=[[1.0]],
    total_capital=1_000_000,
    guarantee_coverage=0.30,
    grant_reserve=50_000,
)
portfolio_inputs = PortfolioInputs(
    vehicles=[vehicle],
    total_budget=1_000_000,
    n_sims=1000,
    seed=42,
)
result = PortfolioOptimizer(portfolio_inputs).run()
print(f"Leverage ratio: {result.leverage_ratio:.2f}x")
```

---

## Project Structure

```
calibration-tool/
├── pyproject.toml              # dependencies and build config
├── run_e2e.py                  # end-to-end runner (CLI)
├── examples/
│   ├── projects_vehicle_1.csv  # sample: East Africa projects
│   ├── projects_vehicle_2.csv  # sample: West Africa projects
│   └── portfolio.json          # sample: full JSON spec
├── calibration/
│   ├── project/                # Monte Carlo project simulation
│   ├── vehicle/                # Dual waterfall + catalytic calibration
│   ├── portfolio/              # LP portfolio optimizer
│   └── utils/                  # IRR computation, stats, Cholesky draws
└── tests/                      # 23 pytest tests
```

---

## Troubleshooting

**`ValueError: Constraints infeasible`**
The calibrator cannot find an `alpha` meeting both the hurdle IRR and max loss probability. Options:
- Lower `investor_hurdle_irr` (e.g. from 0.10 to 0.07)
- Raise `max_loss_probability` (e.g. from 0.03 to 0.08)
- Add a guarantee (`guarantee_coverage > 0`)
- Increase `grant_reserve`

**LP solver returns `infeasible`**
The portfolio allocation LP has no feasible solution. Options:
- Raise `cvar_max` (e.g. from 0.20 to 0.35)
- Lower `min_expected_return`
- Raise `max_allocation_fraction` (allow more concentration)

**`UserWarning: Base-case lifetime revenue is less than capex`**
The project has negative expected NPV at base-case assumptions. This is a warning, not an error — the Monte Carlo will still run. Review your `price`, `yield_`, and `lifetime_years` inputs.

**Slow runtime**
Default is 1000 simulations. For faster iteration during model development:
```bash
python run_e2e.py --sims 200
```
For production / final results, use 5000–10000:
```bash
python run_e2e.py --json examples/portfolio.json --sims 10000
```
