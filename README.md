# Calibration Tool

**Decision engine for allocating catalytic capital across blended-finance portfolios.**

It solves the core question in blended finance: *how much concessional capital is the minimum needed to make a vehicle commercially investable?* — and then optimises its allocation across a portfolio of vehicles to maximise leverage (commercial capital mobilised per catalytic dollar).

The system models the full hierarchy: **PROJECT → VEHICLE → PORTFOLIO**.
- **Calibration** solves the minimum catalytic fraction per vehicle using Brent's method.
- **Portfolio optimisation** allocates capital across vehicles via a CVaR-constrained LP.
- **Catalytic capital is never an input** — it is always a solved output.

---

## Key Concepts

If you're new to blended finance, here's the minimum you need to understand the tool's outputs:

| Term | Plain-English meaning |
|---|---|
| **Catalytic capital** | Concessional money (grants, first-loss equity, guarantees) that takes the most risk so private investors will join |
| **α (alpha)** | The share of a vehicle's total capital that must be catalytic. The tool solves for the *minimum* α — never higher than necessary |
| **Tranche** | A layer of a fund with a defined risk/return priority. Senior = safest, first-loss = riskiest |
| **Waterfall** | The rule for who gets paid first (cashflow waterfall) and who absorbs losses first (loss waterfall) |
| **IRR** | Internal Rate of Return — the annualised yield an investor earns |
| **CVaR (95%)** | Expected loss rate across the worst 5% of Monte Carlo scenarios — the portfolio's tail risk |
| **Leverage ratio** | Commercial capital mobilised per catalytic dollar: `(1−α)/α`. A 3x ratio means $3 private for every $1 concessional |
| **Vehicle** | A blended-finance fund pooling several projects; has its own tranche structure and calibrated α |

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"       # core + tests
pip install -e ".[ui]"        # adds Streamlit + Plotly for the web demo
```

### 2. Launch the Streamlit demo

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**. Sample data (3 vehicles, 9 projects) is pre-loaded — click **Run Calibration** to see results immediately.

### 3. Run with built-in sample data (CLI)

```bash
python run_e2e.py
```

### 4. Run the test suite

```bash
pytest
```

All tests should pass in under 10 seconds.

---

## Streamlit UI

The web demo has four pages accessible from the sidebar:

| Page | What it does |
|------|-------------|
| **Setup** | Choose sample data or upload your own CSVs; configure guarantee, reserve, mezzanine, coupon; run calibration |
| **Results** | KPI cards (catalytic capital, leverage, IRR, CVaR), vehicle breakdown table, IRR histogram, capital stack chart, alpha chart |
| **Sensitivity** | Four stress tests (A: guarantee ↑, B: price vol ↑, C: revenue ↓, D: price ↓); comparison table + grouped bar chart |
| **How It Works** | Conceptual explanation, key terms, CSV format spec, downloadable templates |

### Recommended CSV Format (Format 2)

The UI uses a single standard format: **year/yield/capex/opex with embedded price parameters**.

```csv
year,yield,capex,opex,revenue_type,base_price,price_growth_rate,price_vol
2025,0,4000000,100000,carbon,15.0,0.05,0.30
2026,0,3000000,100000,carbon,15.0,0.05,0.30
2027,30000,0,200000,carbon,15.0,0.05,0.30
2028,50000,0,200000,carbon,15.0,0.05,0.30
```

| Column | Required | Description |
|--------|----------|-------------|
| `year` | ✓ | Calendar year (e.g. 2025). Rows with `yield=0` are construction years. |
| `yield` | ✓ | Physical output / year (tCO₂e, tons, m³). `0` = construction year. |
| `capex` | ✓ | Capital expenditure (positive = outflow). `0` during operations. |
| `opex` | ✓ | Operating cost / year (positive = outflow). |
| `revenue_type` | optional | `"carbon"` or `"commodity"` — UI label only; same math for both. |
| `base_price` | ✓ | Current price per unit in USD. |
| `price_growth_rate` | ✓ | Annual log-price drift (e.g. `0.05` = 5 %/yr). |
| `price_vol` | ✓ | Annual price volatility / GBM σ (e.g. `0.30` = 30 %). |

**Revenue types:**
- **Carbon** (REDD+, ARR, biochar) — no liquid futures market; enter analyst assumptions for `base_price`, `price_growth_rate`, `price_vol`.
- **Commodity** (cocoa, timber, water) — use spot/futures data to derive `base_price`, drift, and vol.

**Typical parameters:**

| Project type | Yield units | Base price | Growth | Vol |
|-------------|------------|-----------|--------|-----|
| Forestry ARR | tCO₂e / yr | $15 | 5 % | 30 % |
| REDD+ | tCO₂e / yr | $12 | 6 % | 35 % |
| Biochar | tons / yr | $130–200 | 4 % | 28 % |
| Agroforestry (cocoa) | tons / yr | $1,800 | 3 % | 22 % |

### File upload naming convention

Files must be named **`vehiclename_projectname.csv`**. The prefix before the first `_` groups projects into the same vehicle:

```
forestry_arr.csv        → Forestry vehicle, project 1
forestry_conservative.csv → Forestry vehicle, project 2
agro_standard.csv       → Agro vehicle, project 1
```

### MVP constraints

| Constraint | Limit |
|-----------|-------|
| Max vehicles | 3 |
| Max projects / vehicle | 5 |
| Max rows per file (years) | 30 |
| File format | CSV only |
| Simulations | 100 – 2,000 |
| Historical price series upload | Post-MVP (use embedded params for now) |

### Sample data

Pre-loaded sample files live in `examples/ui_sample/`:

```
examples/ui_sample/
  vehicle_1_forestry/      — 3 ARR/forestry carbon projects
  vehicle_2_agroforestry/  — 3 cocoa agroforestry projects (commodity)
  vehicle_3_mixed/         — REDD+, biochar, hybrid (carbon + commodity)
```

Download templates from the **Setup** or **How It Works** page within the app.

---

## Loading Your Own Data

### Option A — CSV files

Two CSV sub-formats are supported and **auto-detected** by column names:
- If the file has a `cashflow` column → **A1 (cashflow-based)**
- If the file has `capex`, `price`, `yield_` columns → **A2 (parametric)**

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

The simulator applies a **GBM price path** to the positive cashflow periods
(`price_index[s,t] = exp(cumsum((μ−σ²/2) + σ·ε_t))`). Set `price_vol` and optionally
`price_drift` via JSON for tighter/looser volatility. This mode is ideal when you have a
financial model that produces an explicit year-by-year cashflow forecast.

**Multi-year capex:** include negative cashflows at t=1, t=2, … for construction draw-downs:
```csv
year,cashflow
0,-1000000
1,-500000
2,0
3,180000
```
Negative periods are passed through unchanged (GBM multiplier is not applied to outflows).

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
| `price_vol` | 0.15 | Annual price volatility (GBM σ) |
| `price_drift` | 0.0 | Annual log-price drift (GBM μ). If omitted and `price_series` not provided, drift = 0 |
| `yield_vol` | 0.10 | Annual yield volatility |
| `inflation_rate` | 0.03 | Annual opex inflation |
| `fx_vol` | 0.05 | Annual FX volatility on revenue |
| `delay_prob` | 0.05 | Per-period probability of production delay |
| `revenue` | — | Year-by-year revenue array (t=1..T). If present, also provide `cost`; populates `base_revenue`/`base_costs` instead of `base_cashflows` |
| `cost` | — | Year-by-year operating cost array (t=1..T). Used with `revenue` column |
| `price_series` | — | Historical price time-series (list of floats). If provided, μ and σ for the GBM are estimated from log-returns of this series, overriding `price_drift` and `price_vol` |

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
| `min_deployment` | float | Minimum total capital deployed `∑w_v ≥ min_deployment` (default 0.0; set > 0 to prevent degenerate all-zero LP solution) |
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
| `projects` | array | List of project objects (same fields as parametric CSV, plus `base_cashflows`, `base_revenue`, `base_costs`, `price_drift`) |

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
      "correlation_matrix": [[1.0, 0.35], [0.35, 1.0]],
      "projects": [
        {"capex": 2000000, "opex_annual": 80000, "price": 45.0, "yield_": 60000,
         "lifetime_years": 15, "price_vol": 0.12, "yield_vol": 0.08,
         "inflation_rate": 0.04, "fx_vol": 0.06, "delay_prob": 0.03},
        {"capex": 800000, "opex_annual": 35000, "price": 12.0, "yield_": 80000,
         "lifetime_years": 10, "price_vol": 0.18, "yield_vol": 0.15,
         "inflation_rate": 0.05, "fx_vol": 0.08, "delay_prob": 0.07}
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

### Option C — Folder-per-vehicle with Year/Yield/Capex/Opex files

This is the recommended format when you have project financial models with explicit
year-by-year construction and operating cashflows. Each vehicle is a sub-folder;
each `.csv` (or `.xlsx`) file inside it is one project.

#### File mask

```
examples/folder/
├── <vehicle_name>/
│   ├── <project_name>.csv
│   ├── <project_name>.csv
│   └── <project_name>.csv
└── <vehicle_name>/
    └── ...
```

- The **folder name** becomes the vehicle display name.
- The **file name** (without extension) becomes the project display name.
- Glob pattern: `examples/folder/**/*.csv` (or `*.xlsx`).
- Files are loaded in alphabetical order within each vehicle folder.

#### File mask

```
<vehicle_folder>/
├── project_<name>.csv        ← project data  (prefix "project_" required)
└── price_<commodity>.csv     ← price series  (prefix "price_",  optional)
```

The `project_` prefix distinguishes project files from price files in the same folder.
One price file can be shared by multiple projects in the same vehicle.

#### Column format

| Column | Required | Description |
|---|---|---|
| `Year` | Yes | Calendar year (e.g. 2025). Used for ordering; construction vs operating rows are inferred from `Yield == 0`. |
| `Yield` | Yes | Physical output per year (tons, m³, kWh, etc.). Zero during construction years. |
| `Capex` | Yes | Capital expenditure per year ($). Zero during operating years. |
| `Opex` | Yes | Operating cost per year ($). May be non-zero during construction. |
| `Project_Lifetime` | Recommended | Total rows in file (validation only). Warning if mismatch. |

**Price columns — Option 1 (embedded, all prices in USD):**

| Column | Description |
|---|---|
| `Base_Price` | Current price per unit (e.g. `20.0` for $20/ton carbon). Revenue = `Yield × Base_Price × GBM_path`. |
| `Price_Growth_Rate` | Annual log-price drift μ (e.g. `0.03` = 3%). |
| `Price_Vol` | Annual price volatility σ (e.g. `0.20` = 20%). |

**Price columns — Option 2 (external price series):**

| Column | Description |
|---|---|
| `Price_File` | Name of price series CSV in the same folder, without extension and without `price_` prefix (e.g. `carbon_credits` → loads `price_carbon_credits.csv`). `base_price`, μ and σ are derived from that file. |

**Price series format** (`price_<commodity>.csv`):

| Column | Description |
|---|---|
| `Date` | Observation date (any parseable format: `M/D/YYYY`, `YYYY-MM-DD`, etc.). |
| `Price` | Observed price per unit. |

The loader auto-detects frequency (monthly ≤ 35 days avg → ×12 for drift, ×√12 for vol; quarterly ≤ 100 days; annual otherwise). `base_price` = last observation.

> **GBM is applied to revenue only** (not net cashflow). Construction rows (Yield == 0) are passed through as deterministic outflows. This is the financially correct treatment: Opex is cost certainty, price uncertainty affects only the revenue line.

#### Worked example — Option 1 (embedded price params)

```csv
Year,Yield,Capex,Opex,Project_Lifetime,Base_Price,Price_Growth_Rate,Price_Vol
2025,0,500000,100000,21,20.0,0.03,0.20
2026,0,300000,100000,21,20.0,0.03,0.20
2027,0,100000,100000,21,20.0,0.03,0.20
2028,0,0,100000,21,20.0,0.03,0.20
2029,25000,0,100000,21,20.0,0.03,0.20
...
2045,25000,0,100000,21,20.0,0.03,0.20
```

`Yield = 25,000 tons/yr × $20/ton = $500k/yr revenue`. Construction CFs = `−600k, −400k, −200k, −100k` (deterministic). Operating revenue shocked by GBM.

#### Worked example — Option 2 (external price series)

```csv
Year,Yield,Capex,Opex,Project_Lifetime,Price_File
2025,0,500000,100000,21,carbon_credits
...
2045,250000,0,100000,21,carbon_credits
```

The loader reads `price_carbon_credits.csv` from the same folder and derives `base_price` (last price), `annual_drift`, `annual_vol` from its log-returns.

#### Built-in sample data (3 vehicles, 3 projects each)

```
examples/folder/
├── v1_east_africa_solar/                  ← Option 2: external price series
│   ├── project_solar_kenya.csv            # 4yr constr, 17yr ops, 250k units/yr
│   ├── project_solar_tanzania.csv         # 3yr constr, 15yr ops, 191k units/yr
│   ├── project_solar_ethiopia.csv         # 2yr constr, 12yr ops, 125k units/yr
│   └── price_carbon_credits.csv           # 60 monthly obs, base=$1.99, σ=16.9%/yr
├── v2_west_africa_agri/                   ← Option 1: Base_Price=$20, Growth=3%, Vol=20%
│   ├── project_agroforestry_ghana.csv     # 3yr constr + ramp-up, 13yr full ops
│   ├── project_cashcrop_nigeria.csv       # 2yr constr, 13yr ops, 17.5k tons/yr
│   └── project_forestry_senegal.csv       # 4yr constr + ramp-up, 22yr full ops
└── v3_sea_water/                          ← Option 1: Base_Price=$0.10/m³, Growth=2%, Vol=12%
    ├── project_water_indonesia.csv        # 3yr constr, 16yr ops, 5.5M m³/yr
    ├── project_water_vietnam.csv          # 2yr constr, 14yr ops, 4.0M m³/yr
    └── project_water_philippines.csv      # 3yr constr, 14yr ops, 3.5M m³/yr
```

Run:
```bash
python run_e2e.py --folder examples/folder/ --sims 1000 --seed 42
```

#### Key validation points

Before running, verify each project file passes these checks:

| # | Check | Why it matters |
|---|---|---|
| 1 | **Year column is present and strictly increasing** (no gaps, no duplicates) | The loader sorts by Year; gaps or duplicates cause incorrect period indices |
| 2 | **At least one row with Capex > 0** | No investment = no project; loader may infer `capex=0` silently |
| 3 | **Net CF at t=0 is negative** (`Yield − Opex − Capex < 0` in the first year) | IRR is undefined without an initial outflow; sentinel `-1.0` will be returned for all paths |
| 4 | **At least one operating year with `Yield − Opex > 0`** | If operating CF is never positive, IRR calibration will always be infeasible |
| 5 | **Capex and Opex are non-negative** | These are costs; negative values indicate a data entry error |
| 6 | **Yield = 0 during all construction years** | Non-zero yield during construction mixes capex and revenue, distorting cashflow timing |
| 7 | **Capex = 0 during all operating years** | Construction capex in operating years inflates the negative cashflow and underestimates IRR |
| 8 | **Lifetime ≤ 50 years** | `ProjectInputs` enforces a 50-year maximum; files exceeding this will raise a validation error |
| 9 | **Total operating cashflow > total construction cashflow (absolute)** | `∑(Yield−Opex) > ∑(Capex+Opex_construction)` ensures the project is NPV-positive at 0% discount; failing this means no commercially viable α exists |
| 10 | **No blank / NaN cells in numeric columns** | Blanks are filled with 0; a blank Capex row in a construction year silently drops that outflow |
| 11 | **Vehicle folder contains ≥ 1 project file** | An empty vehicle folder raises an error at load time |
| 12 | **`price_vol` is set** (default 0.15 if omitted) | Controls the width of Monte Carlo spread; very low values (< 0.05) may compress IRR variance and make calibration trivially easy or trivially hard |

**Quick sanity check** (run before `run_e2e.py`):
```python
from calibration.utils.loaders import load_project_from_excel
import numpy as np

p = load_project_from_excel("examples/folder/v1_east_africa_solar/solar_kenya.csv")
cf = np.array(p.base_cashflows)
print("t=0 CF (should be negative):", cf[0])
print("Operating CFs (should be positive):", cf[1:])
print("Simple NPV at 0% discount:", cf.sum())
assert cf[0] < 0, "ERROR: no initial outflow"
assert cf.sum() > 0, "ERROR: project has negative total cashflow"
```

---

## Runner Options

```
usage: run_e2e.py [-h] [--csv DIR | --json FILE | --folder DIR] [--sims SIMS] [--seed SEED]

optional arguments:
  --csv DIR      Directory with projects_vehicle_N.csv files (parametric/cashflow CSV)
  --folder DIR   Folder-per-vehicle mode: each sub-folder is one vehicle; each .xlsx/.csv
                 file inside is one project (loaded via load_project_from_excel)
  --json FILE    JSON file with full portfolio specification
  --sims SIMS    Number of Monte Carlo simulations (default: 1000)
  --seed SEED    Random seed for reproducibility
```

**`--folder` layout:**
```
examples/
├── vehicle_1/
│   ├── solar_farm.xlsx
│   └── agroforestry.xlsx
└── vehicle_2/
    └── clean_energy.xlsx
```
Each Excel/CSV file must have a `cashflow` column (or `Year`/`Yield`/`Capex`/`Opex` columns for the new format). Vehicle-level settings (guarantee, correlation, etc.) default to conservative values:

| Setting | Default |
|---|---|
| `guarantee_coverage` | 25% of senior notional |
| `grant_reserve` | 5% of estimated total capital |
| `mezzanine_fraction` | 10% |
| `senior_coupon` | 8% |
| `mezzanine_coupon` | 12% |
| `correlation_matrix` | 0.30 off-diagonal (moderate positive correlation) |
| `total_capital` | 1.2× sum of construction capex across all projects |

For full control over these settings, use the `--json` format instead.

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
                            Allocation $  Alpha  Catalytic $  Commercial $  Leverage  Marg.Eff
  East Africa Nature Fund    4,000,000   28.5%    1,140,000     2,860,000     2.5x      2.51x
  West Africa Clean Energy   6,000,000   34.3%    2,058,000     3,942,000     1.9x      1.91x
```

**Key metrics:**

- **Alpha (cat %)** — the calibrated catalytic fraction: minimum share of vehicle capital that must be subordinated (first-loss + grant reserve) to meet the investor hurdle IRR and loss probability constraints.
- **Leverage** — commercial capital mobilised per dollar of catalytic capital (`(1−α)/α`). Higher is better.
- **Marg.Eff** — marginal catalytic efficiency: `(1−α*)/α*`, the commercial capital unlocked per additional catalytic dollar at the minimum binding point. Equals leverage at calibrated α*.
- **CVaR (95%)** — expected portfolio loss rate in the worst 5% of scenarios. Must be below `cvar_max`.
- **Solver status** — `optimal` means the LP found a feasible allocation maximising total commercial capital. `optimal_inaccurate` is acceptable. `infeasible` means the constraints cannot be jointly satisfied — relax `cvar_max` or `investor_hurdle_irr`.

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
print(f"Solver status:    {result.status}")
print(f"Leverage ratio:   {result.leverage_ratio:.2f}x")
print(f"CVaR (95%):       {result.cvar_95:.1%}")
# Per-vehicle breakdown (keyed by vehicle index)
for v_idx, alloc in result.allocations.items():
    alpha = result.catalytic_fractions[v_idx]
    marg  = result.marginal_catalytic_efficiency[v_idx]
    print(f"  Vehicle {v_idx}: ${alloc:,.0f} total, α={alpha:.1%}, marg.eff={marg:.2f}x")
# Full distributions for further analysis
# result.portfolio_irr_distribution  → np.ndarray shape (n_sims,)
# result.portfolio_loss_distribution → np.ndarray shape (n_sims,)
```

---

## Project Structure

```
calibration-tool/
├── pyproject.toml              # dependencies and build config
├── run_e2e.py                  # end-to-end runner (CLI)
├── examples/
│   ├── projects_vehicle_1.csv  # sample: parametric CSV (Option A2)
│   ├── projects_vehicle_2.csv  # sample: parametric CSV (Option A2)
│   ├── cashflow_vehicle.csv    # sample: cashflow CSV (Option A1)
│   ├── portfolio.json          # sample: full JSON spec (Option B)
│   └── folder/                 # sample: folder-per-vehicle (Option C)
│       ├── v1_east_africa_solar/   (3 projects)
│       ├── v2_west_africa_agri/    (3 projects)
│       └── v3_sea_water/           (3 projects)
├── calibration/
│   ├── project/                # Monte Carlo project simulation
│   ├── vehicle/                # Dual waterfall + catalytic calibration
│   ├── portfolio/              # LP portfolio optimizer
│   └── utils/                  # IRR computation, stats, Cholesky draws
└── tests/                      # pytest suite (project / vehicle / portfolio layers)
```

---

## Validation

A structured 6-step validation protocol verifies that the system produces economically
correct, numerically stable, and decision-useful outputs.

### Mock dataset

A ready-to-use mock portfolio is included at `examples/mock_portfolio/` with 3 vehicles
× 3 projects representing nature-based investments:

```
examples/mock_portfolio/
  vehicle_1_forestry/
    forestry_arr.csv           ARR/timber, $8M capex, 3yr construction
    forestry_conservative.csv  Lower-yield ARR, $7M capex
    forestry_risky.csv         High capex ($9M), delayed income — tests negative-NPV flow-through
  vehicle_2_agroforestry/
    agro_standard.csv          J-curve (multi-year negative before positive)
    agro_conservative.csv      Lower-yield agroforestry
    agro_volatile.csv          Alternating cashflow pattern (high dispersion)
  vehicle_3_mixed/
    redd_credits.csv           REDD+, cash-generating from year 1
    biochar_volatile.csv       Biochar with alternating high/low cashflows
    hybrid_diversified.csv     Hybrid with 2yr construction then steady growth
```

All files use `year,cashflow` CSV format (years 0–15). The loader auto-detects this format.

### Run the mock portfolio through the main CLI

```bash
python run_e2e.py --folder examples/mock_portfolio/ --sims 1000 --seed 42
```

### Automated validation (pytest — for CI)

```bash
# Full e2e test suite (35 tests covering all 6 steps)
pytest tests/test_e2e.py -v

# Run a specific step
pytest tests/test_e2e.py -v -k Step2    # deterministic validation
pytest tests/test_e2e.py -v -k Step3    # Monte Carlo
pytest tests/test_e2e.py -v -k Step4    # sensitivity checks

# Full test suite (unit + integration + e2e)
pytest -v
```

### Interactive validation script (for analysts)

```bash
python validate_e2e.py                     # 1 000 sims, seed 42
python validate_e2e.py --sims 500          # faster run
python validate_e2e.py --sims 2000 --seed 7
python validate_e2e.py --charts            # also produce matplotlib charts
```

The script prints a structured report with pass/fail for 18 checks:

| Step | What is checked |
|------|----------------|
| 1 Data loading | 3 vehicles × 3 projects loaded; CF lengths; capex sign |
| 2 Deterministic | Capex determinism; multi-year construction; negative-NPV flow-through; waterfall ordering |
| 3 Monte Carlo | Solver status; no overflow warnings; IRR percentiles; leverage; loss bounds |
| 4 Sensitivity | Higher guarantee → alpha ↓; higher vol → dispersion ↑; lower CFs → alpha ↑ |
| 5 Diagnostics | NaN ratio bounds; loss statistics; IrrDiagnostics API |
| 6 Charts | IRR histogram; loss distribution (optional, requires matplotlib) |

### Sensitivity tests: expected directions

| Change | Expected direction | Economic rationale |
|--------|-------------------|-------------------|
| Guarantee coverage 25% → 50% | Alpha decreases | DFI covers more senior losses; less first-loss needed |
| Price volatility 0.15 → 0.40 | IRR dispersion increases | Wider revenue swings → wider return distribution |
| Operating cashflows × 0.80 | Alpha increases | Weaker economics → more concessional capital required |

### Diagnostic thresholds

| Metric | Flag if... | Likely cause |
|--------|-----------|-------------|
| NaN IRR ratio | > 40% | Cashflow assumptions too aggressive or project too short |
| NaN IRR ratio | = 0% | Possible over-filtering; verify edge cases are reaching solver |
| All loss rates = 0% | Always | Capital stack may be over-sized; consider tighter guarantee |

---

## Developer Tools — AI Code Review Plugin

Uses the OpenAI API (GPT-4o or GPT-4 Turbo) to review the calibration tool's
core financial source files for mathematical errors, numerical stability issues,
edge cases, and logic errors — a second model's perspective on the implementation.

### Setup

```bash
# From the repo root (editable install — recommended):
pip install -e ".[codex]"

# Or install openai directly:
pip install openai

export OPENAI_API_KEY=sk-...
```

> **Note:** Code Review is a developer/internal tool. The nav item is hidden
> automatically when `openai` is not installed, so it will not appear in
> public demos deployed without the optional dependency.

### CLI usage

```bash
# Full review, text report (default):
python run_codex_review.py

# Specific files only:
python run_codex_review.py --files calibration/vehicle/calibration.py calibration/utils/irr.py

# Choose model:
python run_codex_review.py --model gpt-4-turbo

# Machine-readable output:
python run_codex_review.py --output-format json
python run_codex_review.py --output-format markdown > review.md
```

Exit codes: `0` = completed (findings are normal), `1` = missing key/package, `2` = all API calls failed.

### Python API

```python
from calibration.plugins.openai_codex import CodexReviewer

result = CodexReviewer(model="gpt-4o").review()

for f in result.findings:
    print(f"{f.severity:10s}  {f.file}  [{f.category}]  {f.description}")

print(f"\n{len(result.findings)} findings across {len(result.files_reviewed)} files")
print(f"Tokens used: {result.total_tokens_used:,}")
```

### Streamlit

Visit the **🤖 Code Review** page (5th item in the sidebar navigation).
Enter your OpenAI API key in the form (or pre-set `OPENAI_API_KEY` in the environment
or in the Streamlit Cloud secrets dashboard — see [Deployment](#streamlit-cloud-deployment)).

### What it reviews

The plugin sends each of these files to the API in a separate call:

| File | Focus areas |
|------|-------------|
| `calibration/project/simulation.py` | GBM drift/variance correction, shock application |
| `calibration/vehicle/calibration.py` | Brent's method bracketing, monotonicity probe |
| `calibration/vehicle/capital_stack.py` | Waterfall arithmetic, loss allocation ordering |
| `calibration/vehicle/risk_mitigants.py` | Guarantee absorption logic, reserve depletion |
| `calibration/portfolio/optimizer.py` | Rockafellar-Uryasev CVaR LP formulation |
| `calibration/utils/irr.py` | IRR bracket logic, sentinel handling |
| `calibration/utils/stats.py` | Cholesky decomposition, CVaR formula |

---

## Authentication & Security

The Streamlit app ships with an optional password gate (`auth.py`). It has
three modes depending on how `.streamlit/secrets.toml` is configured:

| `secrets.toml` | Mode | Use case |
|---|---|---|
| No `[auth]` section | **Open access** (dev) | Local development, private networks |
| `[auth]` with empty hash | **Fail-closed** | Misconfiguration guard |
| `[auth]` with valid hash | **Password gate** | Private demos, Streamlit Cloud |

### Generate a password hash

```bash
python auth.py                 # interactive — uses getpass.getpass (no shell history)
python auth.py YourPassword    # non-interactive (automation/CI)
```

Output is a salted PBKDF2-SHA256 string:

```
pbkdf2_sha256$390000$3c80...e2f1$04aa...858c
```

### Configure `secrets.toml`

Create `.streamlit/secrets.toml` in the repo root (or paste into the
Streamlit Cloud → Settings → Secrets dashboard):

```toml
[auth]
password_hash = "pbkdf2_sha256$390000$3c80...e2f1$04aa...858c"
```

Restart the app; the login form appears at startup. File permissions on a
self-hosted deployment should be `chmod 600 .streamlit/secrets.toml`.

### Hash format

| Format | Purpose | Notes |
|---|---|---|
| `pbkdf2_sha256$<iters>$<salt_hex>$<dk_hex>` | **Recommended.** Salted PBKDF2-SHA256 (390,000 iters). | Produced by `python auth.py`. |
| 64-char hex digest | **Legacy** unsalted SHA-256. | Still verifies; rotate to PBKDF2. |

### Rate limiting

After 5 failed login attempts the session is locked for 60 seconds with a
visible countdown; the form becomes usable again automatically (no full
page refresh required). On successful login or explicit logout, the
attempt counter and lockout timestamp are cleared.

> **Known limitation — session-scoped lockout.** The 5-attempt/60-s window
> is tracked in Streamlit `session_state`, so an attacker can bypass it by
> opening a fresh incognito session. The lockout is a **UX rate limiter,
> not a hard security control**. For real brute-force defense, front the
> app with Cloudflare Turnstile, Streamlit Community Cloud's Google SSO
> (Settings → Sharing → *Only specific people*), or an IP-aware reverse
> proxy.

### Running the auth tests

```bash
pytest tests/test_auth.py -v
```

### Manual test plan

Before promoting a new deployment to users, walk through the checklist in
[`docs/AUTH_MANUAL_TEST_PLAN.md`](docs/AUTH_MANUAL_TEST_PLAN.md). It covers
open-access mode, fail-closed mode, successful/failed logins, lockout and
expiry, logout cleanup, PBKDF2↔legacy verification, and the session-scope
bypass acknowledgement.

---

## Streamlit Cloud Deployment

To deploy for non-technical users (URL-only, no Python install required):

**1. Ensure `requirements.txt` is present** (already in the repo):
```
pip install -r requirements.txt   # verified list of all UI dependencies
```

**2. Push to GitHub** and connect the repo at [share.streamlit.io](https://share.streamlit.io):
- Repository: `lwei-seedling/calibration-tool`
- Branch: `main`
- Main file path: `app.py`

**3. Configure secrets** in the Streamlit Cloud dashboard (Settings → Secrets):

```toml
[auth]
password_hash = "pbkdf2_sha256$390000$...$..."   # generate via python auth.py

# Optional — enables the Code Review page
OPENAI_API_KEY = "sk-..."
```

If `OPENAI_API_KEY` is set, also add `openai>=1.0` to `requirements.txt` before deploying.
Omit the `[auth]` section entirely for an open demo (not recommended for sensitive data).

**4. Share the deployed URL.** Users can upload CSVs, run calibration, explore results,
and download reports — no Python installation needed.

---

## Working with Claude Code

The repo includes a small `.claude/` config that activates automatically when you
open the project in Claude Code (the CLI, desktop app, or web). It does two things:

### 1. `/calibrate-smoke` — one-keystroke sanity check

In a Claude Code session, type `/calibrate-smoke`. It runs:

```bash
python run_e2e.py --sims 200 --seed 42   # fast deterministic e2e (~10 s)
python -m pytest -x -q                    # full unit tests, fail-fast
```

Claude then summarises calibrated α, leverage, and any failures. Use it as
a pre-commit check after touching `calibration/**`.

Edit `.claude/commands/calibrate-smoke.md` to change what it runs.

### 2. Auto-hooks

Two automated guardrails fire during a session:

- **Secrets guard** — blocks any `git add` / `git commit` that references
  `.streamlit/secrets.toml` (your password hash file). Defence-in-depth on
  top of `.gitignore`.
- **Layer-local test runner** — when Claude edits a file under
  `calibration/vehicle/`, `calibration/portfolio/`, `calibration/project/`,
  or `auth.py`, the matching `tests/test_<layer>.py` runs automatically.
  Failures are surfaced back to Claude so it can self-correct.

Both require `pip install -e ".[dev]"` (so `pytest` is on `PATH`).

To disable temporarily: rename `.claude/settings.json`, or pass `--no-hooks`
when launching Claude Code. Inspect with `/hooks` inside a session.

### 3. Optional: subagents and MCP

The project does **not** ship custom subagents or MCP servers. Add them only
when a manual workflow has bitten you twice:

| Primitive | Add when… |
|---|---|
| Skill / slash command | A multi-step recipe is repeated >3×/week |
| Subagent (`Explore`, `Plan`, custom) | You need parallel investigation across layers, or `app.py` (1130 LOC) bloats your context |
| Hook | A guardrail must be deterministic — model cannot be trusted to remember |
| MCP server | Integrating a stateful external service (price-data feed, log tail) |

See `CLAUDE.md` "Claude Code Integration" for the architectural rationale.

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
- Lower `min_deployment` if it was set too high relative to `total_budget`

**LP solver returns all-zero weights**
No capital is deployed. Set `min_deployment > 0` in `PortfolioInputs` (or `"min_deployment"` in JSON) to enforce a minimum total deployment floor.

**`RuntimeWarning: overflow encountered in power / multiply`**
Harmless. Newton's method for IRR evaluation hits overflow on high-return outlier paths. The sentinel cap (`10.0` = 1000% IRR) is applied by `clean_irr()` afterwards. Results are unaffected.

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
