# CLAUDE.md — Developer Guide for Claude Code

## Project Overview

**Calibration Tool** — a Python system that solves for the minimum catalytic capital
(first-loss, grants, guarantees) required to make a blended-finance vehicle investable
for commercial capital. The hierarchy is:

```
PROJECT → VEHICLE → PORTFOLIO
```

| Entity | Concrete meaning | Code entry point |
|---|---|---|
| **Project** | A single real-world investment (e.g. a solar farm). Has revenue, costs, and risk parameters. | `ProjectInputs`, `ProjectSimulator` |
| **Vehicle** | A blended-finance fund that pools several projects and splits returns across investor tranches (senior, mezzanine, first-loss). | `VehicleInputs`, `CatalyticCalibrator` |
| **Portfolio** | A foundation's allocation across multiple vehicles. The LP decides how much to invest in each. | `PortfolioInputs`, `PortfolioOptimizer` |

**α (alpha)** — the catalytic fraction: what share of a vehicle's total capital must be
concessional (first-loss + grants) so that commercial senior lenders meet their return
and risk targets. **Catalytic capital is never an input** — it is solved by calibration.

### Finance term glossary

| Term | Meaning |
|---|---|
| **Blended finance** | Structures that use concessional (donor/DFI) capital to de-risk returns for private investors |
| **DFI** | Development Finance Institution (e.g. DFC, MIGA, GuarantCo) — providers of guarantees and subordinated capital |
| **First-loss tranche** | Capital that absorbs losses first; accepts high risk in exchange for unlocking senior lending |
| **Guarantee** | A DFI commitment to cover a fraction of senior losses if they occur |
| **IRR** | Internal Rate of Return — the discount rate that makes NPV of cashflows = 0; the investor's yield |
| **CVaR** | Conditional Value at Risk — the expected loss rate in the worst X% of scenarios (tail risk measure) |
| **GBM** | Geometric Brownian Motion — a log-normal random walk used to model price evolution over time |
| **Waterfall** | The contractual priority order in which investors receive cashflows or absorb losses |

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
├── utils/
│   ├── irr.py         # batch_irr(), clean_irr() with edge-case sentinels
│   ├── stats.py       # var(), cvar(), cholesky_correlated_draws()
│   └── loaders.py     # load_project_from_excel(), load_price_series()
└── plugins/           # Optional integrations (not required for core functionality)
    └── openai_codex.py  # CodexReviewer: GPT-4 code review (requires openai extra)

app.py                 # Streamlit web demo (5 pages: Setup, Results, Sensitivity, How It Works, Code Review)
auth.py                # Password gate for app.py: PBKDF2-SHA256 hashes, session lockout
examples/
└── ui_sample/         # Format 2 CSV sample data for the Streamlit demo
    ├── vehicle_1_forestry/     (project_forestry_arr, _conservative, _risky)
    ├── vehicle_2_agroforestry/ (project_agro_standard, _conservative, _volatile)
    └── vehicle_3_mixed/        (project_mixed_redd, _biochar, _hybrid)
```

---

## Data Flow

```
ProjectInputs (risk params)
        │
        ▼
ProjectSimulator.run()        ← Monte Carlo: draws GBM/lognormal shocks per period
        │  cashflows[n_sims, T]
        ▼
CatalyticCalibrator.calibrate()   ← binary-searches α so senior IRR ≥ hurdle
        │  alpha* (scalar)             and loss_prob ≤ threshold
        │  VehicleResult
        ▼
PortfolioOptimizer.run()      ← LP: maximise commercial capital subject to
        │  PortfolioResult         catalytic budget and CVaR constraints
        ▼
      Output (allocations, leverage, CVaR)
```

Each layer is independent — `PortfolioOptimizer` only sees `VehicleResult` objects and
does not know how cashflows were generated. `ProjectSimulator` does not know about
tranches or waterfalls.

---

## Setup

```bash
pip install -e ".[dev]"       # core + tests
pip install -e ".[ui]"        # adds streamlit + plotly for the web demo
```

Dependencies: `numpy`, `scipy`, `numpy-financial`, `cvxpy`, `pydantic`, `pandas`, `pytest`.
UI extras: `streamlit>=1.32`, `plotly>=5.18`.

## Streamlit UI

```bash
streamlit run app.py          # opens at http://localhost:8501
```

The UI has four pages:

| Page | Purpose |
|------|---------|
| **Setup** | Load sample portfolio or upload your own CSVs; configure vehicle parameters; run calibration |
| **Results** | KPI cards (catalytic capital, leverage, IRR, CVaR), vehicle breakdown table, 3 charts |
| **Sensitivity** | Four stress tests (A: guarantee ↑, B: price vol ↑, C/D: revenue/price ↓); comparison table + chart |
| **How It Works** | Conceptual explanation, glossary, CSV format spec, template downloads |

### Format 2 CSV (recommended for the UI)

```csv
year,yield,capex,opex,revenue_type,base_price,price_growth_rate,price_vol
2025,0,4000000,100000,carbon,15.0,0.05,0.30   ← construction year (yield=0)
2026,0,3000000,100000,carbon,15.0,0.05,0.30   ← construction year
2027,30000,0,200000,carbon,15.0,0.05,0.30     ← operating year (yield>0)
```

- Construction rows (`yield=0`) → `base_cashflows` (capex + opex as negative CFs)
- Operating rows (`yield>0`) → `base_revenue = yield × base_price`, `base_costs = opex`
- GBM price shocks applied to revenue only; capex/opex are deterministic pass-throughs
- `revenue_type` ("carbon" or "commodity") is a UI label only — same math for both

### MVP constraints

| Limit | Value |
|-------|-------|
| Max vehicles | 3 |
| Max projects / vehicle | 5 |
| Max horizon | 30 years |
| File format | CSV only |
| Simulations | 100–2000 |

### File upload naming

Uploaded files must follow `vehiclename_projectname.csv`. The prefix before the first `_`
determines which vehicle the project belongs to:
- `forestry_arr.csv` + `forestry_redd.csv` → **Forestry** vehicle (2 projects)
- `agro_standard.csv` → **Agro** vehicle (1 project)

---

## Authentication (`auth.py`)

`auth.py` is a **single-password gate** in front of `app.py`. It is called after
`st.set_page_config()` and before any other UI code:

```python
from auth import check_auth, logout
if not check_auth():
    st.stop()
```

### Configuration modes (three states of `secrets.toml`)

| `secrets.toml` state | Behaviour |
|---|---|
| No `[auth]` section | **Open access** (dev mode). `check_auth()` returns True immediately. |
| `[auth]` present but `password_hash` empty | **Fail-closed.** Shows "Auth is enabled but no password is configured." |
| `[auth]` with valid `password_hash` | **Password gate.** Renders login form. |

### Hash formats

| Format | Shape | When to use |
|---|---|---|
| **PBKDF2-SHA256** (recommended) | `pbkdf2_sha256$<iters>$<salt_hex>$<dk_hex>` | All new deployments. Default: 390,000 iterations, 16-byte salt. |
| **Legacy SHA-256** | 64-char hex digest | Kept for backward compatibility only; rotate to PBKDF2. |

Generate a PBKDF2 hash for `secrets.toml`:

```bash
python auth.py                 # interactive (getpass.getpass; no shell history leak)
python auth.py YourPassword    # non-interactive (automation/CI)
```

### Session-state keys used

| Key | Lifetime | Meaning |
|---|---|---|
| `_authenticated` | Until logout or session end | True if user passed the gate this session |
| `_login_attempts` | Until lockout expires + reset | Count of consecutive failed password submissions |
| `_login_locked_until` | Until 60 s after 5th failure | Unix timestamp when the lockout lifts |

Both `logout()` and successful login call `_reset_auth_state()` which pops all three keys.

### Known limitation — session-scoped lockout

`_MAX_ATTEMPTS = 5` and `_LOCKOUT_SECONDS = 60` are enforced via Streamlit
`session_state`. A fresh browser session (new incognito tab, cleared cookies)
starts with an empty state and bypasses the lockout. This is a **UX rate
limiter, not a hard security control**. For defense against scripted brute
force, put an IP-aware proxy or Cloudflare Turnstile in front of the app.

### Developer notes

- `_verify_password` accepts both hash formats, ignores leading/trailing
  whitespace on `expected_hash` (copy-paste-friendly) and is
  case-insensitive for legacy hex digests.
- Empty passwords are rejected unconditionally (even if the stored hash
  matches `sha256(b"")` — that's a misconfiguration, not a valid login).
- PBKDF2 iteration counts are capped at `_PBKDF2_MAX_ITERATIONS = 10_000_000`
  as DoS defense-in-depth.
- Tests live in `tests/test_auth.py` — pure-function tests only, no
  Streamlit AppTest dependency.

See `docs/AUTH_MANUAL_TEST_PLAN.md` for the ops checklist before shipping a
new deployment.

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
pytest tests/test_codex_plugin.py -v   # requires: pip install -e ".[codex]"
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

## Plugins (Optional)

### AI Code Review (`calibration/plugins/openai_codex.py`)

Requires `pip install -e ".[codex]"` (or `pip install openai`) and `OPENAI_API_KEY`.

```bash
python run_codex_review.py          # text report (default)
python run_codex_review.py --output-format markdown > review.md
python run_codex_review.py --model gpt-4-turbo
```

See `README.md` "Developer Tools" section for full CLI/API documentation.

In the Streamlit UI, the "🤖 Code Review" nav item is auto-hidden when `openai`
is not installed, so it does not appear in public demos.

---

**Folder mode file rules** (loaded by `_load_folder_inputs` in `run_e2e.py`):
- `project_*.csv` / `*.xlsx` → loaded as project data via `load_project_from_excel()`
- `price_*.csv` → price series; referenced by project files via `Price_File` column, **not** loaded directly
- Other `.csv` files → treated as legacy cashflow or parametric format

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

A "waterfall" is the contractual rule for how money flows (or losses are allocated) between
investor tranches. There are two separate waterfalls:

- **Loss waterfall** — runs once per scenario at maturity. Computes total vehicle loss
  `L[s] = max(0, -NPV(CF[s], discount_rate))` and allocates it junior-first.
  `discount_rate` defaults to 0.0 (undiscounted sum). Set `discount_rate > 0` on
  `VehicleInputs` to use time-value-of-money discounting in loss calculations.
- **Cashflow waterfall** — runs per-period during the vehicle's life. Distributes
  available cash to each tranche according to coupon entitlement before any residual
  goes to the first-loss (equity) layer.

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

**Goal:** find the smallest α such that, at α, senior lenders meet *both* their hurdle IRR
and their maximum loss-probability tolerance. The search is 1-D (α ∈ [0, 1]).

`CatalyticCalibrator.calibrate()`:
1. Probe `_h(α)` at 20 evenly-spaced α points to check monotonicity
2. If monotone → `scipy.optimize.brentq` — a bracketed root-finder that converges in
   10–15 function evaluations (xtol=1e-4). This is the fast path.
3. If not monotone (rare; can happen with small-sample noise) → two-phase grid search
   (50 coarse + 50 fine points around the best coarse point)
4. **Common-random-numbers**: the Monte Carlo paths are drawn once and reused for every
   α evaluation, so noise does not mask the monotone signal.

**Calibration objective:** `_h(alpha) = min(g1, g2)` where:
- `g1` = (senior median IRR) − (hurdle IRR): positive means IRR constraint is met
- `g2` = (max_loss_prob) − (observed loss_prob): positive means loss constraint is met

Calibration finds the α* where `min(g1, g2) = 0` (at least one constraint is binding).

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

### 6. Portfolio LP (Rockafellar-Uryasev CVaR)

**What it does:** given a fixed catalytic budget `B_cat`, find how much to invest in each
vehicle (`w_v`) to maximise total commercial capital, while keeping portfolio tail-loss
(CVaR) within a tolerance.

**CVaR** (Conditional Value-at-Risk) at confidence level β is the *expected* loss in the
worst `(1−β)` fraction of scenarios. The Rockafellar-Uryasev linearisation makes this
tractable as a standard LP by introducing auxiliary variables `u_s` (per-scenario excess
loss) and `zeta` (VaR threshold).

Variables: `w_v` (dollar allocations), `zeta` (VaR threshold), `u_s` (excess loss auxiliaries).
Objective: **maximise `∑ (1 − c_v) * w_v`** (commercial capital mobilised).
Budget constraint: `∑ c_v * w_v ≤ B_cat` (total catalytic budget).
Optional stability constraint: `∑ w_v ≥ min_deployment` (prevents degenerate all-zero solution).
Solver preference: CLARABEL → ECOS → SCS (cvxpy auto-selects from available solvers).

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

- **IRR overflow warnings** (`RuntimeWarning: overflow encountered in power`): expected and
  harmless. They occur when Newton's method evaluates a very high trial IRR (e.g. >500%)
  on outlier paths. The sentinel cap (`10.0`) is applied afterwards by `clean_irr()`.
  Suppress with `warnings.filterwarnings("ignore", "overflow")` if needed.

- **High alpha (>50%) from `--folder` mode**: the folder loader uses conservative vehicle
  defaults (25% guarantee, 5% reserve). If calibrated α is unexpectedly high, either (a) the
  projects are genuinely high-risk given the hurdle IRR, or (b) increase `guarantee_coverage`
  and `grant_reserve` via `--json` for full control. Leverage < 1× means concessional capital
  exceeds commercial — this is realistic for early-stage or high-vol projects.

- **`load_price_series` frequency detection**: if dates can't be parsed (e.g. non-standard
  format), the function defaults to annual frequency (no scaling). Pass pre-annualised
  log-returns by supplying annual price observations to avoid this.
