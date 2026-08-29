# Independent Build Review

> **Status: all findings resolved.** Fixes landed in the commit following this
> document. `scripts/review_repro.py` now *verifies* the fixed behaviour and exits
> non-zero on regression; `tests/test_review_properties.py` holds the same
> properties as unit tests. See "What the fixes changed" at the end for
> before/after numbers.

Reviewed at commit `ed9c1c0`. Suite state at review time: **146 passed** in 75s;
`run_e2e.py --sims 300 --seed 42` completes in 4.4s with solver status `optimal`.

Every finding below was confirmed by executing the code. Reproductions live in
`scripts/review_repro.py` — run `python scripts/review_repro.py` to regenerate all
quoted output.

**Summary:** 4 critical, 4 high, 9 medium. The architecture and layering are sound;
the defects are specific and sit directly under the headline α the tool exists to
produce.

## One defect explains most of the others

`_cashflow_waterfall` pays each tranche its current-year coupon and sweeps everything
left over to equity — every period, with no reserve and no amortisation. Senior
principal is due as a bullet at maturity, by which point only one year of cash remains
in the vehicle. Senior lenders are starved of principal no matter how profitable the
vehicle is, and the calibrator compensates by demanding far more catalytic capital
than the economics require.

---

## Critical

### F1 — Excess cash leaks to equity every period, so senior principal is never repaid
`calibration/vehicle/capital_stack.py:191` · **FIXED**

A vehicle returning 3× its capital ($30M inflows on $10M invested, deterministic)
still leaves senior $5.6M short of principal while equity takes $21.2M:

```
  t   vehicle CF    -> senior  -> first_loss
  0  -10,000,000   -8,000,000     -2,000,000
  1    3,000,000      640,000      2,360,000   <- coupon only; $2.36M leaves
 10    3,000,000    3,000,000              0   <- $8M principal due, $3M left

Senior received      $   8,760,000  on $8,000,000 of principal
Principal shortfall  $   5,640,000  (unpaid at maturity)
First-loss received  $  21,240,000
```

`CLAUDE.md` frames this as a user-side pitfall, but no real fund is structured this
way. The shortfall scales with tenor, so it biases α upward on every vehicle.

**Fix:** retain cash against the bullet — amortise senior principal, or add a reserve
account that traps cash once the coupon is met and releases residual to equity only
after the maturity obligation is funded. Confined to `_cashflow_waterfall`.

### F2 — The correlation matrix has no effect on anything
`calibration/portfolio/optimizer.py:96–105` · **FIXED**

The Iman-Conover reordering computes `source_ranks` on line 100 and never uses it.
The applied permutation derives only from the target draws, so it never aligns paths
to their own cashflow magnitudes. Induced correlation is exactly zero.

```
target rho=0.00 | as-built=-0.0005 | iman-conover=+0.0100
target rho=0.50 | as-built=-0.0057 | iman-conover=+0.5039
target rho=0.90 | as-built=+0.0041 | iman-conover=+0.8996
```

End to end, the UI's "Off-diagonal Correlation" slider moves nothing:

```
   rho |    alpha* |   catalytic $ |  leverage
  0.00 |  0.799938 |     5,759,552 |    0.2501
  0.80 |  0.799979 |     5,759,848 |    0.2500
```

`correlation_matrix` is required, shape-validated, documented as a key design
decision, and exposed as a slider. It is inert. Diversification is unmodelled, biasing
α *downward* — the opposite direction from F1, so they do not cancel predictably.

**Fix:** two lines.

```python
sorted_idx  = np.argsort(cfs.sum(axis=1))
reorder_idx = sorted_idx[target_ranks]
```

Verified to recover ρ = 0.504 and 0.900 against targets of 0.5 and 0.9.

### F3 — Any IRR above 1000% is recorded as a total loss
`calibration/utils/irr.py:89` · **FIXED**

`_irr_single` brackets in [−0.999, 10.0] and returns `NaN` when NPV has the same sign
at both ends. For a very profitable path NPV is still positive at r = 10, so it
returns NaN — and `clean_irr` maps NaN to **−1.0, the total-loss sentinel**.

```
true IRR (numpy_financial): 49.949999   (4995%)
_irr_single              : nan
clean_irr                : [-1.]        <- booked as total loss

NPV(r=-0.999) = 5.000e+36
NPV(r=10.0)   = 3.995e+05    <- still positive: no root in bracket
```

The documented `10.0` cap is unreachable: `clean_irr` only caps `+inf`, and `brentq`
never returns `+inf`. `tests/test_project.py::test_profitable_project` passes while
emitting "100.0% of IRR paths returned NaN".

**Fix:** in the bracket-miss branch return `_R_HI` when `npv_hi > 0` and `_R_LO` when
`npv_lo < 0`; reserve `NaN` for genuinely undefined paths.

### F4 — CVaR collapses when losses are mostly zero, which is the normal case
`calibration/utils/stats.py:12–18` · **FIXED**

`cvar()` selects its tail with `losses >= threshold`. A senior tranche loses in only a
few percent of scenarios, so VaR(95%) is exactly zero and the comparison admits every
zero-loss path into the "tail".

```
P(loss>0)             = 0.020
VaR(95%)              = 0
cvar() as implemented =     60,000   <- tail set = 1000 of 1000 paths
true mean of worst 5% =  1,200,000   <- understated by 20x
```

This feeds every `TrancheResult.cvar_95`, the portfolio CVaR headline, and the LP's
risk constraint — always in the reassuring direction. The e2e run's
"Portfolio CVaR (95%): 0.0%" is this bug, not a safe portfolio.

**Fix:** take a fixed count, not a threshold —
`k = max(1, ceil((1 - confidence) * n))`, then average the `k` largest losses.

---

## High

### F5 — Hurdle at or above the senior coupon silently returns a fake answer
`calibration/vehicle/calibration.py:148` · **FIXED**

Senior IRR is structurally capped at the coupon — the waterfall never pays senior more
than coupon plus principal. So when `investor_hurdle_irr >= senior_coupon`, the only
way to satisfy the IRR constraint is to shrink senior until it disappears. The
calibrator does that and reports it as a solution.

```
hurdle= 6%  ->  alpha*=0.5340  senior notional=$  3,659,846
hurdle= 8%  ->  alpha*=0.8999  senior notional=$      1,398  <- degenerate
hurdle=10%  ->  alpha*=0.9000  senior notional=$        150  <- degenerate
hurdle=12%  ->  alpha*=0.9000  senior notional=$        410  <- degenerate
```

α* pins to `1 − mezzanine_fraction`, the point where senior notional hits zero. No
warning is raised, and `ValueError: Constraints infeasible` is effectively unreachable
because h(α) turns non-negative there for structural rather than economic reasons. The
UI lets a user set a 15% hurdle against an 8% coupon with two independent sliders.

**Fix:** validate `investor_hurdle_irr < senior_coupon` at config time, and treat any α
whose senior notional falls below a floor (say 5% of total capital) as infeasible
rather than as a root. Both belong in `_h` so the grid fallback inherits them.

### F6 — The guarantee and grant reserve cannot affect senior IRR at all
`calibration/vehicle/capital_stack.py:229` · **FIXED**

The two waterfalls never meet. Loss absorption runs once at maturity on an NPV figure;
senior IRR comes from the cashflow waterfall, which the mitigants never enter.

```
 guarantee    reserve | median IRR  loss prob
      0.00          0 |     0.0006      0.000
      0.50          0 |     0.0006      0.000
      1.00          0 |     0.0006      0.000
      0.00  2,000,000 |     0.0006      0.000
```

A 100% guarantee changes senior return by nothing. Since the loss constraint is slack
in every default configuration tested, the guarantee has no effect on α at all — which
makes the UI's Sensitivity Test A ("guarantee ↑") a test of nothing.

**Fix:** route absorbed losses back into the cashflow waterfall — when senior is short
in a period, draw the reserve then the guarantee to top up the payment and record it as
senior cash received. Same change surface as F1; do them together.

### F7 — The LP constrains CVaR on one denominator and reports it on another
`calibration/portfolio/optimizer.py:258` vs `:335` · **FIXED**

The constraint normalises by `total_budget` (`(L.T @ w) / B`); the reported
distribution normalises by capital deployed (`w_v / sum(w)`). In catalytic-budget mode
`sum(w)` can be far below `B`, so the CVaR a user reads can exceed `cvar_max` while the
solver reports `optimal`. `min_expected_return` divides by `B` the same way, tightening
the return floor in proportion to how little is deployed.

**Fix:** pick one basis and use it in both places, or state the basis in the UI.

### F8 — A failed LP silently returns an equal split
`calibration/portfolio/optimizer.py:275` · **FIXED**

When no solver produces a solution, `_solve_lp` returns `np.full(N_v, B / N_v)` — a
plausible-looking allocation that never satisfied any constraint. Nothing in `run()` or
the UI stops on the status, so an infeasible portfolio still renders full results. The
solver loop also re-solves a genuinely infeasible problem three times.

**Fix:** warn, break on `INFEASIBLE` instead of retrying, and have the UI surface a
non-optimal status as an error.

---

## Medium

| Location | Issue |
|---|---|
| `portfolio/models.py:39` | `max_allocation_fraction` is set in four places including the UI, and never read by the LP. A dead knob that looks live. |
| `optimizer.py:314` | α is calibrated against `vehicle.total_capital` but applied to `w_v`, while `grant_reserve` is absolute. When the LP allocates below capacity the calibrated structure no longer holds. |
| `optimizer.py:334` | Portfolio "IRR" is a capital-weighted mean of per-path IRRs. IRRs are not additive, and the `-1.0` / `10.0` sentinels enter the average linearly. |
| `calibration.py:102` | `_find_bracket` is dead code. The 20 monotonicity probes are also discarded — they could seed the brentq bracket and roughly halve search cost. |
| `calibration.py:179` | brentq's root is never checked for feasibility. With `xtol=1e-4` the answer can land marginally on the infeasible side. |
| `utils/irr.py:130` | `batch_irr` is a Python loop running one `brentq` per path, re-run for every α evaluation — the dominant cost in the pipeline. Vectorise the NPV grid or bisect all paths at once. |
| `auth.py:111` | The gate fails *open*: any error reading secrets returns `True`. Fine as documented dev-mode behaviour, risky as deployment behaviour — prefer an explicit `AUTH_DISABLED` opt-in. |
| `utils/stats.py:21` | `nearest_positive_definite` is eigenvalue clipping, not Higham's iterative algorithm as documented. The PD test `eigenvalues <= 0` has no tolerance, so a 1e-16 eigenvalue skips repair and can still fail Cholesky. |
| `simulation.py:84` | `_get_price_params` estimates σ with `np.std` (ddof=0), biasing volatility low on short series, and applies no annualisation despite `load_price_series` accepting sub-annual data. |

---

## Test coverage

The 146 tests are well organised by layer but assert on shapes, ranges, and
non-crashing far more than on economics. Every critical finding lives in that gap.

Three property tests would have caught F2, F3, and F6 on the day they landed:

1. A correlation matrix of ρ produces vehicle cashflows correlated at approximately ρ.
2. A project with a known analytic IRR returns that IRR.
3. Raising guarantee coverage weakly reduces α.

The suite also emits "100.0% of IRR paths returned NaN" during a *passing* test.
Warnings that indicate corrupted output should be promoted to failures.

---

## Suggested order

1. **F2 and F3** — small, self-contained, independently verifiable; ~10 lines between them.
2. **F4** — contained fix in `cvar()`, immediately corrects every reported tail-risk number.
3. **F1 and F6 together** — shared `_cashflow_waterfall` surface; fixing either alone leaves the model half-connected. This is the real work and will move α substantially.
4. **F5** — after the waterfall is fixed, since achievable senior IRR changes once principal is actually repaid. Then add the guards.
5. **F7, F8 and the medium list** — largely hygiene once the engine is right.

Add the three property tests before any of it, so the fixes have something to prove
themselves against.


---

## What the fixes changed

### Resolutions

| # | Resolution |
|---|---|
| F1 | Sinking-fund cash retention in `_cashflow_waterfall`: after coupons, cash is held to `(senior + mezz outstanding) * t / T` before equity receives a residual, and the balance stays liquid so it also absorbs coupon shortfalls. |
| F2 | `reorder_idx = np.argsort(cfs.sum(axis=1))[target_ranks]`. `correlation_matrix` is now documented and tested as a **rank** correlation, with the normal-copula transform applied so the requested rho is the achieved rho. |
| F3 | A same-sign bracket reports the bound it fell past (`10.0` / `-0.999`) instead of `NaN`. `batch_irr` was also rewritten as vectorised bisection — agrees with the scalar solver to 3e-9, ~25x faster. |
| F4 | `cvar()` averages the worst `ceil((1-confidence)*n)` losses by rank instead of selecting `losses >= VaR`. |
| F5 | `investor_hurdle_irr >= senior_coupon` raises at calibration time; the search is capped at `1 - mezzanine_fraction - min_senior_fraction`; the returned alpha is feasibility-checked. The `CalibratorConfig` default hurdle moved 0.08 -> 0.07, because 0.08 collided with the default `senior_coupon` and made the out-of-the-box configuration unsatisfiable. |
| F6 | The grant reserve seeds the retained cash balance and the guarantee is drawn to cover senior shortfalls, capped at `coverage * senior notional`. |
| F7 | The CVaR constraint is expressed on dollar losses against `cvar_max * sum(w)` — linear by positive homogeneity — so it is stated on the same deployed-capital basis the result is reported on. `min_expected_return` likewise multiplies through by `sum(w)` rather than dividing by the fixed budget. |
| F8 | A failed LP warns and returns zeros rather than an equal split, and infeasible/unbounded no longer retries across three solvers. `PortfolioOptimizer` also no longer falls back to `alpha=0.99` when a vehicle cannot be calibrated — it raises, naming the vehicle. |
| Medium | `max_allocation_fraction` enforced; `_find_bracket` dead code removed and the monotonicity probe reused to seed the brentq bracket; `nearest_positive_definite` docstring corrected and its PD test given a tolerance; volatility estimated with `ddof=1`; `CALIBRATION_REQUIRE_AUTH=1` added for fail-closed auth; infinite leverage rendered as `n/a` instead of `1e16x`. |

### Before / after

Identical inputs, measured against both engines: one vehicle, $7.2M capital, three
projects, 10-year horizon, 7% hurdle against an 8% senior coupon, 25% guarantee,
$360k grant reserve, 600 paths, seed 42.

| | before | after |
|---|---|---|
| alpha* | 0.7999 | 0.2209 |
| leverage | 0.25x | 3.53x |
| alpha as rho goes 0.0 -> 0.8 | 0.7960 -> 0.8002 (inert) | 0.1859 -> 0.2692 |
| alpha as guarantee goes 0% -> 60% | 0.7999 -> 0.7999 (inert) | 0.344 -> 0.000 |
| portfolio CVaR, UI sample data | 0.0% always | 13.9% |
| test suite | 146 passed, 76s | 164 passed, 10s |

The pre-fix alpha of 80% was dominated by the cash leak starving senior of principal.
The post-fix figure is the honest requirement for these inputs.

### Two changes that need your sign-off

**1. Sample data was rescaled.** The built-in `run_e2e.py` portfolio and all nine
`examples/ui_sample/*.csv` files described projects returning 3.7x-8.9x their capex
over their lifetime — one earned $2.7M/year on $2M of capex. Vehicles that profitable
need no concessional capital at all, and with F1 fixed they calibrate to alpha = 0.
They produced plausible-looking alphas before only because the cash leak was
cancelling the implausible economics out.

Both sample portfolios have since been **rebuilt from published figures** rather
than rescaled by a fudge factor: ARR credits rated BBB+ averaged $26/tCO2e in 2025
offtakes, planted tropical forest sequesters 4.5-40.7 tCO2e/ha/yr in its first 20
years, and African restoration runs $185-6,000/ha depending on method. Every input
is now sourced in `docs/SAMPLE_DATA_SOURCES.md`, and the UI CSVs are generated by
`scripts/build_sample_data.py` so the assumptions are editable in one place rather
than buried in nine spreadsheets.

The resulting portfolio sits inside the range Convergence reports for real blended
vehicles (leverage 0.30-22, average 4.1x). The largest remaining judgement call is
offtake coverage, which drives alpha more than any other input and is flagged as
such in the sources document.

**2. Two existing IRR tests were changed.** `test_high_irr_capped_at_bracket` asserted
`np.isnan(r)` despite its own name, and its comment read "Should return ~10.0 ... or
NaN"; it now asserts the cap. `test_non_converging_returns_nan_no_warning` asserted
NaN for a cashflow its own comment identified as having an IRR below the bracket
floor; it now asserts the floor and is renamed accordingly. Both encoded the F3
defect. No other pre-existing test was modified.

### Verification

```bash
python -m pytest tests/ -q          # 164 passed
python scripts/review_repro.py      # per-finding checks; exits non-zero on regression
python run_e2e.py --sims 400 --seed 42
python validate_e2e.py              # 18 checks
```
