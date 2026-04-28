---
description: Fast end-to-end calibration smoke check (e2e + tests)
---

Run a quick deterministic sanity check on the full calibration pipeline and report the results.

Steps:
1. Run `python run_e2e.py --sims 200 --seed 42` — fast Monte Carlo run on the built-in 3-vehicle sample portfolio (deterministic via seed).
2. Run `python -m pytest -x -q` — full unit-test suite, fail-fast.
3. Summarise in 5–8 lines:
   - Calibrated α (catalytic fraction) per vehicle
   - Total commercial capital mobilised
   - Portfolio leverage ratio
   - Test pass/fail count; on failure, list the first failing test
4. If e2e or tests fail, stop and surface the error — do not attempt fixes unless the user asks.
