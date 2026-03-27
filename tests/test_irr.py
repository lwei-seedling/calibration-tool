"""Tests for calibration/utils/irr.py.

Covers: normal IRR, sentinel cases, numerical stability at large t, boundary
behaviour, non-convergence, and the diagnostics interface.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from calibration.utils.irr import (
    IrrDiagnostics,
    _irr_single,
    batch_irr,
    clean_irr,
    npv_loss,
)


# ---------------------------------------------------------------------------
# _irr_single — unit cases
# ---------------------------------------------------------------------------

class TestIrrSingle:
    def test_normal_case(self):
        # -100 at t=0, +110 at t=1 → IRR = 10%
        cf = np.array([-100.0, 110.0])
        r = _irr_single(cf)
        assert abs(r - 0.10) < 1e-5

    def test_multi_period(self):
        # -1000, +300/yr for 5 years → IRR ≈ 15.24%
        cf = np.array([-1000.0, 300.0, 300.0, 300.0, 300.0, 300.0])
        r = _irr_single(cf)
        # Verify via NPV ≈ 0 at computed rate
        t = np.arange(len(cf), dtype=float)
        npv = np.sum(cf / (1.0 + r) ** t)
        assert abs(npv) < 1e-4

    def test_no_outflow_returns_nan(self):
        # All non-negative → no investment → undefined
        cf = np.array([0.0, 100.0, 200.0])
        assert np.isnan(_irr_single(cf))

    def test_no_positive_inflows_returns_total_loss_sentinel(self):
        # Negative outflow only → total loss
        cf = np.array([-500.0, 0.0, 0.0, 0.0])
        assert _irr_single(cf) == -1.0

    def test_all_zero_cashflows_returns_nan(self):
        cf = np.zeros(5)
        assert np.isnan(_irr_single(cf))

    def test_single_element_negative_returns_total_loss(self):
        cf = np.array([-1.0])
        assert _irr_single(cf) == -1.0

    def test_single_element_positive_returns_nan(self):
        cf = np.array([1.0])
        assert np.isnan(_irr_single(cf))

    def test_high_irr_capped_at_bracket(self):
        # -1 at t=0, huge payoff at t=1 → IRR very high, brentq hits upper bound
        # Should return ~10.0 (the upper bracket limit) or NaN if NPV(10) > 0
        cf = np.array([-1.0, 1_000_000.0])
        r = _irr_single(cf)
        # NPV(-0.999, cf) is large positive; NPV(10, cf) = -1 + 1e6/11 ≈ +90k > 0
        # So both endpoints are positive → no bracket → NaN
        assert np.isnan(r)

    def test_irr_near_lower_bound(self):
        # Near-total-loss: invest 1000, recover only 1 after 10 years.
        # True IRR: 1 = 1000*(1+r)^10 → r = 0.001^(1/10) - 1 ≈ -0.499
        cf = np.array([-1000.0] + [0.0] * 9 + [1.0])
        r = _irr_single(cf)
        assert r is not None and np.isfinite(r)
        expected = 0.001 ** (1.0 / 10) - 1.0  # ≈ -0.4988
        assert abs(r - expected) < 1e-4

    def test_no_overflow_warning_long_horizon(self):
        # 50-year project — previously triggered overflow in (1+r)**t
        cf = np.array([-1000.0] + [80.0] * 50)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning → test fails
            r = _irr_single(cf)
        assert np.isfinite(r)
        assert -1.0 <= r <= 10.0

    def test_no_overflow_warning_extreme_horizon(self):
        # 100-year project — stress test for log1p stability
        cf = np.array([-1000.0] + [60.0] * 100)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = _irr_single(cf)
        assert np.isfinite(r)

    def test_non_converging_returns_nan_no_warning(self):
        # Construct a cashflow where NPV has same sign at both bracket endpoints:
        # invest 1, get 0.0001 back after 1 year → IRR ≈ -99.99%, below _R_LO=-0.999
        cf = np.array([-1.0, 0.0001])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            r = _irr_single(cf)
        assert np.isnan(r)


# ---------------------------------------------------------------------------
# batch_irr — shape and sentinel propagation
# ---------------------------------------------------------------------------

class TestBatchIrr:
    def test_shape(self):
        cashflows = np.array([
            [-100.0, 110.0],
            [-100.0, 120.0],
            [-100.0,  90.0],
        ])
        irr = batch_irr(cashflows)
        assert irr.shape == (3,)

    def test_correct_values(self):
        cashflows = np.array([
            [-100.0, 110.0],   # 10%
            [-100.0, 200.0],   # 100%
        ])
        irr = batch_irr(cashflows)
        assert abs(irr[0] - 0.10) < 1e-5
        assert abs(irr[1] - 1.00) < 1e-5

    def test_sentinel_total_loss(self):
        cashflows = np.array([[-500.0, 0.0, 0.0]])
        irr = batch_irr(cashflows)
        assert irr[0] == -1.0

    def test_sentinel_no_outflow(self):
        cashflows = np.array([[0.0, 100.0, 200.0]])
        irr = batch_irr(cashflows)
        assert np.isnan(irr[0])

    def test_no_overflow_in_batch(self):
        # 30-year project repeated 200 times
        single = np.array([-1000.0] + [90.0] * 30)
        cashflows = np.tile(single, (200, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            irr = batch_irr(cashflows)
        assert np.all(np.isfinite(irr) | np.isnan(irr))

    def test_diagnostics_returned_when_requested(self):
        # All rows same length (padded with 0.0 to align)
        cashflows = np.array([
            [-100.0, 110.0, 0.0],   # valid IRR
            [-500.0,   0.0, 0.0],   # total loss — no positive inflows
            [  0.0, 100.0, 0.0],    # no outflow — no sign change
        ])
        irr, diag = batch_irr(cashflows, return_diagnostics=True)
        assert isinstance(diag, IrrDiagnostics)
        assert diag.n_computed == 3
        assert diag.n_no_sign_change == 2   # rows 1 and 2

    def test_diagnostics_not_returned_by_default(self):
        cashflows = np.array([[-100.0, 110.0]])
        result = batch_irr(cashflows)
        assert isinstance(result, np.ndarray)

    def test_all_valid_paths_zero_failures(self):
        single = np.array([-1000.0] + [120.0] * 15)
        cashflows = np.tile(single, (100, 1))
        irr, diag = batch_irr(cashflows, return_diagnostics=True)
        assert diag.n_no_sign_change == 0
        assert diag.n_failures == 0
        assert np.all(np.isfinite(irr))


# ---------------------------------------------------------------------------
# clean_irr
# ---------------------------------------------------------------------------

class TestCleanIrr:
    def test_nan_replaced_with_minus_one(self):
        v = np.array([0.10, float("nan"), -1.0])
        out = clean_irr(v)
        assert out[1] == -1.0

    def test_pos_inf_capped(self):
        v = np.array([float("inf"), 0.05])
        out = clean_irr(v)
        assert out[0] == 10.0

    def test_neg_inf_becomes_minus_one(self):
        v = np.array([float("-inf"), 0.05])
        out = clean_irr(v)
        assert out[0] == -1.0

    def test_warns_when_nan_fraction_high(self):
        v = np.full(100, float("nan"))
        with pytest.warns(RuntimeWarning, match="NaN"):
            clean_irr(v)

    def test_no_warn_when_nan_fraction_low(self):
        v = np.array([0.10, 0.12, float("nan")])  # 1/3 nan → above 5% threshold
        # Only testing that below-threshold case doesn't warn
        v_low = np.array([0.10] * 99 + [float("nan")])  # 1% nan → no warn
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            clean_irr(v_low)


# ---------------------------------------------------------------------------
# npv_loss — unchanged but verify no regression
# ---------------------------------------------------------------------------

class TestNpvLoss:
    def test_zero_discount(self):
        cf = np.array([[-100.0, 50.0, 60.0]])  # NPV = 10 → loss = 0
        loss = npv_loss(cf, discount_rate=0.0)
        assert loss[0] == 0.0

    def test_positive_loss(self):
        cf = np.array([[-100.0, 30.0, 30.0]])  # NPV = -40 → loss = 40
        loss = npv_loss(cf, discount_rate=0.0)
        assert abs(loss[0] - 40.0) < 1e-10

    def test_with_discount_rate(self):
        cf = np.array([[-100.0, 0.0, 110.0]])  # PV = 110/1.05^2 ≈ 99.77 → loss > 0
        loss = npv_loss(cf, discount_rate=0.05)
        assert loss[0] > 0.0
