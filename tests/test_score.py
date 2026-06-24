"""
test_score.py
-------------
Unit tests for the vendor scoring engine (Polars).
Run with:  pytest tests/
"""

import pytest
import polars as pl
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.score import (
    compute_payment_score,
    compute_delivery_score,
    compute_quality_score,
    WEIGHTS,
)


def make_invoices(**kwargs):
    defaults = dict(
        invoice_id=["I1", "I2"],
        vendor_id=["V1", "V1"],
        status=["Paid", "Paid"],
        is_late_payment=[False, False],
        is_overdue=[False, False],
        gst_amount=[1800.0, 3600.0],
        total_amount=[10000.0, 20000.0],
    )
    defaults.update(kwargs)
    return pl.DataFrame(defaults)


def make_deliveries(**kwargs):
    defaults = dict(
        delivery_id=["D1", "D2"],
        vendor_id=["V1", "V1"],
        on_time=[True, True],
        fill_rate_pct=[100.0, 100.0],
        quality_score=[4.5, 4.8],
        defect_rate_pct=[0.5, 0.2],
    )
    defaults.update(kwargs)
    return pl.DataFrame(defaults)


def get_score(df, vendor="V1", col="payment_score"):
    return df.filter(pl.col("vendor_id") == vendor)[col][0]


class TestPaymentScore:
    def test_perfect_payment_gives_100(self):
        result = compute_payment_score(make_invoices())
        assert get_score(result, col="payment_score") == pytest.approx(100.0)

    def test_all_late_payments_penalised(self):
        result = compute_payment_score(make_invoices(is_late_payment=[True, True]))
        assert get_score(result, col="payment_score") < 100.0

    def test_score_clipped_at_zero(self):
        result = compute_payment_score(
            make_invoices(is_late_payment=[True, True], is_overdue=[True, True])
        )
        assert get_score(result, col="payment_score") >= 0.0


class TestDeliveryScore:
    def test_perfect_on_time_full_fill_gives_100(self):
        result = compute_delivery_score(make_deliveries())
        assert get_score(result, col="delivery_score") == pytest.approx(100.0)

    def test_late_delivery_reduces_score(self):
        result = compute_delivery_score(make_deliveries(on_time=[False, False]))
        assert get_score(result, col="delivery_score") < 100.0

    def test_partial_fill_reduces_score(self):
        result = compute_delivery_score(make_deliveries(fill_rate_pct=[80.0, 80.0]))
        assert get_score(result, col="delivery_score") < 100.0


class TestQualityScore:
    def test_perfect_quality_no_defects(self):
        result = compute_quality_score(
            make_deliveries(quality_score=[5.0, 5.0], defect_rate_pct=[0.0, 0.0])
        )
        assert get_score(result, col="quality_score") == pytest.approx(100.0)

    def test_high_defect_rate_reduces_score(self):
        result = compute_quality_score(make_deliveries(defect_rate_pct=[8.0, 9.0]))
        assert get_score(result, col="quality_score") < 80.0


class TestWeights:
    def test_weights_sum_to_one(self):
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)