"""
score.py
--------
Step 2: Build a composite vendor performance score (0–100)
using Polars for fast grouped aggregations.

Why Polars over pandas here:
  - Lazy evaluation — only computes what's needed
  - Multi-threaded by default — faster on large vendor/invoice datasets
  - Expressive chained API for grouped transforms

Dimensions & weights:
  Payment behaviour     25%
  Delivery performance  30%
  Quality               25%
  GST compliance        20%
"""

import polars as pl
import logging

log = logging.getLogger(__name__)

WEIGHTS = {
    "payment_score":  0.25,
    "delivery_score": 0.30,
    "quality_score":  0.25,
    "gst_score":      0.20,
}


def compute_payment_score(invoices: pl.DataFrame) -> pl.DataFrame:
    """
    Score = 100 − (late_rate × 60) − (overdue_rate × 40)
    Only consider Paid invoices for late rate.
    """
    paid = invoices.filter(pl.col("status") == "Paid")

    late_rate = (
        paid.group_by("vendor_id")
        .agg(pl.col("is_late_payment").mean().alias("late_rate"))
    )
    overdue_rate = (
        invoices.group_by("vendor_id")
        .agg(pl.col("is_overdue").mean().alias("overdue_rate"))
    )
    return (
        late_rate.join(overdue_rate, on="vendor_id", how="full", coalesce=True)
        .with_columns([
            pl.col("late_rate").fill_null(0),
            pl.col("overdue_rate").fill_null(0),
        ])
        .with_columns(
            (100 - pl.col("late_rate") * 60 - pl.col("overdue_rate") * 40)
            .clip(0, 100)
            .alias("payment_score")
        )
        .select(["vendor_id", "payment_score"])
    )


def compute_delivery_score(deliveries: pl.DataFrame) -> pl.DataFrame:
    """
    Score = (on_time_rate × 60) + (avg_fill_rate / 100 × 40)
    """
    return (
        deliveries.group_by("vendor_id")
        .agg([
            pl.col("on_time").mean().alias("on_time_rate"),
            pl.col("fill_rate_pct").mean().fill_null(100).alias("avg_fill_rate"),
        ])
        .with_columns(
            (pl.col("on_time_rate") * 60 + (pl.col("avg_fill_rate") / 100) * 40)
            .clip(0, 100)
            .alias("delivery_score")
        )
        .select(["vendor_id", "delivery_score"])
    )


def compute_quality_score(deliveries: pl.DataFrame) -> pl.DataFrame:
    """
    Score = (avg_quality / 5 × 70) + ((1 − avg_defect / 10) × 30)
    """
    return (
        deliveries.group_by("vendor_id")
        .agg([
            pl.col("quality_score").mean().fill_null(3.0).alias("avg_quality"),
            pl.col("defect_rate_pct").mean().fill_null(0).alias("avg_defect"),
        ])
        .with_columns(
            ((pl.col("avg_quality") / 5) * 70 + (1 - pl.col("avg_defect") / 10) * 30)
            .clip(0, 100)
            .alias("quality_score")
        )
        .select(["vendor_id", "quality_score"])
    )


def compute_gst_score(invoices: pl.DataFrame, gstr2a: pl.DataFrame) -> pl.DataFrame:
    """
    Match invoices against GSTR-2A; score = matched% per vendor.
    """
    g2a = gstr2a.select([
        pl.col("invoice_number").alias("invoice_id"),
        pl.col("gst_amount").alias("gst_2a_amount"),
    ])
    merged = (
        invoices.join(g2a, on="invoice_id", how="left")
        .with_columns(
            (
                pl.col("gst_2a_amount").is_not_null()
                & ((pl.col("gst_amount") - pl.col("gst_2a_amount")).abs()
                   / pl.col("gst_amount").replace(0, None) <= 0.01)
            ).alias("gst_matched")
        )
    )
    return (
        merged.group_by("vendor_id")
        .agg([
            pl.len().alias("total_invoices"),
            pl.col("gst_matched").sum().alias("matched"),
        ])
        .with_columns(
            (pl.col("matched") / pl.col("total_invoices") * 100)
            .clip(0, 100)
            .alias("gst_score")
        )
        .select(["vendor_id", "gst_score"])
    )


def build_scorecard(vendors, invoices, deliveries, gstr2a) -> pl.DataFrame:
    # Convert pandas → Polars if needed (ingest.py saves parquet, we reload as Polars)
    if not isinstance(vendors, pl.DataFrame):
        vendors    = pl.from_pandas(vendors)
        invoices   = pl.from_pandas(invoices)
        deliveries = pl.from_pandas(deliveries)
        gstr2a     = pl.from_pandas(gstr2a)

    payment  = compute_payment_score(invoices)
    delivery = compute_delivery_score(deliveries)
    quality  = compute_quality_score(deliveries)
    gst      = compute_gst_score(invoices, gstr2a)

    scorecard = (
        vendors.select(["vendor_id", "vendor_name", "category", "state"])
        .join(payment,  on="vendor_id", how="left")
        .join(delivery, on="vendor_id", how="left")
        .join(quality,  on="vendor_id", how="left")
        .join(gst,      on="vendor_id", how="left")
        .with_columns([
            pl.col(dim).fill_null(50).round(1)
            for dim in WEIGHTS
        ])
        .with_columns(
            sum(pl.col(dim) * w for dim, w in WEIGHTS.items())
            .round(1)
            .alias("composite_score")
        )
        .with_columns(
            pl.when(pl.col("composite_score") >= 90).then(pl.lit("Excellent"))
            .when(pl.col("composite_score") >= 75).then(pl.lit("Good"))
            .when(pl.col("composite_score") >= 60).then(pl.lit("Average"))
            .otherwise(pl.lit("Poor"))
            .alias("tier")
        )
        .sort("composite_score", descending=True)
        .with_row_index("rank", offset=1)
    )

    log.info(f"Scorecard built for {len(scorecard)} vendors")
    log.info(f"Tier counts:\n{scorecard.group_by('tier').len()}")
    return scorecard


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    vendors    = pl.read_parquet("data/processed/vendors.parquet")
    invoices   = pl.read_parquet("data/processed/invoices.parquet")
    deliveries = pl.read_parquet("data/processed/deliveries.parquet")
    gstr2a     = pl.read_parquet("data/processed/gstr2a.parquet")

    scorecard = build_scorecard(vendors, invoices, deliveries, gstr2a)
    print(scorecard.select(["rank", "vendor_name", "composite_score", "tier"]))
    scorecard.write_parquet("data/processed/scorecard.parquet")