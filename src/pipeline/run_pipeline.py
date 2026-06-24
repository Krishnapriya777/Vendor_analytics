"""
run_pipeline.py
---------------
Master runner. Executes all pipeline steps in sequence:

  1. Ingest & clean raw CSVs  →  Parquet  (pandas)
  2. Vendor scorecard          →  Parquet  (Polars)
  3. GST reconciliation        →  Parquet  (DuckDB)
  4. Risk flags                →  Parquet  (pandas)

Run from project root:
  python src/pipeline/run_pipeline.py
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import polars as pl
from pipeline.ingest import run as run_ingest
from analysis.score import build_scorecard
from analysis.gst_recon import reconcile, itc_summary
from analysis.risk_flags import run_all_flags

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

PROCESSED = "data/processed"


def main():
    log.info("╔══════════════════════════════════════════╗")
    log.info("║   Vendor Analytics Pipeline — Start      ║")
    log.info("╚══════════════════════════════════════════╝")

    # Step 1: Ingest with pandas
    log.info("── Step 1: Ingestion & Cleaning (pandas) ──")
    run_ingest()

    # Step 2: Score with Polars
    log.info("── Step 2: Vendor Scoring (Polars) ──")
    vendors    = pl.read_parquet(f"{PROCESSED}/vendors.parquet")
    invoices   = pl.read_parquet(f"{PROCESSED}/invoices.parquet")
    deliveries = pl.read_parquet(f"{PROCESSED}/deliveries.parquet")
    gstr2a     = pl.read_parquet(f"{PROCESSED}/gstr2a.parquet")

    scorecard = build_scorecard(vendors, invoices, deliveries, gstr2a)
    scorecard.write_parquet(f"{PROCESSED}/scorecard.parquet")

    # Step 3: GST Reconciliation with DuckDB (reads Parquet directly)
    log.info("── Step 3: GST Reconciliation (DuckDB) ──")
    recon = reconcile(
        invoices_path=f"{PROCESSED}/invoices.parquet",
        gstr2a_path=f"{PROCESSED}/gstr2a.parquet",
        vendors_path=f"{PROCESSED}/vendors.parquet",
    )
    itc = itc_summary(recon)
    recon.to_parquet(f"{PROCESSED}/gst_recon.parquet", index=False)
    itc.to_parquet(f"{PROCESSED}/itc_summary.parquet", index=False)

    # Step 4: Risk Flags with pandas
    log.info("── Step 4: Risk Flagging (pandas) ──")
    sc_pd  = scorecard.to_pandas()
    ven_pd = vendors.to_pandas()
    inv_pd = invoices.to_pandas()

    flags = run_all_flags(sc_pd, ven_pd, itc, inv_pd)
    flags.to_parquet(f"{PROCESSED}/risk_flags.parquet", index=False)

    log.info("╔══════════════════════════════════════════╗")
    log.info("║   Pipeline Complete ✓                    ║")
    log.info("╚══════════════════════════════════════════╝")

    print("\n📊 Scorecard preview:")
    print(scorecard.select(["rank","vendor_name","composite_score","tier"]).head(10))
    print(f"\n⚠️  Risk flags: {len(flags)} total")
    print(flags[["severity","flag","vendor_name"]].to_string(index=False))


if __name__ == "__main__":
    main()