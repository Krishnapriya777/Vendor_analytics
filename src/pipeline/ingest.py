"""
ingest.py
---------
Step 1 of the pipeline: Load raw CSV files, validate schema,
clean data, and save to processed/ as clean Parquet files.

In production this would run as an AWS Glue PySpark job,
reading from S3 and writing back to S3.
"""

import pandas as pd
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

EXPECTED_COLUMNS = {
    "vendors":   ["vendor_id", "vendor_name", "category", "gstin", "state", "city",
                  "onboarded_date", "payment_terms_days", "contact_email"],
    "invoices":  ["invoice_id", "vendor_id", "invoice_date", "due_date", "invoice_amount",
                  "taxable_amount", "gst_rate", "gst_amount", "total_amount",
                  "payment_date", "status", "po_number"],
    "gstr2a":    ["gstin", "supplier_gstin", "invoice_number", "invoice_date",
                  "taxable_amount", "gst_amount", "total_amount", "filing_status", "filing_date"],
    "deliveries":["delivery_id", "vendor_id", "po_number", "promised_date", "actual_date",
                  "quantity_ordered", "quantity_delivered", "quality_score", "defect_rate_pct"],
}

DATE_COLUMNS = {
    "vendors":    ["onboarded_date"],
    "invoices":   ["invoice_date", "due_date", "payment_date"],
    "gstr2a":     ["invoice_date", "filing_date"],
    "deliveries": ["promised_date", "actual_date"],
}


def validate_schema(df: pd.DataFrame, name: str) -> bool:
    expected = set(EXPECTED_COLUMNS[name])
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        log.error(f"[{name}] Missing columns: {missing}")
        return False
    log.info(f"[{name}] Schema OK — {len(df)} rows loaded")
    return True


def clean_vendors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="vendor_id")
    df["onboarded_date"] = pd.to_datetime(df["onboarded_date"], errors="coerce")
    df["gstin"] = df["gstin"].str.strip().str.upper()
    # Basic GSTIN format check: 15 alphanumeric characters
    df["gstin_valid"] = df["gstin"].str.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z]{1}\d[Z][A-Z\d]$")
    invalid = df[~df["gstin_valid"]]
    if not invalid.empty:
        log.warning(f"[vendors] {len(invalid)} invalid GSTINs: {invalid['vendor_id'].tolist()}")
    return df


def clean_invoices(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["invoice_date", "due_date", "payment_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["invoice_amount", "taxable_amount", "gst_amount", "total_amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset="invoice_id")
    # Derived fields
    df["days_to_payment"] = (df["payment_date"] - df["invoice_date"]).dt.days
    df["is_overdue"] = (
        df["payment_date"].isna() & (pd.Timestamp.today() > df["due_date"])
    )
    df["is_late_payment"] = df["days_to_payment"] > df["invoice_id"].map(
        lambda x: 30  # simplified; in prod join with vendor payment_terms
    )
    log.info(f"[invoices] Overdue count: {df['is_overdue'].sum()}")
    return df


def clean_gstr2a(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["invoice_date", "filing_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["taxable_amount", "gst_amount", "total_amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset="invoice_number")
    return df


def clean_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["promised_date", "actual_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["quantity_ordered", "quantity_delivered", "quality_score", "defect_rate_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["delivery_delay_days"] = (df["actual_date"] - df["promised_date"]).dt.days
    df["on_time"] = df["delivery_delay_days"] <= 0
    df["fill_rate_pct"] = (df["quantity_delivered"] / df["quantity_ordered"] * 100).round(2)
    log.info(f"[deliveries] On-time rate: {df['on_time'].mean():.1%}")
    return df


CLEANERS = {
    "vendors":    clean_vendors,
    "invoices":   clean_invoices,
    "gstr2a":     clean_gstr2a,
    "deliveries": clean_deliveries,
}


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    start = datetime.now()
    log.info("=== Ingestion pipeline started ===")

    for name in EXPECTED_COLUMNS:
        path = os.path.join(RAW_DIR, f"{name}.csv")
        if not os.path.exists(path):
            log.error(f"File not found: {path}")
            continue
        df = pd.read_csv(path)
        if not validate_schema(df, name):
            continue
        df = CLEANERS[name](df)
        out = os.path.join(PROCESSED_DIR, f"{name}.parquet")
        df.to_parquet(out, index=False)
        log.info(f"[{name}] Saved → {out}")

    elapsed = (datetime.now() - start).seconds
    log.info(f"=== Ingestion complete in {elapsed}s ===")


if __name__ == "__main__":
    run()