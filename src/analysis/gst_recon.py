"""
gst_recon.py
------------
GST GSTR-2A Reconciliation engine — powered by DuckDB.

Why DuckDB here:
  - Reconciliation is fundamentally a SQL JOIN problem
  - DuckDB queries Parquet files directly (no load step)
  - Identical SQL runs on Amazon Athena in production — zero rewrite
  - In-process, no server needed locally

Classifies each invoice as:
  Matched             → safe to claim ITC
  Amount Mismatch     → GST amount differs > 5% or > ₹5,000
  Missing in GSTR-2A  → supplier hasn't filed
  GSTIN Invalid       → GSTIN failed format validation
"""

import duckdb
import pandas as pd
import logging

log = logging.getLogger(__name__)

VARIANCE_THRESHOLD_PCT = 0.05
VARIANCE_THRESHOLD_ABS = 5000


def reconcile(
    invoices_path: str = "data/processed/invoices.parquet",
    gstr2a_path:   str = "data/processed/gstr2a.parquet",
    vendors_path:  str = "data/processed/vendors.parquet",
) -> pd.DataFrame:
    """
    Run reconciliation entirely in DuckDB SQL on Parquet files.
    Returns a pandas DataFrame with one row per invoice.

    The same SQL works on Amazon Athena — just swap the file paths
    for s3:// URIs and register them as external tables.
    """
    con = duckdb.connect()

    # DuckDB can query Parquet directly — no loading required
    recon = con.execute(f"""
        WITH invoice_vendor AS (
            SELECT
                i.invoice_id,
                i.vendor_id,
                i.invoice_date,
                i.gst_amount,
                i.total_amount,
                i.status,
                v.vendor_name,
                v.gstin,
                v.gstin_valid
            FROM read_parquet('{invoices_path}') i
            LEFT JOIN read_parquet('{vendors_path}') v
                ON i.vendor_id = v.vendor_id
        ),

        joined AS (
            SELECT
                iv.*,
                g.gst_amount  AS gst_2a_amount,
                g.filing_status,
                g.filing_date AS supplier_filing_date
            FROM invoice_vendor iv
            LEFT JOIN read_parquet('{gstr2a_path}') g
                ON iv.invoice_id = g.invoice_number
        ),

        classified AS (
            SELECT
                *,
                CASE
                    WHEN gstin_valid = false
                        THEN 'GSTIN Invalid'
                    WHEN gst_2a_amount IS NULL
                        THEN 'Missing in GSTR-2A'
                    WHEN ABS(gst_amount - gst_2a_amount) > {VARIANCE_THRESHOLD_ABS}
                      OR ABS(gst_amount - gst_2a_amount) / NULLIF(gst_amount, 0)
                         > {VARIANCE_THRESHOLD_PCT}
                        THEN 'Amount Mismatch'
                    ELSE 'Matched'
                END AS recon_status,

                CASE
                    WHEN gstin_valid = false
                      OR gst_2a_amount IS NULL
                      OR ABS(gst_amount - gst_2a_amount) > {VARIANCE_THRESHOLD_ABS}
                      OR ABS(gst_amount - gst_2a_amount) / NULLIF(gst_amount, 0)
                         > {VARIANCE_THRESHOLD_PCT}
                        THEN gst_amount
                    ELSE 0
                END AS itc_risk_amount
            FROM joined
        )

        SELECT * FROM classified
        ORDER BY itc_risk_amount DESC, invoice_date
    """).df()

    con.close()

    log.info("Reconciliation summary:")
    log.info(recon["recon_status"].value_counts().to_string())
    log.info(f"Total ITC at risk: ₹{recon['itc_risk_amount'].sum():,.0f}")
    return recon


def itc_summary(recon: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ITC claimed vs at-risk by vendor using DuckDB SQL.
    """
    con = duckdb.connect()
    result = con.execute("""
        SELECT
            vendor_id,
            vendor_name,
            COUNT(*)                                        AS total_invoices,
            SUM(gst_amount)                                 AS itc_claimed,
            SUM(itc_risk_amount)                            AS itc_at_risk,
            SUM(CASE WHEN recon_status = 'Matched' THEN 1 ELSE 0 END) AS matched_count,
            ROUND(
                SUM(CASE WHEN recon_status = 'Matched' THEN 1.0 ELSE 0 END)
                / COUNT(*) * 100, 1
            )                                               AS match_rate
        FROM recon
        GROUP BY vendor_id, vendor_name
        ORDER BY itc_at_risk DESC
    """).df()
    con.close()
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    recon   = reconcile()
    summary = itc_summary(recon)

    print("\n--- ITC Summary by Vendor ---")
    print(summary.to_string(index=False))

    recon.to_parquet("data/processed/gst_recon.parquet", index=False)
    summary.to_parquet("data/processed/itc_summary.parquet", index=False)