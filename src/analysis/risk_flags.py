"""
risk_flags.py
-------------
Auto-detect vendor risk signals from the scorecard and reconciliation data.

Each rule returns a DataFrame of flagged vendor_ids with a reason and severity.
In production, flags trigger email alerts via AWS Lambda + SES.
"""

import pandas as pd

SEVERITY_HIGH   = "High"
SEVERITY_MEDIUM = "Medium"
SEVERITY_LOW    = "Low"


def flag_low_scores(scorecard: pd.DataFrame, threshold: float = 60.0) -> pd.DataFrame:
    flagged = scorecard[scorecard["composite_score"] < threshold].copy()
    flagged["flag"] = "Low composite score"
    flagged["severity"] = SEVERITY_HIGH
    flagged["detail"] = flagged["composite_score"].apply(lambda s: f"Score: {s}")
    return flagged[["vendor_id", "vendor_name", "flag", "severity", "detail"]]


def flag_gstin_invalid(vendors: pd.DataFrame) -> pd.DataFrame:
    flagged = vendors[~vendors["gstin_valid"]].copy()
    flagged["flag"] = "Invalid GSTIN"
    flagged["severity"] = SEVERITY_HIGH
    flagged["detail"] = flagged["gstin"].apply(lambda g: f"GSTIN {g} failed format check")
    return flagged[["vendor_id", "vendor_name", "flag", "severity", "detail"]]


def flag_itc_at_risk(itc_summary: pd.DataFrame, threshold: float = 10000) -> pd.DataFrame:
    flagged = itc_summary[itc_summary["itc_at_risk"] > threshold].copy()
    flagged["flag"] = "ITC at risk"
    flagged["severity"] = SEVERITY_HIGH
    flagged["detail"] = flagged["itc_at_risk"].apply(lambda v: f"₹{v:,.0f} ITC at risk")
    return flagged[["vendor_id", "vendor_name", "flag", "severity", "detail"]]


def flag_poor_delivery(scorecard: pd.DataFrame, threshold: float = 65.0) -> pd.DataFrame:
    flagged = scorecard[scorecard["delivery_score"] < threshold].copy()
    flagged["flag"] = "Poor delivery performance"
    flagged["severity"] = SEVERITY_MEDIUM
    flagged["detail"] = flagged["delivery_score"].apply(lambda s: f"Delivery score: {s}")
    return flagged[["vendor_id", "vendor_name", "flag", "severity", "detail"]]


def flag_gst_compliance(scorecard: pd.DataFrame, threshold: float = 70.0) -> pd.DataFrame:
    flagged = scorecard[scorecard["gst_score"] < threshold].copy()
    flagged["flag"] = "Low GST compliance"
    flagged["severity"] = SEVERITY_MEDIUM
    flagged["detail"] = flagged["gst_score"].apply(lambda s: f"GST score: {s}")
    return flagged[["vendor_id", "vendor_name", "flag", "severity", "detail"]]


def flag_spend_concentration(invoices: pd.DataFrame, top_n: int = 2, threshold_pct: float = 35.0) -> pd.DataFrame:
    spend = invoices.groupby("vendor_id")["total_amount"].sum()
    total = spend.sum()
    top = spend.nlargest(top_n)
    concentration_pct = top.sum() / total * 100
    if concentration_pct > threshold_pct:
        return pd.DataFrame([{
            "vendor_id": ", ".join(top.index),
            "vendor_name": "Multiple",
            "flag": "Spend concentration risk",
            "severity": SEVERITY_MEDIUM,
            "detail": f"Top {top_n} vendors = {concentration_pct:.1f}% of total spend",
        }])
    return pd.DataFrame()


def run_all_flags(scorecard, vendors, itc_summary_df, invoices) -> pd.DataFrame:
    all_flags = pd.concat([
        flag_low_scores(scorecard),
        flag_gstin_invalid(vendors),
        flag_itc_at_risk(itc_summary_df),
        flag_poor_delivery(scorecard),
        flag_gst_compliance(scorecard),
        flag_spend_concentration(invoices),
    ], ignore_index=True)

    severity_order = {SEVERITY_HIGH: 0, SEVERITY_MEDIUM: 1, SEVERITY_LOW: 2}
    all_flags["_order"] = all_flags["severity"].map(severity_order)
    all_flags = all_flags.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return all_flags


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    scorecard  = pd.read_parquet("data/processed/scorecard.parquet")
    vendors    = pd.read_parquet("data/processed/vendors.parquet")
    invoices   = pd.read_parquet("data/processed/invoices.parquet")
    itc        = pd.read_parquet("data/processed/itc_summary.parquet")

    flags = run_all_flags(scorecard, vendors, itc, invoices)
    print(flags.to_string(index=False))
    flags.to_parquet("data/processed/risk_flags.parquet", index=False)