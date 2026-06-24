"""
app.py  —  Vendor Performance Analytics Dashboard
--------------------------------------------------
Run with:  streamlit run dashboard/app.py

Reads processed Parquet files from data/processed/.
If files don't exist, runs the pipeline automatically.
"""

import streamlit as st
import pandas as pd
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vendor Analytics",
    page_icon="📊",
    layout="wide",
)

PROCESSED = "data/processed"


# ── Data loader with auto-pipeline trigger ────────────────────────────────────
@st.cache_data
def load_data():
    required = ["vendors", "invoices", "deliveries", "gstr2a",
                "scorecard", "gst_recon", "itc_summary", "risk_flags"]
    missing = [f for f in required if not os.path.exists(f"{PROCESSED}/{f}.parquet")]
    if missing:
        st.info("Running pipeline for the first time…")
        from src.pipeline.run_pipeline import main
        main()

    return {
        "vendors":    pd.read_parquet(f"{PROCESSED}/vendors.parquet"),
        "invoices":   pd.read_parquet(f"{PROCESSED}/invoices.parquet"),
        "deliveries": pd.read_parquet(f"{PROCESSED}/deliveries.parquet"),
        "scorecard":  pd.read_parquet(f"{PROCESSED}/scorecard.parquet"),
        "recon":      pd.read_parquet(f"{PROCESSED}/gst_recon.parquet"),
        "itc":        pd.read_parquet(f"{PROCESSED}/itc_summary.parquet"),
        "flags":      pd.read_parquet(f"{PROCESSED}/risk_flags.parquet"),
    }


data = load_data()
sc   = data["scorecard"]
inv  = data["invoices"]
del_ = data["deliveries"]
recon= data["recon"]
flags= data["flags"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📦 Vendor Performance Analytics")
st.caption("GST Compliance & Procurement Intelligence · FY 2024–25")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Vendor Scorecard", "GST Reconciliation", "Risk Flags"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Vendors", len(sc))
    c2.metric("Avg Score", f"{sc['composite_score'].mean():.1f}")
    gst_rate = (recon["recon_status"] == "Matched").mean() * 100
    c3.metric("GST Match Rate", f"{gst_rate:.1f}%")
    total_spend = inv["total_amount"].sum()
    c4.metric("Total Spend", f"₹{total_spend/1e7:.1f}Cr")
    ontime = del_["on_time"].mean() * 100
    c5.metric("On-Time Delivery", f"{ontime:.1f}%")

    st.subheader("Score distribution")
    tier_counts = sc["tier"].value_counts().reindex(["Excellent","Good","Average","Poor"])
    st.bar_chart(tier_counts)

    st.subheader("Category breakdown")
    cat = sc.groupby("category")["composite_score"].mean().sort_values(ascending=False)
    st.bar_chart(cat)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — VENDOR SCORECARD
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns([1, 3])
    with col1:
        cats = ["All"] + sorted(sc["category"].dropna().unique().tolist())
        sel_cat = st.selectbox("Category", cats)
        sel_tier = st.selectbox("Tier", ["All", "Excellent", "Good", "Average", "Poor"])
        min_score = st.slider("Min score", 0, 100, 0)

    filtered = sc.copy()
    if sel_cat != "All":
        filtered = filtered[filtered["category"] == sel_cat]
    if sel_tier != "All":
        filtered = filtered[filtered["tier"] == sel_tier]
    filtered = filtered[filtered["composite_score"] >= min_score]

    with col2:
        st.dataframe(
            filtered[["rank","vendor_name","category","state",
                       "composite_score","payment_score","delivery_score",
                       "quality_score","gst_score","tier"]]
            .rename(columns={
                "vendor_name": "Vendor", "category": "Category", "state": "State",
                "composite_score": "Score", "payment_score": "Payment",
                "delivery_score": "Delivery", "quality_score": "Quality",
                "gst_score": "GST", "tier": "Tier",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Score breakdown — top 10")
    top10 = sc.head(10).set_index("vendor_name")
    st.bar_chart(top10[["payment_score","delivery_score","quality_score","gst_score"]])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — GST RECONCILIATION
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    status_counts = recon["recon_status"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matched",         int(status_counts.get("Matched", 0)))
    c2.metric("Amount Mismatch", int(status_counts.get("Amount Mismatch", 0)), delta_color="inverse")
    c3.metric("Missing in 2A",   int(status_counts.get("Missing in GSTR-2A", 0)), delta_color="inverse")
    itc_risk = recon["itc_risk_amount"].sum()
    c4.metric("ITC at Risk",     f"₹{itc_risk:,.0f}", delta_color="inverse")

    st.subheader("Reconciliation status")
    st.bar_chart(status_counts)

    st.subheader("ITC at risk — by vendor")
    itc_display = data["itc"].copy()
    itc_display["itc_claimed"]  = itc_display["itc_claimed"].apply(lambda x: f"₹{x:,.0f}")
    itc_display["itc_at_risk"]  = itc_display["itc_at_risk"].apply(lambda x: f"₹{x:,.0f}")
    itc_display["match_rate"]   = itc_display["match_rate"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(
        itc_display[["vendor_name","total_invoices","matched_count","match_rate",
                      "itc_claimed","itc_at_risk"]],
        use_container_width=True, hide_index=True,
    )

    st.subheader("Full reconciliation detail")
    st.dataframe(
        recon[["invoice_id","vendor_name","invoice_date","gst_amount",
               "gst_2a_amount","recon_status","itc_risk_amount"]],
        use_container_width=True, hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK FLAGS
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    high   = flags[flags["severity"] == "High"]
    medium = flags[flags["severity"] == "Medium"]

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High Severity",   len(high))
    c2.metric("🟡 Medium Severity", len(medium))
    c3.metric("Total Flags",        len(flags))

    st.subheader("All risk flags")
    def color_severity(val):
        colors = {"High": "background-color:#FCEBEB", "Medium": "background-color:#FAEEDA"}
        return colors.get(val, "")

    st.dataframe(
        flags[["severity","flag","vendor_name","detail"]].style.applymap(
            color_severity, subset=["severity"]
        ),
        use_container_width=True, hide_index=True,
    )