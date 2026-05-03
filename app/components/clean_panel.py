"""
DataPilot AI — Data Cleaning Panel Component (Stable Production Version)
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

from app.modules.cleaner import (
    detect_cleaning_opportunities,
    apply_cleaning_pipeline,
)


def render_clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render the data cleaning UI. Returns the cleaned dataframe.
    """

    # ── Ensure session state keys exist ─────────────────────────────
    if "cleaned_df" not in st.session_state:
        st.session_state["cleaned_df"] = None
    if "cleaning_ops" not in st.session_state:
        st.session_state["cleaning_ops"] = []

    st.markdown("### 🧹 Data Cleaning Studio")
    st.caption("Clean your dataset interactively. Preview changes before applying.")

    # ── Dataset Status (NEW FEATURE) ────────────────────────────────
    current_df = st.session_state.get("cleaned_df")
    if current_df is None or current_df.empty:
        current_df = df

    col1, col2 = st.columns(2)
    col1.metric("📄 Rows", f"{current_df.shape[0]:,}")
    col2.metric("📊 Columns", f"{current_df.shape[1]:,}")

    st.divider()

    # ── Detect issues ───────────────────────────────────────────────
    with st.spinner("Scanning for cleaning opportunities..."):
        issues = detect_cleaning_opportunities(df)

    if not issues:
        st.success("✅ No major data quality issues detected! Dataset looks clean.")
    else:
        st.markdown(f"**Found {len(issues)} area(s) needing attention:**")

        if "missing_values" in issues:
            st.warning(f"⚠️ Missing values in {len(issues['missing_values'])} column(s)")

        if "duplicates" in issues:
            st.warning(f"⚠️ {issues['duplicates']} duplicate rows")

        if "outliers" in issues:
            st.warning(f"⚠️ Outliers detected in {len(issues['outliers'])} column(s)")

        if "constant_columns" in issues:
            st.warning(f"⚠️ Constant columns: {issues['constant_columns']}")

    st.divider()

    # ── Cleaning configuration ──────────────────────────────────────
    config = {}

    # Missing values
    st.markdown("**1️⃣ Missing Value Handling**")
    missing_info = issues.get("missing_values", {})
    if missing_info:
        missing_strategy = {}
        for col, info in missing_info.items():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"`{col}` — {info['pct']}% missing ({info['count']} rows)")
            with col2:
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                options = (
                    ["mean", "median", "mode", "ffill", "bfill", "zero", "drop"]
                    if is_numeric
                    else ["mode", "ffill", "bfill", "drop"]
                )
                method = st.selectbox("Method:", options, key=f"miss_{col}")
                missing_strategy[col] = method

        config["missing_strategy"] = missing_strategy
    else:
        st.success("✅ No missing values")

    # Duplicates
    st.markdown("**2️⃣ Duplicate Rows**")
    dup_count = issues.get("duplicates", 0)
    if dup_count:
        config["remove_duplicates"] = st.checkbox(
            f"Remove {dup_count} duplicate rows", value=True
        )
    else:
        st.success("✅ No duplicates")

    # Constant columns
    if issues.get("constant_columns"):
        st.markdown("**3️⃣ Constant Columns**")
        config["drop_constants"] = st.checkbox(
            f"Drop constant columns: {issues['constant_columns']}", value=True
        )

    # Outliers
    st.markdown("**4️⃣ Outlier Handling**")
    outlier_cols = issues.get("outliers", {})
    if outlier_cols:
        st.write(f"Outliers detected in: {list(outlier_cols.keys())[:5]}")
        outlier_method = st.selectbox(
            "Outlier strategy:",
            ["None", "iqr_clip", "zscore_clip", "winsorize", "iqr_drop"],
            index=1,
        )
        if outlier_method != "None":
            config["outlier_strategy"] = outlier_method
    else:
        st.success("✅ No significant outliers")

    # Encoding
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        st.markdown("**5️⃣ Categorical Encoding**")
        encode_method = st.selectbox(
            "Encoding method:", ["None", "onehot", "label", "frequency"], index=0
        )
        if encode_method != "None":
            config["encode_method"] = encode_method

    # Scaling
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        st.markdown("**6️⃣ Feature Scaling**")
        scale_method = st.selectbox(
            "Scaling method:", ["None", "minmax", "zscore", "robust"], index=0
        )
        if scale_method != "None":
            config["scale_method"] = scale_method

    st.divider()

    # ── Safe cleaned_df assignment (FIXED) ──────────────────────────
    cleaned_df = st.session_state.get("cleaned_df")
    if cleaned_df is None:
        cleaned_df = df.copy()

    # ── Preview + Apply Cleaning ────────────────────────────────────
    if st.button("👁️ Preview Cleaned Data", use_container_width=True):
        with st.spinner("Applying cleaning steps..."):
            try:
                cleaned_df, ops_log = apply_cleaning_pipeline(df, config)
                st.session_state["cleaned_df"] = cleaned_df
                st.session_state["cleaning_ops"] = ops_log
            except Exception as e:
                st.error(f"Cleaning error: {e}")
                cleaned_df = df.copy()
                st.session_state["cleaned_df"] = cleaned_df
                st.session_state["cleaning_ops"] = []

    # ── Show metrics if cleaned ─────────────────────────────────────
    if st.session_state["cleaned_df"] is not None:

        cleaned_df = st.session_state["cleaned_df"]

        st.markdown("**Before vs After:**")
        c1, c2 = st.columns(2)

        c1.metric("Original Rows", f"{len(df):,}")
        c2.metric(
            "Cleaned Rows",
            f"{len(cleaned_df):,}",
            delta=f"{len(cleaned_df) - len(df):,}",
        )

        c1.metric("Original Cols", f"{df.shape[1]}")
        c2.metric(
            "Cleaned Cols",
            f"{cleaned_df.shape[1]}",
            delta=f"{cleaned_df.shape[1] - df.shape[1]:,}",
        )

        if st.session_state["cleaning_ops"]:
            with st.expander("📋 Operations Applied"):
                for op in st.session_state["cleaning_ops"]:
                    st.write(f"→ {op}")

        st.dataframe(cleaned_df.head(20), use_container_width=True)

    # ── Safe dataset for download (FIXED) ───────────────────────────
    dataset_for_download = st.session_state.get("cleaned_df")
    if dataset_for_download is None or dataset_for_download.empty:
        dataset_for_download = df

    # ── Download Section ────────────────────────────────────────────
    if dataset_for_download is not None and not dataset_for_download.empty:

        st.markdown("### 📥 Export Dataset")

        col1, col2 = st.columns(2)

        # CSV
        with col1:
            csv_bytes = dataset_for_download.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv_bytes,
                "datapilot_dataset.csv",
                mime="text/csv",
                key="dl_csv",
            )

        # Excel
        with col2:
            buffer = io.BytesIO()
            try:
                dataset_for_download.to_excel(buffer, index=False, engine="openpyxl")
                buffer.seek(0)
                st.download_button(
                    "Download Excel",
                    buffer.getvalue(),
                    "datapilot_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_xlsx",
                )
            except Exception as e:
                st.error(f"Excel export failed: {e}")

    return dataset_for_download 
