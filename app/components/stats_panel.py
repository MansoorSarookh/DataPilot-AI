"""
DataPilot AI — Stats Panel Component
Statistical test recommender, distribution analyzer, hypothesis builder UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.modules.stats_engine import (
    recommend_statistical_test,
    analyze_distribution,
    build_hypothesis,
)


def render_stats_panel(df: pd.DataFrame):
    """Render the Statistical Intelligence Engine UI."""
    st.markdown("### 📐 Statistical Intelligence Engine")

    tab1, tab2, tab3 = st.tabs(["🔬 Test Recommender", "📊 Distribution Analyzer", "🧪 Hypothesis Builder"])

    # ── Tab 1: Statistical Test Recommender ───────────────────────────────────
    with tab1:
        st.markdown("**Auto-recommend the right statistical test for any two columns.**")
        col1, col2 = st.columns(2)
        with col1:
            col_a = st.selectbox("Variable 1:", df.columns.tolist(), key="stat_col_a")
        with col2:
            remaining = [c for c in df.columns if c != col_a]
            col_b = st.selectbox("Variable 2:", remaining, key="stat_col_b")

        if st.button("🔬 Run Statistical Test", use_container_width=True):
            with st.spinner("Running analysis..."):
                result = recommend_statistical_test(df, col_a, col_b)

            st.markdown(f"**Test Selected:** `{result.get('test', 'N/A')}`")

            # Result metrics
            r = result.get("result", {})
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Test Statistic", f"{r.get('statistic', r.get('chi2', 'N/A'))}")
            with m_col2:
                p = r.get("p_value", None)
                st.metric("P-Value", f"{p:.4f}" if p is not None else "N/A")
            with m_col3:
                sig = r.get("significant", False)
                st.metric("Significant?", "✅ Yes" if sig else "❌ No")

            # Assumptions
            assumptions = result.get("assumptions", {})
            if assumptions:
                with st.expander("📋 Assumptions Check"):
                    for k, v in assumptions.items():
                        if isinstance(v, dict):
                            st.write(f"**{k}:** {v.get('test', '')} — {'✅ Passed' if v.get('passed') else '❌ Failed'} (p={v.get('p_value', 'N/A')})")
                        else:
                            st.write(f"**{k}:** {v}")

            # Interpretation
            st.info(f"📊 **Interpretation:** {result.get('interpretation', '')}")
            if result.get("recommendation"):
                st.success(f"✅ **Recommendation:** {result['recommendation']}")

    # ── Tab 2: Distribution Analyzer ──────────────────────────────────────────
    with tab2:
        st.markdown("**Analyze the distribution of any numeric column.**")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns found.")
        else:
            dist_col = st.selectbox("Select column:", numeric_cols, key="dist_col")

            if st.button("📊 Analyze Distribution", use_container_width=True):
                with st.spinner("Analyzing..."):
                    analysis = analyze_distribution(df, dist_col)

                if "error" in analysis:
                    st.error(analysis["error"])
                else:
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("Mean", f"{analysis['mean']:.3f}")
                    d2.metric("Std Dev", f"{analysis['std']:.3f}")
                    d3.metric("Skewness", f"{analysis['skewness']:.3f}")
                    d4.metric("Kurtosis", f"{analysis['kurtosis']:.3f}")

                    st.markdown(f"- **Shape:** {analysis['skewness_label']}")
                    st.markdown(f"- **Normality ({analysis['normality']['test']}):** {'✅ Normal' if analysis['normality']['passed'] else '❌ Non-normal'} (p={analysis['normality'].get('p_value', 'N/A')})")
                    st.markdown(f"- **IQR Outliers:** {analysis['iqr_outliers']} | **Z-Score Outliers:** {analysis['zscore_outliers']}")
                    st.info(f"💡 **Transform Suggestion:** {analysis['transform_suggestion']}")

                    # Histogram + KDE
                    series = df[dist_col].dropna()
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=series, name="Distribution", nbinsx=40, opacity=0.7, 
                                               marker_color="#6366f1"))
                    fig.update_layout(title=f"Distribution: {dist_col}", xaxis_title=dist_col, yaxis_title="Count",
                                      template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)

                    # Q-Q Plot
                    from scipy import stats as scipy_stats
                    q_theory, q_data = scipy_stats.probplot(series, dist="norm")[:2]
                    qq_fig = go.Figure()
                    qq_fig.add_trace(go.Scatter(x=q_theory[0], y=q_theory[1], mode="markers", name="Data", marker_color="#10b981"))
                    qq_fig.add_trace(go.Scatter(x=[q_theory[0].min(), q_theory[0].max()],
                                                y=[q_theory[0].min() * q_data[0] + q_data[1],
                                                   q_theory[0].max() * q_data[0] + q_data[1]],
                                                mode="lines", name="Normal Line", line=dict(color="#ef4444")))
                    qq_fig.update_layout(title="Q-Q Plot (Normality Check)", template="plotly_dark", height=350)
                    st.plotly_chart(qq_fig, use_container_width=True)

    # ── Tab 3: Hypothesis Builder ─────────────────────────────────────────────
    with tab3:
        st.markdown("**No-code hypothesis testing. Select variables and relationship — we do the rest.**")

        h_col1, h_rel, h_col2 = st.columns([2, 1, 2])
        with h_col1:
            h_var1 = st.selectbox("Variable 1:", df.columns.tolist(), key="h_var1")
        with h_rel:
            relationship = st.selectbox("Relationship:", ["affects", "differs by", "associated with"], key="h_rel")
        with h_col2:
            remaining_h = [c for c in df.columns if c != h_var1]
            h_var2 = st.selectbox("Variable 2:", remaining_h, key="h_var2")

        if st.button("🧪 Build & Test Hypothesis", use_container_width=True):
            with st.spinner("Building hypothesis..."):
                result = build_hypothesis(df, h_var1, relationship, h_var2)

            st.markdown(f"**H₀:** {result['null_hypothesis']}")
            st.markdown(f"**H₁:** {result['alternative_hypothesis']}")
            st.markdown(f"**Test:** `{result['test_used']}`")

            r = result.get("result", {})
            hc1, hc2 = st.columns(2)
            with hc1:
                p = r.get("p_value")
                st.metric("P-Value", f"{p:.4f}" if p else "N/A")
            with hc2:
                significant = r.get("significant", False)
                st.metric("Decision", "Reject H₀" if significant else "Fail to Reject H₀")

            if significant:
                st.success(result["conclusion"])
            else:
                st.warning(result["conclusion"])

            st.info(f"📊 {result.get('interpretation', '')}")
            if result.get("recommendation"):
                st.markdown(f"✅ **Next Step:** {result['recommendation']}")
 
