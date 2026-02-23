"""
DataPilot AI — ML Studio Component
ML readiness advisor, algorithm execution, and results dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.modules.ml_advisor import (
    assess_ml_readiness,
    suggest_feature_engineering,
    run_ml_pipeline,
    run_kmeans,
    predict_model_feasibility,
)


def render_ml_studio(df: pd.DataFrame):
    """Render the ML Studio UI."""
    st.markdown("### 🎯 ML Studio")

    tab1, tab2, tab3 = st.tabs(["🔍 Readiness Advisor", "⚗️ Train Model", "🔵 Clustering"])

    # ── Tab 1: ML Readiness ───────────────────────────────────────────────────
    with tab1:
        st.markdown("**Assess your dataset's readiness for machine learning.**")
        all_cols = df.columns.tolist()
        target_col = st.selectbox("Select Target Column (optional):", ["None"] + all_cols, key="ml_target_readiness")
        target = None if target_col == "None" else target_col

        if st.button("🔍 Assess ML Readiness", use_container_width=True):
            with st.spinner("Analyzing ML readiness..."):
                readiness = assess_ml_readiness(df, target)
                feasibility = predict_model_feasibility(df, target) if target else None

            # Readiness score meter
            score = readiness["score"]
            score_color = "#10b981" if score >= 75 else "#f59e0b" if score >= 50 else "#ef4444"
            st.markdown(f"""
            <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:12px;margin-bottom:16px;">
              <div style="font-size:52px;font-weight:900;color:{score_color};">{score}/100</div>
              <div style="color:#94a3b8;font-size:14px;">ML Readiness Score</div>
              <div style="color:white;font-size:16px;font-weight:600;margin-top:4px;">{readiness['problem_type'].upper()}</div>
            </div>
            """, unsafe_allow_html=True)

            # Checks, warnings, errors
            if readiness["checks"]:
                for c in readiness["checks"]:
                    st.success(c)
            if readiness["warnings"]:
                for w in readiness["warnings"]:
                    st.warning(w)
            if readiness["errors"]:
                for e in readiness["errors"]:
                    st.error(e)

            # Feature suggestions
            st.markdown("---")
            st.markdown("**💡 Feature Engineering Suggestions:**")
            suggestions = suggest_feature_engineering(df)
            if suggestions:
                for sug in suggestions[:6]:
                    with st.expander(f"[{sug['impact']}] {sug['type']} — `{sug['column']}`"):
                        st.write(sug["suggestion"])
                        st.code(sug.get("code_hint", ""), language="python")
            else:
                st.info("No feature engineering suggestions at this time.")

            # Feasibility
            if feasibility:
                st.markdown("---")
                st.markdown(f"**🤖 Recommended Algorithms:** {', '.join(feasibility['recommended_models'])}")
                perf = feasibility.get("expected_performance", {})
                if perf:
                    for k, v in perf.items():
                        st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")

    # ── Tab 2: Train Model ────────────────────────────────────────────────────
    with tab2:
        st.markdown("**Train an ML model directly in your browser.**")
        
        all_cols = df.columns.tolist()
        target_col = st.selectbox("Target Column:", all_cols, key="ml_target_train")

        # Auto-detect problem type
        if target_col:
            target_data = df[target_col].dropna()
            is_class = not pd.api.types.is_numeric_dtype(target_data) or target_data.nunique() <= 20
            problem_type = "Classification" if is_class else "Regression"
            st.info(f"🔍 Detected Problem Type: **{problem_type}** ({target_data.nunique()} unique values)")

        algo_options = {
            "Classification": ["Random Forest", "Logistic Regression", "XGBoost"],
            "Regression": ["Random Forest", "Linear Regression", "XGBoost"],
        }
        algo = st.selectbox("Algorithm:", algo_options.get(problem_type, ["Random Forest"]), key="ml_algo")
        test_size = st.slider("Test Set Size:", 0.1, 0.4, 0.2, 0.05, key="ml_test_size")

        if st.button("🚀 Train Model", use_container_width=True, type="primary"):
            with st.spinner(f"Training {algo}... Please wait ⏳"):
                try:
                    result = run_ml_pipeline(df, target_col, algorithm=algo, test_size=test_size)
                    st.session_state["ml_result"] = result
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    result = None

            if result:
                st.success("✅ Model trained successfully!")

                # Metrics
                metrics = result.get("metrics", {})
                metric_cols = st.columns(len(metrics))
                for i, (k, v) in enumerate(metrics.items()):
                    metric_cols[i].metric(k.replace("_", " ").upper(), f"{v:.4f}")

                # Cross-validation
                cv = result.get("cv_scores", [])
                if cv:
                    st.markdown(f"**5-Fold CV Score:** {result['cv_mean']:.4f} ± {np.std(cv):.4f}")

                # Feature importance
                fi = result.get("feature_importance", {})
                if fi:
                    st.markdown("**Top Feature Importances:**")
                    fi_df = pd.DataFrame.from_dict(fi, orient="index", columns=["Importance"]).sort_values("Importance", ascending=True).tail(15)
                    fig = px.bar(fi_df, x="Importance", y=fi_df.index, orientation="h",
                                 title="Feature Importance", color="Importance",
                                 color_continuous_scale="viridis")
                    fig.update_layout(template="plotly_dark", height=400, yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)

                # Download predictions
                predictions = result.get("predictions", [])
                if predictions:
                    pred_df = pd.DataFrame({"prediction": predictions})
                    csv_bytes = pred_df.to_csv(index=False).encode()
                    st.download_button(
                        "📥 Download Predictions (CSV)",
                        csv_bytes,
                        "datapilot_predictions.csv",
                        "text/csv",
                        key="dl_predictions",
                    )

    # ── Tab 3: Clustering ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("**Discover natural groupings (K-Means clustering).**")
        n_clusters = st.slider("Number of clusters:", 2, 10, 3, key="km_n")

        if st.button("🔵 Run K-Means Clustering", use_container_width=True):
            with st.spinner("Clustering..."):
                result = run_kmeans(df, n_clusters=n_clusters)

            if "error" in result:
                st.error(result["error"])
            else:
                sil = result.get("silhouette_score", 0)
                km_c1, km_c2 = st.columns(2)
                km_c1.metric("Silhouette Score", f"{sil:.4f}", help="Closer to 1.0 = better clusters")
                km_c2.metric("Inertia", f"{result['inertia']:.2f}")

                st.markdown("**Cluster Sizes:**")
                cluster_df = pd.DataFrame.from_dict(result["cluster_counts"], orient="index", columns=["Count"]).reset_index()
                cluster_df.columns = ["Cluster", "Count"]
                fig = px.bar(cluster_df, x="Cluster", y="Count", title="Cluster Distribution",
                             color="Count", color_continuous_scale="viridis")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                # Add cluster labels to df and offer download
                labels = result["labels"]
                if len(labels) == len(df):
                    df_clustered = df.copy()
                    df_clustered["cluster"] = labels
                    csv_bytes = df_clustered.to_csv(index=False).encode()
                    st.download_button(
                        "📥 Download Clustered Dataset",
                        csv_bytes,
                        "datapilot_clustered.csv",
                        "text/csv",
                        key="dl_clustered",
                    )
 
