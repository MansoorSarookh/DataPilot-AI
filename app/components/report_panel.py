"""
DataPilot AI — Export & Report Panel Component
"""

import streamlit as st
import pandas as pd
import io

from app.modules.report_generator import generate_pdf_report, generate_html_report
from app.modules.notebook_exporter import generate_notebook
from app.modules.ai_engine import generate_executive_summary


def render_report_panel(df: pd.DataFrame, trust_score: dict, file_name: str = "dataset"):
    """Render the report generation and export UI."""

    st.markdown("### 📥 Export & Report Center")

    tab1, tab2, tab3 = st.tabs(
        ["📄 Executive Report", "📓 Jupyter Notebook", "💾 Data Export"]
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: Executive Report
    # ──────────────────────────────────────────────────────────────────────────
    with tab1:

        st.markdown("**Generate a one-click professional analysis report.**")

        include_ai = st.checkbox(
            "✨ Include AI-generated executive summary (requires Groq API)",
            value=True,
        )

        custom_insights = []

        with st.expander("📝 Add Custom Insights (Optional)"):
            for i in range(3):
                insight = st.text_input(
                    f"Insight {i+1}:",
                    key=f"insight_{i}",
                    placeholder="e.g. Q3 revenue dropped due to seasonal factors",
                )
                if insight:
                    custom_insights.append(insight)

        col1, col2 = st.columns(2)

        # ── PDF REPORT ────────────────────────────────────────────────────────
        with col1:
            if st.button(
                "📄 Generate PDF Report",
                use_container_width=True,
                type="primary",
            ):
                with st.spinner("Generating executive report..."):

                    ai_summary = ""
                    if include_ai:
                        ai_summary = generate_executive_summary(
                            df, trust_score, custom_insights
                        )

                    pdf_bytes = generate_pdf_report(
                        df=df,
                        trust_score=trust_score,
                        insights=custom_insights,
                        file_name=file_name,
                        ai_summary=ai_summary,
                    )

                if isinstance(pdf_bytes, bytes):
                    st.download_button(
                        "⬇️ Download PDF Report",
                        pdf_bytes,
                        f"datapilot_report_{file_name}.pdf",
                        "application/pdf",
                        key="dl_pdf_report",
                    )
                    st.success("✅ PDF Report ready!")
                else:
                    st.error("PDF generation failed. Try HTML format instead.")

        # ── HTML REPORT ───────────────────────────────────────────────────────
        with col2:
            if st.button("🌐 Generate HTML Report", use_container_width=True):

                with st.spinner("Generating HTML report..."):

                    ai_summary = ""
                    if include_ai:
                        ai_summary = generate_executive_summary(
                            df, trust_score, custom_insights
                        )

                    html_content = generate_html_report(
                        df=df,
                        trust_score=trust_score,
                        insights=custom_insights,
                        file_name=file_name,
                        ai_summary=ai_summary,
                    )

                st.download_button(
                    "⬇️ Download HTML Report",
                    html_content.encode("utf-8"),
                    f"datapilot_report_{file_name}.html",
                    "text/html",
                    key="dl_html_report",
                )

                st.success(
                    "✅ HTML Report ready! Open in browser for interactive charts."
                )

        # ── AI SUMMARY PREVIEW ────────────────────────────────────────────────
        if include_ai:
            if st.button("👁️ Preview AI Summary", key="preview_ai"):
                with st.spinner("Generating AI narrative..."):
                    summary = generate_executive_summary(
                        df, trust_score, custom_insights
                    )

                st.markdown("---")
                st.markdown("**AI Executive Summary:**")
                st.markdown(summary)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: Jupyter Notebook Export
    # ──────────────────────────────────────────────────────────────────────────
    with tab2:

        st.markdown(
            "**Export a fully reproducible Jupyter notebook with all your analysis steps.**"
        )

        st.info(
            "💡 The notebook includes: setup, data loading, trust score, "
            "EDA, cleaning, and ML training code."
        )

        include_ml = st.checkbox("Include ML training code", value=False)

        ml_target = None
        ml_algo = "Random Forest"

        if include_ml:
            cols = df.columns.tolist()
            ml_target = st.selectbox(
                "Target column for ML:",
                cols,
                key="nb_target",
            )

            ml_algo = st.selectbox(
                "Algorithm:",
                [
                    "Random Forest",
                    "Logistic Regression",
                    "XGBoost",
                    "Linear Regression",
                ],
                key="nb_algo",
            )

        if st.button(
            "📓 Generate Jupyter Notebook",
            use_container_width=True,
            type="primary",
        ):

            with st.spinner("Building notebook..."):

                cleaning_ops = st.session_state.get("cleaning_ops", [])

                ml_config = (
                    {"target_col": ml_target, "algorithm": ml_algo}
                    if include_ml and ml_target
                    else None
                )

                nb_json = generate_notebook(
                    file_name=file_name,
                    df=df,
                    cleaning_steps=cleaning_ops,
                    ml_config=ml_config,
                    trust_score=trust_score,
                )

            st.download_button(
                "⬇️ Download .ipynb Notebook",
                nb_json.encode("utf-8"),
                f"datapilot_analysis_{file_name.replace('.', '_')}.ipynb",
                "application/json",
                key="dl_notebook",
            )

            st.success("✅ Notebook generated! Open with Jupyter Lab or VS Code.")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: Data Export
    # ──────────────────────────────────────────────────────────────────────────
    with tab3:

        st.markdown("**Download your data in multiple formats.**")

        cleaned = st.session_state.get("cleaned_df")

        if cleaned is None:
            export_df = df
            label = "Original Dataset"
        else:
            export_df = cleaned
            label = "Cleaned Dataset"

        st.info(
            f"Exporting: **{label}** "
            f"({export_df.shape[0]:,} rows × {export_df.shape[1]} columns)"
        )

        c1, c2, c3 = st.columns(3)

        # ── CSV ───────────────────────────────────────────────────────────────
        with c1:
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 CSV",
                csv_bytes,
                f"{file_name}_export.csv",
                "text/csv",
                use_container_width=True,
                key="dl_exp_csv",
            )

        # ── EXCEL ─────────────────────────────────────────────────────────────
        with c2:
            buf = io.BytesIO()
            export_df.to_excel(buf, index=False, engine="openpyxl")
            buf.seek(0)

            st.download_button(
                "📥 Excel",
                buf.getvalue(),
                f"{file_name}_export.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_exp_xlsx",
            )

        # ── JSON ──────────────────────────────────────────────────────────────
        with c3:
            json_bytes = export_df.to_json(
                orient="records",
                indent=2,
            ).encode("utf-8")

            st.download_button(
                "📥 JSON",
                json_bytes,
                f"{file_name}_export.json",
                "application/json",
                use_container_width=True,
                key="dl_exp_json",
            )

        # ── STATISTICS EXPORT ─────────────────────────────────────────────────
        st.markdown("**📊 Statistics Export:**")

        stats_df = export_df.describe(include="all").round(4)
        stats_csv = stats_df.to_csv().encode("utf-8")

        st.download_button(
            "📥 Download Statistics CSV",
            stats_csv,
            f"{file_name}_stats.csv",
            "text/csv",
            key="dl_stats_csv",
        ) 




# """
# DataPilot AI — Export & Report Panel Component
# """

# import streamlit as st
# import pandas as pd
# import io
# from app.modules.report_generator import generate_pdf_report, generate_html_report
# from app.modules.notebook_exporter import generate_notebook
# from app.modules.ai_engine import generate_executive_summary


# def render_report_panel(df: pd.DataFrame, trust_score: dict, file_name: str = "dataset"):
#     """Render the report generation and export UI."""
#     st.markdown("### 📥 Export & Report Center")

#     tab1, tab2, tab3 = st.tabs(["📄 Executive Report", "📓 Jupyter Notebook", "💾 Data Export"])

#     # ── Tab 1: Executive Report ───────────────────────────────────────────────
#     with tab1:
#         st.markdown("**Generate a one-click professional analysis report.**")

#         include_ai = st.checkbox("✨ Include AI-generated executive summary (requires Groq API)", value=True)

#         custom_insights = []
#         with st.expander("📝 Add Custom Insights (Optional)"):
#             for i in range(3):
#                 insight = st.text_input(f"Insight {i+1}:", key=f"insight_{i}", placeholder="e.g. Q3 revenue dropped due to seasonal factors")
#                 if insight:
#                     custom_insights.append(insight)

#         col1, col2 = st.columns(2)

#         with col1:
#             if st.button("📄 Generate PDF Report", use_container_width=True, type="primary"):
#                 with st.spinner("Generating executive report..."):
#                     ai_summary = ""
#                     if include_ai:
#                         ai_summary = generate_executive_summary(df, trust_score, custom_insights)
                    
#                     pdf_bytes = generate_pdf_report(
#                         df=df,
#                         trust_score=trust_score,
#                         insights=custom_insights,
#                         file_name=file_name,
#                         ai_summary=ai_summary,
#                     )

#                 if isinstance(pdf_bytes, bytes):
#                     st.download_button(
#                         "⬇️ Download PDF Report",
#                         pdf_bytes,
#                         f"datapilot_report_{file_name}.pdf",
#                         "application/pdf",
#                         key="dl_pdf_report",
#                     )
#                     st.success("✅ PDF Report ready!")
#                 else:
#                     st.error("PDF generation failed. Try HTML format instead.")

#         with col2:
#             if st.button("🌐 Generate HTML Report", use_container_width=True):
#                 with st.spinner("Generating HTML report..."):
#                     ai_summary = ""
#                     if include_ai:
#                         ai_summary = generate_executive_summary(df, trust_score, custom_insights)

#                     html_content = generate_html_report(
#                         df=df,
#                         trust_score=trust_score,
#                         insights=custom_insights,
#                         file_name=file_name,
#                         ai_summary=ai_summary,
#                     )

#                 st.download_button(
#                     "⬇️ Download HTML Report",
#                     html_content.encode("utf-8"),
#                     f"datapilot_report_{file_name}.html",
#                     "text/html",
#                     key="dl_html_report",
#                 )
#                 st.success("✅ HTML Report ready! Open in browser for interactive charts.")

#         # Preview AI summary
#         if include_ai:
#             if st.button("👁️ Preview AI Summary", key="preview_ai"):
#                 with st.spinner("Generating AI narrative..."):
#                     summary = generate_executive_summary(df, trust_score, custom_insights)
#                 st.markdown("---")
#                 st.markdown("**AI Executive Summary:**")
#                 st.markdown(summary)

#     # ── Tab 2: Jupyter Notebook ───────────────────────────────────────────────
#     with tab2:
#         st.markdown("**Export a fully reproducible Jupyter notebook with all your analysis steps.**")
#         st.info("💡 The notebook includes: setup, data loading, trust score, EDA, cleaning, and ML training code.")

#         include_ml = st.checkbox("Include ML training code", value=False)
#         ml_target = None
#         ml_algo = "Random Forest"
#         if include_ml:
#             cols = df.columns.tolist()
#             ml_target = st.selectbox("Target column for ML:", cols, key="nb_target")
#             ml_algo = st.selectbox("Algorithm:", ["Random Forest", "Logistic Regression", "XGBoost", "Linear Regression"], key="nb_algo")

#         if st.button("📓 Generate Jupyter Notebook", use_container_width=True, type="primary"):
#             with st.spinner("Building notebook..."):
#                 cleaning_ops = st.session_state.get("cleaning_ops", [])
#                 ml_config = {"target_col": ml_target, "algorithm": ml_algo} if include_ml and ml_target else None

#                 nb_json = generate_notebook(
#                     file_name=file_name,
#                     df=df,
#                     cleaning_steps=cleaning_ops,
#                     ml_config=ml_config,
#                     trust_score=trust_score,
#                 )

#             st.download_button(
#                 "⬇️ Download .ipynb Notebook",
#                 nb_json.encode("utf-8"),
#                 f"datapilot_analysis_{file_name.replace('.', '_')}.ipynb",
#                 "application/json",
#                 key="dl_notebook",
#             )
#             st.success("✅ Notebook generated! Open with Jupyter Lab or VS Code.")

#     # ── Tab 3: Data Export ─────────────────────────────────────────────────────
#     with tab3:
#         st.markdown("**Download your data in multiple formats.**")

#         # export_df = st.session_state.get("cleaned_df", df)
        
#         # label = "Cleaned Dataset" if "cleaned_df" in st.session_state else "Original Dataset"
#         # st.info(f"Exporting: **{label}** ({export_df.shape[0]:,} rows × {export_df.shape[1]} columns)")
#         cleaned = st.session_state.get("cleaned_df")

#          # fallback if cleaned is None
#         if cleaned is None:
#             export_df = df
#             label = "Original Dataset"
#         else:
#             export_df = cleaned
#             label = "Cleaned Dataset"

#         st.info(
#             f"Exporting: **{label}** "
#             f"({export_df.shape[0]:,} rows × {export_df.shape[1]} columns)"
#         )
        

#         c1, c2, c3 = st.columns(3)

#         with c1:
#             csv_bytes = export_df.to_csv(index=False).encode("utf-8")
#             st.download_button("📥 CSV", csv_bytes, f"{file_name}_export.csv", "text/csv", use_container_width=True, key="dl_exp_csv")

#         with c2:
#             buf = io.BytesIO()
#             export_df.to_excel(buf, index=False, engine="openpyxl")
#             buf.seek(0)
#             st.download_button("📥 Excel", buf.getvalue(), f"{file_name}_export.xlsx",
#                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#                                use_container_width=True, key="dl_exp_xlsx")

#         with c3:
#             json_bytes = export_df.to_json(orient="records", indent=2).encode("utf-8")
#             st.download_button("📥 JSON", json_bytes, f"{file_name}_export.json", "application/json", use_container_width=True, key="dl_exp_json")

#         # Descriptive stats export
#         st.markdown("**📊 Statistics Export:**")
#         stats_df = export_df.describe(include="all").round(4)
#         stats_csv = stats_df.to_csv().encode("utf-8")
#         st.download_button("📥 Download Statistics CSV", stats_csv, f"{file_name}_stats.csv", "text/csv", key="dl_stats_csv")
 
