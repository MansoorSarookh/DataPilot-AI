"""
DataPilot AI — Professional PDF & HTML Report Generator
Enterprise-safe, Unicode-safe, Streamlit Cloud compatible
"""

import io
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF


# ─────────────────────────────────────────────────────────────
# Professional Styled PDF Class
# ─────────────────────────────────────────────────────────────

class DataPilotPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, "DataPilot AI — Executive Report", ln=True)
        self.ln(4)

        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(
            0,
            10,
            f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | Page {self.page_no()}",
            align="C",
        )


# ─────────────────────────────────────────────────────────────
# Chart Generator (Embedded PNG)
# ─────────────────────────────────────────────────────────────

def _generate_correlation_chart(df: pd.DataFrame):
    numeric = df.select_dtypes(include="number")

    if numeric.shape[1] < 2:
        return None

    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.matshow(corr)
    fig.colorbar(cax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)

    return buf


# ─────────────────────────────────────────────────────────────
# Main PDF Generator
# ─────────────────────────────────────────────────────────────

def generate_pdf_report(
    df: pd.DataFrame,
    trust_score: dict = None,
    insights: list = None,
    file_name: str = "dataset",
    ai_summary: str = "",
):
    pdf = DataPilotPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "", 12)

    # ─────────────────────────────────────────────
    # Dataset Overview
    # ─────────────────────────────────────────────
    pdf.cell(0, 8, f"Dataset: {file_name}", ln=True)
    pdf.cell(0, 8, f"Rows: {df.shape[0]:,}", ln=True)
    pdf.cell(0, 8, f"Columns: {df.shape[1]}", ln=True)
    pdf.ln(6)

    # ─────────────────────────────────────────────
    # Trust Score Section
    # ─────────────────────────────────────────────
    if trust_score:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Dataset Quality Assessment", ln=True)
        pdf.set_font("Helvetica", "", 12)

        overall = trust_score.get("overall", 0)
        pdf.cell(0, 8, f"Overall Trust Score: {overall:.0%}", ln=True)
        pdf.ln(4)

        dimensions = trust_score.get("dimensions", {})
        for k, v in dimensions.items():
            pdf.cell(0, 6, f"{k}: {float(v):.0%}", ln=True)

        pdf.ln(6)

    # ─────────────────────────────────────────────
    # AI Executive Summary
    # ─────────────────────────────────────────────
    if ai_summary:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "AI Executive Summary", ln=True)
        pdf.set_font("Helvetica", "", 12)

        pdf.multi_cell(0, 7, ai_summary)
        pdf.ln(6)

    # ─────────────────────────────────────────────
    # Custom Insights
    # ─────────────────────────────────────────────
    if insights:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Custom Insights", ln=True)
        pdf.set_font("Helvetica", "", 12)

        for idx, insight in enumerate(insights, 1):
            pdf.multi_cell(0, 6, f"{idx}. {insight}")
            pdf.ln(2)

        pdf.ln(4)

    # ─────────────────────────────────────────────
    # Statistical Summary (Limited Safe Rows)
    # ─────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Statistical Overview", ln=True)
    pdf.set_font("Helvetica", "", 10)

    summary_df = df.describe(include="all").transpose().head(20)

    for col in summary_df.index:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, col, ln=True)
        pdf.set_font("Helvetica", "", 9)

        for stat, val in summary_df.loc[col].items():
            text = f"{stat}: {str(val)[:50]}"
            pdf.multi_cell(0, 5, text)

        pdf.ln(3)

    # ─────────────────────────────────────────────
    # Correlation Chart
    # ─────────────────────────────────────────────
    chart_buf = _generate_correlation_chart(df)
    if chart_buf:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Correlation Analysis", ln=True)
        pdf.ln(5)

        pdf.image(chart_buf, x=15, w=180)

    # ─────────────────────────────────────────────
    # Closing
    # ─────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "About DataPilot AI", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(
        0,
        7,
        "DataPilot AI is an intelligent data analysis assistant designed "
        "to help teams understand datasets, detect quality issues, and "
        "generate executive-ready insights instantly.",
    )

    # ─────────────────────────────────────────────
    # SAFE RETURN (Python 3.13 compatible)
    # ─────────────────────────────────────────────
    output = pdf.output(dest="S")

    if isinstance(output, (bytes, bytearray)):
        return bytes(output)

    return output.encode("latin-1", "ignore")


# ─────────────────────────────────────────────────────────────
# Simple HTML Generator
# ─────────────────────────────────────────────────────────────

def generate_html_report(
    df: pd.DataFrame,
    trust_score: dict = None,
    insights: list = None,
    file_name: str = "dataset",
    ai_summary: str = "",
):
    html = f"""
    <html>
    <head>
        <title>DataPilot AI Report</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            h1 {{ color: #1f2937; }}
            h2 {{ color: #374151; }}
        </style>
    </head>
    <body>
        <h1>DataPilot AI — Executive Report</h1>
        <h2>Dataset Overview</h2>
        <p><strong>Dataset:</strong> {file_name}</p>
        <p><strong>Rows:</strong> {df.shape[0]:,}</p>
        <p><strong>Columns:</strong> {df.shape[1]}</p>
    """

    if trust_score:
        html += f"<h2>Trust Score</h2><p>{trust_score.get('overall',0):.0%}</p>"

    if ai_summary:
        html += f"<h2>AI Executive Summary</h2><p>{ai_summary}</p>"

    if insights:
        html += "<h2>Custom Insights</h2><ul>"
        for ins in insights:
            html += f"<li>{ins}</li>"
        html += "</ul>"

    html += "</body></html>"
    return html 



# =================================================================================================================================
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
 
