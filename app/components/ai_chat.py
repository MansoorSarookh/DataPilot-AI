"""
DataPilot AI — AI Chat Component
Conversational data analyst interface backed by Groq LLM.
"""

import streamlit as st
import pandas as pd
from app.modules.ai_engine import chat_with_data, get_groq_client


QUICK_PROMPTS = [
    "📊 What trends exist in this dataset?",
    "🔍 Which features matter most?",
    "⚠️ Is this dataset reliable for ML?",
    "🧹 What cleaning steps do you recommend?",
    "🎯 Prepare ML-ready feature suggestions",
    "📈 Are there any anomalies or outliers?",
]


def render_ai_chat(df: pd.DataFrame):
    """Render the conversational AI chat interface."""
    st.markdown("### 🤖 AI Data Copilot")

    # Check AI availability
    client = get_groq_client()
    if client:
        st.success("✅ AI powered by **Groq LLM** (llama-3.3-70b-versatile)")
    else:
        st.warning("⚠️ Groq API key not configured — using intelligent fallback responses. Add your key to `.streamlit/secrets.toml`.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Quick prompts
    st.markdown("**Quick Actions:**")
    cols = st.columns(3)
    for i, prompt in enumerate(QUICK_PROMPTS):
        with cols[i % 3]:
            if st.button(prompt, key=f"quick_{i}", use_container_width=True):
                _process_message(prompt.split(" ", 1)[1], df)
                st.rerun()

    st.divider()

    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask anything about your data... (e.g. 'Why is revenue dropping?')")
    if user_input:
        _process_message(user_input, df)
        st.rerun()

    # Clear history button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Dataset context summary
    with st.expander("📋 Dataset Context Summary"):
        st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.write(f"**Numeric cols:** {', '.join(df.select_dtypes(include='number').columns[:8].tolist())}")
        st.write(f"**Categorical cols:** {', '.join(df.select_dtypes(include='object').columns[:8].tolist())}")
        nulls = df.isna().sum().sum()
        st.write(f"**Missing values:** {nulls:,}")


def _process_message(user_input: str, df: pd.DataFrame):
    """Add user message, get AI response, update history."""
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Get AI response
    with st.spinner("🧠 Analyzing your data..."):
        response = chat_with_data(
            question=user_input,
            df=df,
            chat_history=st.session_state.chat_history,
        )

    # Add assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": response})
 
