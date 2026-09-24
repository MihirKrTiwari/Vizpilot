"""
app.py
------
Streamlit frontend for the Data Analysis & Dynamic Visualization Copilot.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

import agent  # noqa: E402  (import after load_dotenv so env vars are set)

UPLOAD_DIR = Path("static/uploads")
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "static/artifacts"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

STATUS_LABELS = {
    "inspect_schema": "🔍 Inspecting schema",
    "generate_code": "🧠 Generating analysis code",
    "execute_code": "⚙️ Running analysis in sandbox",
    "self_correct": "🩹 Self-correcting error",
    "synthesize_insights": "📝 Writing executive summary",
    "error_abort": "❌ Aborted after max retries",
}

st.set_page_config(page_title="Data Copilot", page_icon="📊", layout="wide")

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "selected_table" not in st.session_state:
    st.session_state.selected_table = None
if "chat_history" not in st.session_state:
    # list of dicts: {role, content, artifact_path?, summary?}
    st.session_state.chat_history = []

# --------------------------------------------------------------------------
# Sidebar: dataset / database source
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("📁 Data & Database")
    source_type = st.radio(
        "Source Type",
        ["Upload File", "Database Connection"],
        horizontal=True,
    )

    if source_type == "Upload File":
        uploaded_file = st.file_uploader(
            "Drop a CSV, JSON, Excel, or SQLite file",
            type=["csv", "json", "xlsx", "xls", "db", "sqlite", "sqlite3"],
        )
        if uploaded_file is not None:
            save_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.dataset_path = str(save_path)
            st.session_state.dataset_name = uploaded_file.name
            st.success(f"Loaded `{uploaded_file.name}`")
    else:
        db_uri = st.text_input(
            "Database URI or SQLite path",
            placeholder="sqlite:///database.db or postgresql://...",
            help="Supported: SQLite (.db/.sqlite), PostgreSQL, MySQL",
        )
        if db_uri:
            st.session_state.dataset_path = db_uri.strip()
            st.session_state.dataset_name = db_uri.split("/")[-1] or "Connected Database"
            st.success(f"Connected to `{st.session_state.dataset_name}`")

    # If database tables are detected, offer table selection
    if st.session_state.dataset_path:
        tables = agent.get_database_tables(st.session_state.dataset_path)
        if tables:
            st.session_state.selected_table = st.selectbox(
                "🗄️ Select Active Table",
                options=tables,
                index=0 if st.session_state.selected_table not in tables else tables.index(st.session_state.selected_table),
            )
            st.caption(f"Analyzing table **`{st.session_state.selected_table}`** ({len(tables)} tables in DB)")
        else:
            st.session_state.selected_table = None

    if st.session_state.dataset_name:
        st.divider()
        st.caption(f"Active source: **{st.session_state.dataset_name}**")

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption(
        "Model: `{}`  \nMax retries: **{}**".format(
            agent.LLM_MODEL, agent.MAX_RETRIES
        )
    )

st.title("📊 Data Analysis & Dynamic Visualization Copilot")
st.caption(
    "Upload a dataset, then ask a question in plain English. "
    "The agent will inspect the data, write and run analysis code, "
    "self-correct on errors, and summarize the findings."
)

# --------------------------------------------------------------------------
# Render chat history
# --------------------------------------------------------------------------
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("artifact_path") and Path(turn["artifact_path"]).exists():
            html = Path(turn["artifact_path"]).read_text(encoding="utf-8")
            components.html(html, height=550, scrolling=True)
        if turn.get("summary"):
            st.markdown(turn["summary"])

# --------------------------------------------------------------------------
# Chat input -> run agent
# --------------------------------------------------------------------------
query = st.chat_input(
    "Ask a question about your data..."
    if st.session_state.dataset_path
    else "Upload a dataset in the sidebar first"
)

if query:
    if not st.session_state.dataset_path:
        st.warning("Please upload a dataset before asking a question.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        status_box = st.status("Starting analysis...", expanded=True)
        final_state = {}
        try:
            for node_name, node_state in agent.stream_pipeline(
                st.session_state.dataset_path,
                query,
                selected_table=st.session_state.selected_table,
            ):
                label = STATUS_LABELS.get(node_name, node_name)
                status_box.update(label=label, state="running")
                status_box.write(label)
                final_state = node_state
        except Exception as e:  # graph-level failure (e.g. bad API key)
            status_box.update(label="Pipeline error", state="error")
            st.error(f"The agent pipeline raised an unexpected error: {e}")
            st.stop()

        success = final_state.get("success", False)
        status_box.update(
            label="Done" if success else "Finished with errors",
            state="complete" if success else "error",
        )

        summary_md = final_state.get("final_summary", "_No summary produced._")
        artifact_path = final_state.get("artifact_path")

        st.markdown(summary_md)
        if artifact_path and Path(artifact_path).exists():
            html = Path(artifact_path).read_text(encoding="utf-8")
            components.html(html, height=550, scrolling=True)

        if not success:
            with st.expander("Show last error trace"):
                st.code(final_state.get("error_trace", "No trace captured."))

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": summary_md,
                "artifact_path": artifact_path,
            }
        )
