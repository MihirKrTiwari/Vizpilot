"""
agent.py
--------
LangGraph state machine for the Data Analysis & Dynamic Visualization
Copilot.

Graph shape:

    inspect_schema -> generate_code -> execute_code --(success)--> synthesize_insights -> END
                                              |
                                              +--(error, retries left)--> self_correct -> execute_code (loop)
                                              |
                                              +--(error, retries exhausted)--> error_abort -> END
"""

from __future__ import annotations

import os
import re
import json
import traceback
from pathlib import Path
from typing import Optional, TypedDict

import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import sandbox

MAX_RETRIES = 3
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "static/artifacts")

LLM_MODEL = os.environ.get("COPILOT_LLM_MODEL", "gemini-3.6-flash")


# --------------------------------------------------------------------------
# State schema
# --------------------------------------------------------------------------
class CopilotState(TypedDict, total=False):
    dataset_path: str
    user_query: str
    schema_info: str
    generated_code: str
    execution_output: dict
    error_trace: str
    retry_count: int
    final_summary: str
    status: str          # human-readable status for the UI ("Inspecting Schema", ...)
    artifact_path: Optional[str]
    artifact_png_path: Optional[str]
    success: bool


def _get_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini API key from environment, Streamlit secrets, or local api_key file."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass
    if not api_key:
        api_key_file = Path("api_key")
        if api_key_file.exists():
            api_key = api_key_file.read_text(encoding="utf-8").strip()
    return api_key


def _extract_text(content: object) -> str:
    """Normalize LLM response content into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "\n".join(text_parts)
    return str(content)


def get_llm(temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    api_key = _get_gemini_api_key()
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=temperature,
        google_api_key=api_key,
        max_tokens=2048,
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _load_dataframe(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".json"):
        return pd.read_json(path)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if lower.endswith((".db", ".sqlite", ".sqlite3")):
        import sqlite3

        conn = sqlite3.connect(path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        if tables.empty:
            raise ValueError("SQLite file has no tables")
        table_name = tables.iloc[0]["name"]
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)
    raise ValueError(f"Unsupported dataset type: {path}")


def _extract_code_block(llm_text: object) -> str:
    """Pull the first ```python ... ``` fenced block out of an LLM response.
    Falls back to the raw text if no fence is found."""
    text = _extract_text(llm_text)
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# --------------------------------------------------------------------------
# Node: inspect_schema
# --------------------------------------------------------------------------
def inspect_schema(state: CopilotState) -> CopilotState:
    df = _load_dataframe(state["dataset_path"])

    buf_dtypes = df.dtypes.astype(str).to_dict()
    null_counts = df.isnull().sum().to_dict()
    describe_numeric = df.describe(include="number").to_string() if not df.select_dtypes("number").empty else "N/A"
    head = df.head(5).to_string()

    schema_info = (
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
        f"Columns & dtypes:\n{json.dumps(buf_dtypes, indent=2)}\n\n"
        f"Null counts per column:\n{json.dumps(null_counts, indent=2)}\n\n"
        f"Numeric summary (df.describe()):\n{describe_numeric}\n\n"
        f"Head (first 5 rows):\n{head}"
    )

    return {**state, "schema_info": schema_info, "status": "Inspecting Schema", "retry_count": 0}


# --------------------------------------------------------------------------
# Node: generate_code
# --------------------------------------------------------------------------
CODEGEN_SYSTEM_PROMPT = """You are a senior data analyst who writes precise, defensive Pandas + Plotly code.

Rules you MUST follow:
- A dataframe named `df` is already loaded. Do NOT re-read the source file.
- Do not import pandas/numpy/plotly yourself; `pd`, `np`, `px`, and `go` are already available.
- Your final chart MUST be assigned to a variable named exactly `fig` (a Plotly figure).
- Never write to disk, access the network, or call subprocess/os.system.
- Handle missing values and dtype mismatches defensively (e.g. coerce, dropna where sensible).
- Keep the code self-contained and runnable top to bottom with no undefined names.
- Output ONLY a single ```python fenced code block, nothing else.
"""


def generate_code(state: CopilotState) -> CopilotState:
    llm = get_llm()
    user_prompt = (
        f"Dataset schema:\n{state['schema_info']}\n\n"
        f"User's analytical question:\n{state['user_query']}\n\n"
        "Write the Pandas + Plotly code to answer this."
    )
    response = llm.invoke(
        [SystemMessage(content=CODEGEN_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    )
    code = _extract_code_block(response.content)
    return {**state, "generated_code": code, "status": "Generating Analysis Code"}


# --------------------------------------------------------------------------
# Node: execute_code
# --------------------------------------------------------------------------
def execute_code(state: CopilotState) -> CopilotState:
    result = sandbox.run_generated_code(
        code=state["generated_code"],
        dataset_path=state["dataset_path"],
        artifacts_dir=ARTIFACTS_DIR,
    )

    execution_output = {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": result.timed_out,
        "blocked_reason": result.blocked_reason,
    }

    return {
        **state,
        "execution_output": execution_output,
        "error_trace": result.error_trace,
        "artifact_path": result.artifact_path,
        "artifact_png_path": result.artifact_png_path,
        "success": result.success,
        "status": "Running Analysis" if result.success else "Execution Failed",
    }


def route_after_execution(state: CopilotState) -> str:
    if state.get("success"):
        return "synthesize_insights"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "self_correct"
    return "error_abort"


# --------------------------------------------------------------------------
# Node: self_correct
# --------------------------------------------------------------------------
SELF_CORRECT_SYSTEM_PROMPT = """You are a senior data analyst debugging your own failed code.
You will be given the dataset schema, the user's question, the code you wrote, and the
error it produced. Fix the code so it runs successfully and still answers the question.

Rules:
- A dataframe named `df` is already loaded; `pd`, `np`, `px`, `go` are already available.
- Your final chart MUST be assigned to a variable named exactly `fig`.
- Do not write to disk, access the network, or call subprocess/os.system.
- Output ONLY a single ```python fenced code block with the CORRECTED, full script -- not a diff.
"""


def self_correct(state: CopilotState) -> CopilotState:
    llm = get_llm()
    user_prompt = (
        f"Dataset schema:\n{state['schema_info']}\n\n"
        f"User's analytical question:\n{state['user_query']}\n\n"
        f"Code that failed:\n```python\n{state['generated_code']}\n```\n\n"
        f"Error / traceback:\n{state['error_trace']}\n\n"
        "Provide the corrected, complete script."
    )
    response = llm.invoke(
        [SystemMessage(content=SELF_CORRECT_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    )
    code = _extract_code_block(response.content)
    return {
        **state,
        "generated_code": code,
        "retry_count": state.get("retry_count", 0) + 1,
        "status": f"Self-Correcting Error (attempt {state.get('retry_count', 0) + 1}/{MAX_RETRIES})",
    }


# --------------------------------------------------------------------------
# Node: synthesize_insights
# --------------------------------------------------------------------------
INSIGHTS_SYSTEM_PROMPT = """You are a data analytics lead writing a crisp executive summary.
Given the schema, the user's question, the code that was run, and its stdout output,
write a markdown report with:
1. **Key Finding** -- one or two sentence direct answer to the user's question.
2. **Supporting Details** -- bullet points citing concrete numbers from the output.
3. **Caveats** -- data quality notes (nulls, small sample size, outliers) if relevant.
Be concise. Do not restate the code. Do not invent numbers that aren't grounded in the output.
"""


def synthesize_insights(state: CopilotState) -> CopilotState:
    llm = get_llm(temperature=0.2)
    user_prompt = (
        f"Dataset schema:\n{state['schema_info']}\n\n"
        f"User's question:\n{state['user_query']}\n\n"
        f"Code executed:\n```python\n{state['generated_code']}\n```\n\n"
        f"Captured stdout:\n{state['execution_output'].get('stdout', '')}\n\n"
        "Write the executive summary now."
    )
    response = llm.invoke(
        [SystemMessage(content=INSIGHTS_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    )
    summary_text = _extract_text(response.content)
    return {**state, "final_summary": summary_text, "status": "Complete"}


# --------------------------------------------------------------------------
# Node: error_abort
# --------------------------------------------------------------------------
def error_abort(state: CopilotState) -> CopilotState:
    summary = (
        "### Analysis Failed\n\n"
        f"The agent attempted **{MAX_RETRIES}** self-correction cycles and could not "
        "produce working code for this query.\n\n"
        f"**Last error:**\n```\n{state.get('error_trace', 'Unknown error')}\n```\n\n"
        "Try rephrasing the question, or check that the requested columns exist in the dataset."
    )
    return {**state, "final_summary": summary, "status": "Failed"}


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(CopilotState)

    graph.add_node("inspect_schema", inspect_schema)
    graph.add_node("generate_code", generate_code)
    graph.add_node("execute_code", execute_code)
    graph.add_node("self_correct", self_correct)
    graph.add_node("synthesize_insights", synthesize_insights)
    graph.add_node("error_abort", error_abort)

    graph.set_entry_point("inspect_schema")
    graph.add_edge("inspect_schema", "generate_code")
    graph.add_edge("generate_code", "execute_code")

    graph.add_conditional_edges(
        "execute_code",
        route_after_execution,
        {
            "synthesize_insights": "synthesize_insights",
            "self_correct": "self_correct",
            "error_abort": "error_abort",
        },
    )

    graph.add_edge("self_correct", "execute_code")
    graph.add_edge("synthesize_insights", END)
    graph.add_edge("error_abort", END)

    return graph.compile()


def run_pipeline(dataset_path: str, user_query: str) -> CopilotState:
    """Convenience entrypoint for non-streaming (script / CLI) usage."""
    app = build_graph()
    initial_state: CopilotState = {
        "dataset_path": dataset_path,
        "user_query": user_query,
        "retry_count": 0,
    }
    return app.invoke(initial_state)


def stream_pipeline(dataset_path: str, user_query: str):
    """Generator used by the Streamlit UI to show live node-by-node status.
    Yields (node_name, state_after_node) tuples as the graph executes."""
    app = build_graph()
    initial_state: CopilotState = {
        "dataset_path": dataset_path,
        "user_query": user_query,
        "retry_count": 0,
    }
    for event in app.stream(initial_state):
        for node_name, node_state in event.items():
            yield node_name, node_state


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python agent.py <dataset_path> <user_query>")
        sys.exit(1)

    final_state = run_pipeline(sys.argv[1], sys.argv[2])
    print("\n=== FINAL SUMMARY ===\n")
    print(final_state.get("final_summary"))
    print("\nArtifact:", final_state.get("artifact_path"))
