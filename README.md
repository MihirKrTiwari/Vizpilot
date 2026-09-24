# Data Analysis & Dynamic Visualization Copilot

An autonomous agent that takes a raw dataset (CSV / JSON / Excel / SQLite) and a natural
language question, then:

1. Inspects the schema (dtypes, nulls, `describe()`, head rows).
2. Writes Pandas + Plotly code to answer the question.
3. Executes that code in an isolated subprocess sandbox.
4. On failure, feeds the traceback back to the LLM and retries (up to 3 times).
5. On success, produces an executive-summary markdown report alongside the interactive chart.

## Architecture & Repository Structure

```
.
├── .streamlit/
│   ├── config.toml           # Sleek dark theme & server upload limits
│   └── secrets.toml.example  # Template for API keys on Streamlit Cloud
├── static/
│   ├── uploads/              # Uploaded user datasets (.gitkeep tracked)
│   └── artifacts/            # Generated chart HTML/PNG artifacts (.gitkeep tracked)
├── agent.py                  # LangGraph state machine (nodes, prompts, routing)
├── sandbox.py                # Isolated subprocess execution + security guards
├── app.py                    # Streamlit frontend application
├── requirements.txt          # Python dependencies
├── .env.example              # Local environment variables template
├── .gitignore                # Production git ignore rules
└── README.md
```

### Graph flow

```
inspect_schema -> generate_code -> execute_code --(success)--> synthesize_insights -> END
                                          |
                                          +--(error, retries < 3)--> self_correct -> execute_code
                                          |
                                          +--(error, retries >= 3)--> error_abort -> END
```

---

## 🌐 Deploy to Web via Streamlit Community Cloud (Free)

Because this application requires a Python runtime (LangGraph, Pandas, and sandboxed code execution), it can be deployed for free with 1 click directly connected to your GitHub repository:

1. **Push this repository to GitHub**:
   ```bash
   git add .
   git commit -m "Configure repo for hosting"
   git push origin main
   ```
2. **Go to [share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
3. **Click "New app"**:
   - **Repository:** `YourUsername/your-repo-name`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **Add your Gemini API Key in Streamlit Secrets**:
   - In your app dashboard, go to **Settings** > **Secrets**.
   - Paste:
     ```toml
     GEMINI_API_KEY = "your_actual_gemini_api_key_here"
     ```
   - Click **Save**. Your app will automatically build and go live at `https://<your-app-name>.streamlit.app`!

---

## 💻 Local Setup & Development

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY
```

> The agent defaults to Google Gemini (`ChatGoogleGenerativeAI` with `gemini-3.6-flash` or `gemini-2.5-flash`). To use OpenAI or Anthropic instead, swap `get_llm()` in [agent.py](file:///Users/mihirkumartiwari/Documents/AI_ML_Lab/agent.py) and uncomment the relevant SDK in [requirements.txt](file:///Users/mihirkumartiwari/Documents/AI_ML_Lab/requirements.txt).

## 2. Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

1. Drop a `.csv`, `.json`, `.xlsx`, or `.sqlite` file in the sidebar.
2. Ask a question in the chat box, e.g. *"What's the correlation between price and
   rating, broken down by category?"*
3. Watch the live status trace (Inspecting Schema → Generating Code → Running Analysis →
   ...) and get back an interactive chart + markdown summary.

## 3. CLI usage (no UI)

```bash
python agent.py path/to/data.csv "What are the top 5 categories by revenue?"
```

This runs the graph once and prints the final summary plus the artifact path.

## Safety notes

- Generated code runs in a **separate subprocess** (`python -I`), not in-process — a
  crash or infinite loop in generated code can't take down the app.
- Hard **30s timeout** and a **512MB memory cap** (POSIX `RLIMIT_AS`) are enforced per run.
- A static denylist blocks obviously destructive patterns (`os.system`, `subprocess`,
  `shutil.rmtree`, file writes, `eval`/`exec`, network sockets, etc.) before the code is
  even executed.
- Chart artifacts are only ever written to `static/artifacts/` — the LLM never chooses
  the output path.
- This is defense-in-depth, not a hardened multi-tenant sandbox (no seccomp/cgroups/
  containers). For a public-facing deployment, run the subprocess inside a container
  (e.g. gVisor, Firecracker, or a locked-down Docker container with `--network=none`
  and a read-only filesystem) rather than relying on the subprocess alone.

## Extending

- **Streaming code generation**: swap `llm.invoke(...)` for `llm.stream(...)` in
  `agent.py` and pipe tokens to the Streamlit status box for a more "live typing" feel.
- **Multi-turn context**: the current graph treats each question independently (fresh
  schema inspection each time). To support follow-ups like "now break that down by
  region", pass the last 1–2 turns' `user_query` + `generated_code` into the
  `generate_code` prompt.
- **More retries / different backoff**: adjust `MAX_RETRIES` in `agent.py`.
- **Different LLM provider**: `agent.get_llm()` is the single seam to swap models.
