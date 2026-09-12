import io
import os
import subprocess
import sys
import tempfile
import traceback
from typing import List, Optional, TypedDict

from flask import Flask, render_template_string, request
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph


# ============================================================
# Configuration
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
ENABLE_CODE_EXECUTION = os.getenv("ENABLE_CODE_EXECUTION", "false").lower() == "true"
EXECUTION_TIMEOUT_SECONDS = int(os.getenv("EXECUTION_TIMEOUT_SECONDS", "8"))

if not GEMINI_API_KEY:
    raise RuntimeError(
        "Missing API key. Set GEMINI_API_KEY (recommended) or GOOGLE_API_KEY."
    )

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
)

app = Flask(__name__)


# ============================================================
# State
# ============================================================

class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# Helpers
# ============================================================

def response_to_text(response) -> str:
    """Normalize Gemini/LangChain response content into a plain string."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


# ============================================================
# Tools
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """Execute generated Python code and return stdout/stderr.

    Execution is disabled by default for safety. Set ENABLE_CODE_EXECUTION=true
    only if you understand the risks of running LLM-generated code.
    """
    if not ENABLE_CODE_EXECUTION:
        return (
            "Code execution is disabled. Set ENABLE_CODE_EXECUTION=true "
            "to allow generated Python code to run."
        )

    clean_code = strip_code_fences(str(code))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "generated_solution.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(clean_code)

            completed = subprocess.run(
                [sys.executable, "-I", script_path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=EXECUTION_TIMEOUT_SECONDS,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONIOENCODING": "utf-8",
                },
            )

            output_parts = []
            if completed.stdout.strip():
                output_parts.append("STDOUT:\n" + completed.stdout.strip())
            if completed.stderr.strip():
                output_parts.append("STDERR:\n" + completed.stderr.strip())

            if not output_parts:
                output_parts.append("Success (no terminal output)")

            output_parts.append(f"Exit code: {completed.returncode}")
            return "\n\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return f"Execution Error: timed out after {EXECUTION_TIMEOUT_SECONDS} seconds."
    except Exception:
        return "Execution Error:\n" + traceback.format_exc()


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a coding task."""
    prompt = (
        "You are a Senior QA Engineer. Generate 3 to 5 highly specific test "
        f"scenarios for the following coding task:\n\n{task_description}\n\n"
        "Include normal cases, edge cases, and expected behavior. "
        "Return a numbered list only."
    )
    response = llm.invoke(prompt)
    return response_to_text(response)


# ============================================================
# LangGraph nodes
# ============================================================

def developer_node(state: CrewState):
    task = state["messages"][-1].content

    prompt = (
        "Write a clean, self-contained Python script to solve the task below.\n"
        "Return only Python code, with no markdown fences and no explanation.\n\n"
        f"Task:\n{task}"
    )

    response = llm.invoke(prompt)
    code_str = strip_code_fences(response_to_text(response))
    return {"code": code_str}


def tester_node(state: CrewState):
    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke({"task_description": task})
    execution_result = run_python_code.invoke({"code": state.get("code", "")})

    report = (
        "EXECUTION OUTPUT\n"
        "================\n"
        f"{execution_result}\n\n"
        "TEST SCENARIOS\n"
        "==============\n"
        f"{test_cases}"
    )

    return {"report": report}


workflow = StateGraph(CrewState)
workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)
workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", END)

rt_app = workflow.compile()


# ============================================================
# Web UI
# ============================================================

PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Coding Crew</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      max-width: 1050px;
      margin: 0 auto;
      padding: 32px 20px;
      background: #f5f7fb;
      color: #172033;
    }
    .card {
      background: white;
      border-radius: 14px;
      padding: 22px;
      margin-bottom: 20px;
      box-shadow: 0 4px 18px rgba(0,0,0,.07);
    }
    textarea {
      width: 100%;
      min-height: 150px;
      box-sizing: border-box;
      padding: 14px;
      border: 1px solid #cfd6e4;
      border-radius: 10px;
      font: inherit;
    }
    button {
      margin-top: 12px;
      background: #1f5eff;
      color: white;
      border: 0;
      border-radius: 9px;
      padding: 11px 18px;
      font-weight: 700;
      cursor: pointer;
    }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #101622;
      color: #e8edf7;
      padding: 16px;
      border-radius: 10px;
      overflow-x: auto;
    }
    .notice {
      padding: 12px 14px;
      border-radius: 10px;
      background: #fff4cf;
      margin-bottom: 18px;
    }
    .error {
      padding: 12px 14px;
      border-radius: 10px;
      background: #ffe0e0;
      color: #8a1616;
      margin-bottom: 18px;
    }
  </style>
</head>
<body>
  <h1>AI Coding Crew</h1>
  <p>Developer → Tester workflow powered by Gemini + LangGraph.</p>

  {% if not execution_enabled %}
    <div class="notice">
      Generated-code execution is currently disabled for safety.
      The app will still generate code and test scenarios.
    </div>
  {% endif %}

  {% if error %}
    <div class="error">{{ error }}</div>
  {% endif %}

  <div class="card">
    <form method="post">
      <label for="task"><strong>Coding task</strong></label>
      <textarea id="task" name="task" required
        placeholder="Example: Write a Python function that checks whether a string is a palindrome.">{{ task }}</textarea>
      <button type="submit">Run Crew</button>
    </form>
  </div>

  {% if code %}
    <div class="card">
      <h2>Developer Output</h2>
      <pre>{{ code }}</pre>
    </div>
  {% endif %}

  {% if report %}
    <div class="card">
      <h2>Tester Report</h2>
      <pre>{{ report }}</pre>
    </div>
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    task = ""
    code = None
    report = None
    error = None

    if request.method == "POST":
        task = request.form.get("task", "").strip()

        if not task:
            error = "Please enter a coding task."
        else:
            try:
                result = rt_app.invoke(
                    {"messages": [HumanMessage(content=task)]},
                    config={"recursion_limit": 20},
                )
                code = result.get("code")
                report = result.get("report")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

    return render_template_string(
        PAGE,
        task=task,
        code=code,
        report=report,
        error=error,
        execution_enabled=ENABLE_CODE_EXECUTION,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": GEMINI_MODEL,
        "code_execution_enabled": ENABLE_CODE_EXECUTION,
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
