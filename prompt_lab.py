"""Prompt Engineering Lab.

Interactive Streamlit demo of how prompt / context / harness engineering
choices affect LLM task performance. Given a user-provided task, runs
multiple strategies in parallel against the OpenAI API and displays their
results side-by-side.

Setup:
    python3 -m pip install streamlit anthropic python-dotenv
    echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env

Run:
    python3 -m streamlit run prompt_lab.py
"""

import concurrent.futures as cf
import os
import re
import time
from collections import Counter

import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024


def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


# ─── Ready-made tasks ────────────────────────────────────────────────────────
EXAMPLE_TASKS = {
    "Easy — proportional reasoning": {
        "task": "A store sells 3 apples for $2. How much do 12 apples cost?",
        "answer": "8",
    },
    "Medium — average speed": {
        "task": ("A car travels 240 miles in 4 hours, then another 90 miles "
                 "in 1.5 hours. What was its average speed for the whole trip, "
                 "in miles per hour?"),
        "answer": "60",
    },
    "Tricky — age word problem": {
        "task": ("Sarah is 5 years older than her brother. In 10 years, "
                 "she will be twice her brother's CURRENT age. "
                 "How old is Sarah now?"),
        "answer": "15",
    },
    "Tricky — sum and difference": {
        "task": ("The sum of two positive numbers is 30 and their difference "
                 "is 4. What is the larger number?"),
        "answer": "17",
    },
}


# ─── Few-shot examples ───────────────────────────────────────────────────────
FEW_SHOT_CORRECT = [
    ("A bakery makes 24 loaves. If they sell 5 in the morning and 8 in the "
     "afternoon, how many are left?",
     "Start: 24. Sold: 5 + 8 = 13. Remaining: 24 - 13 = 11. Answer: 11"),
    ("If 5 workers can finish a job in 6 days, how many worker-days did it take?",
     "5 workers × 6 days = 30 worker-days. Answer: 30"),
    ("A shirt costs $40 before a 15% tax. What's the total price?",
     "Tax: 40 × 0.15 = 6. Total: 40 + 6 = 46. Answer: 46"),
]

# Deliberately WRONG few-shot examples — demonstrates that context can
# hurt as well as help. The model tends to imitate the pattern.
FEW_SHOT_WRONG = [
    ("A bakery makes 24 loaves. If they sell 5 in the morning and 8 in the "
     "afternoon, how many are left?",
     "24 + 5 + 8 = 37. Answer: 37"),
    ("If 5 workers can finish a job in 6 days, how many worker-days did it take?",
     "5 + 6 = 11. Answer: 11"),
    ("A shirt costs $40 before a 15% tax. What's the total price?",
     "40 × 15 = 600. Answer: 600"),
]


# ─── Strategy tables ─────────────────────────────────────────────────────────
PROMPT_STRATEGIES = {
    "Zero-shot (bare)": {
        "description": "Just the task. No system prompt, no formatting hint.",
        "system": None,
        "suffix": "",
    },
    "Zero-shot + format": {
        "description": "Task plus a short formatting instruction.",
        "system": "Answer the math question. End with 'Answer: <number>'.",
        "suffix": "",
    },
    "Chain-of-thought": {
        "description": "Explicitly ask the model to reason step by step.",
        "system": ("Solve the problem step by step. Show your reasoning, "
                   "then end with 'Answer: <number>'."),
        "suffix": "\n\nLet's think step by step.",
    },
    "Role prompt": {
        "description": "Give the model a persona to adopt.",
        "system": ("You are a meticulous math tutor who explains each step "
                   "of your reasoning clearly for a student. After the "
                   "explanation, end with 'Answer: <number>'."),
        "suffix": "",
    },
}

CONTEXT_STRATEGIES = {
    "No examples (0-shot)": {
        "description": "No worked examples in context.",
        "examples": [],
    },
    "1 correct example": {
        "description": "One correct worked example before the task.",
        "examples": FEW_SHOT_CORRECT[:1],
    },
    "3 correct examples": {
        "description": "Three correct worked examples.",
        "examples": FEW_SHOT_CORRECT,
    },
    "3 WRONG examples (poisoned)": {
        "description": ("Three deliberately wrong worked examples — shows "
                        "the model often imitates the pattern, right or wrong."),
        "examples": FEW_SHOT_WRONG,
    },
}

HARNESS_STRATEGIES = {
    "Single call · T=0.7": {
        "description": "One API call at temperature 0.7 (sampled).",
    },
    "Single call · T=0.0": {
        "description": "One API call at temperature 0 (greedy).",
    },
    "Self-consistency · 5 samples, majority vote": {
        "description": ("5 samples at T=0.7. Extract an answer from each; "
                        "the majority wins."),
    },
    "Verify-and-retry": {
        "description": ("Solve once, then ask the model to double-check "
                        "its own answer and correct if needed."),
    },
}


# ─── Message construction ────────────────────────────────────────────────────
def build_messages_prompt(task: str, key: str):
    s = PROMPT_STRATEGIES[key]
    messages = []
    if s["system"]:
        messages.append({"role": "system", "content": s["system"]})
    messages.append({"role": "user", "content": task + s["suffix"]})
    return messages


def build_messages_context(task: str, key: str):
    s = CONTEXT_STRATEGIES[key]
    messages = [{
        "role": "system",
        "content": ("Solve the problem step by step. Show your reasoning, "
                    "then end with 'Answer: <number>'."),
    }]
    for q, a in s["examples"]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": task})
    return messages


def build_messages_harness_base(task: str):
    return [
        {"role": "system",
         "content": ("Solve the problem step by step. Show your reasoning, "
                     "then end with 'Answer: <number>'.")},
        {"role": "user", "content": task},
    ]


# ─── Answer parsing ──────────────────────────────────────────────────────────
def extract_answer(text: str):
    m = re.search(r"Answer\s*[:=]?\s*\**\s*([-+]?\d*\.?\d+)",
                  text, re.IGNORECASE)
    if m:
        return m.group(1).rstrip(".")
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    return nums[-1] if nums else None


def answers_match(extracted, expected):
    if not expected:
        return None
    if extracted is None:
        return False
    try:
        return abs(float(extracted) - float(expected)) < 1e-6
    except ValueError:
        return extracted.strip().lower() == expected.strip().lower()


# ─── API call ────────────────────────────────────────────────────────────────
def call_once(client, messages, temperature=0.7):
    """Adapter for the Anthropic API.

    Anthropic takes the system prompt as a top-level `system` argument,
    not as a message with role='system'. Split it out here.
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    convo = [m for m in messages if m["role"] != "system"]
    kwargs = dict(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=convo,
        temperature=temperature,
    )
    if system_parts:
        kwargs["system"] = "\n\n".join(system_parts)
    r = client.messages.create(**kwargs)
    return {
        "text": r.content[0].text,
        "input_tokens": r.usage.input_tokens,
        "output_tokens": r.usage.output_tokens,
    }


# ─── Per-experiment runners ──────────────────────────────────────────────────
def run_prompt(client, task, key):
    messages = build_messages_prompt(task, key)
    r = call_once(client, messages, temperature=0.3)
    return {**r, "answer": extract_answer(r["text"]), "messages": messages}


def run_context(client, task, key):
    messages = build_messages_context(task, key)
    r = call_once(client, messages, temperature=0.3)
    return {**r, "answer": extract_answer(r["text"]), "messages": messages}


def run_harness(client, task, key):
    base = build_messages_harness_base(task)

    if key == "Single call · T=0.7":
        r = call_once(client, base, temperature=0.7)
        return {**r, "answer": extract_answer(r["text"]), "messages": base}

    if key == "Single call · T=0.0":
        r = call_once(client, base, temperature=0.0)
        return {**r, "answer": extract_answer(r["text"]), "messages": base}

    if key.startswith("Self-consistency"):
        N = 5
        with cf.ThreadPoolExecutor(max_workers=N) as ex:
            rs = list(ex.map(lambda _: call_once(client, base, 0.7), range(N)))
        ans_list = [extract_answer(r["text"]) for r in rs]
        non_null = [a for a in ans_list if a is not None]
        winner, votes = (Counter(non_null).most_common(1)[0]
                         if non_null else (None, 0))
        summary = (
            f"**Majority vote: `{winner}` ({votes}/{N} samples agreed)**\n\n"
            f"Individual samples:\n\n" +
            "\n\n---\n\n".join(
                f"**Sample {i+1}** — extracted `{ans_list[i]}`\n\n{r['text']}"
                for i, r in enumerate(rs))
        )
        return {
            "text": summary,
            "answer": winner,
            "input_tokens": sum(r["input_tokens"] for r in rs),
            "output_tokens": sum(r["output_tokens"] for r in rs),
            "messages": base,
        }

    if key == "Verify-and-retry":
        r1 = call_once(client, base, temperature=0.3)
        a1 = extract_answer(r1["text"])
        verify = base + [
            {"role": "assistant", "content": r1["text"]},
            {"role": "user",
             "content": ("Please double-check your answer. Re-solve the "
                         "problem carefully. If your first answer was wrong, "
                         "correct it. End with 'Answer: <number>'.")},
        ]
        r2 = call_once(client, verify, temperature=0.0)
        a2 = extract_answer(r2["text"])
        summary = (
            f"**First attempt** — extracted `{a1}`:\n\n{r1['text']}\n\n"
            f"---\n\n"
            f"**Verification pass** — final `{a2}`:\n\n{r2['text']}"
        )
        return {
            "text": summary,
            "answer": a2,
            "input_tokens": r1["input_tokens"] + r2["input_tokens"],
            "output_tokens": r1["output_tokens"] + r2["output_tokens"],
            "messages": base,
        }


# ─── Parallel runner ─────────────────────────────────────────────────────────
def run_strategies_in_parallel(runner, client, task, strategy_keys):
    def timed(key):
        start = time.time()
        try:
            r = runner(client, task, key)
        except Exception as e:  # noqa: BLE001
            return key, {"error": str(e), "latency": time.time() - start}
        r["latency"] = time.time() - start
        return key, r

    results = {}
    with cf.ThreadPoolExecutor(max_workers=len(strategy_keys)) as ex:
        for key, r in ex.map(timed, strategy_keys):
            results[key] = r
    return results


# ─── Rendering ───────────────────────────────────────────────────────────────
def render_result_card(strategy_key, result, strategy_dict, expected_answer):
    with st.container(border=True):
        st.markdown(f"**{strategy_key}**")
        st.caption(strategy_dict[strategy_key]["description"])

        if "error" in result:
            st.error(f"API error: {result['error']}")
            return

        answer = result.get("answer") or "—"
        verdict = answers_match(answer, expected_answer)
        if verdict is True:
            st.success(f"### Answer: **{answer}** ✓")
        elif verdict is False:
            st.error(f"### Answer: **{answer}** ✗ "
                     f"(expected `{expected_answer}`)")
        else:
            st.info(f"### Answer: **{answer}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("In tokens", str(result.get("input_tokens", "—")))
        c2.metric("Out tokens", str(result.get("output_tokens", "—")))
        c3.metric("Latency", f"{result.get('latency', 0):.1f}s")

        with st.expander("See full response"):
            st.markdown(result.get("text", ""))
        with st.expander("See messages sent to model"):
            for m in result.get("messages", []):
                st.markdown(f"**{m['role']}:**")
                st.text(m["content"])


def render_results(results, expected_answer, strategy_dict):
    if not results:
        return
    n_correct = sum(1 for r in results.values()
                    if answers_match(r.get("answer"), expected_answer) is True)
    if expected_answer:
        st.markdown(
            f"**{n_correct} / {len(results)} strategies got the correct answer.**"
        )
    cols = st.columns(2)
    for i, (key, result) in enumerate(results.items()):
        with cols[i % 2]:
            render_result_card(key, result, strategy_dict, expected_answer)


def run_tab(tab_key, runner, strategies_dict, client, task, expected_answer,
            intro_text):
    st.markdown(intro_text)
    selected = st.multiselect(
        "Strategies to compare",
        options=list(strategies_dict.keys()),
        default=list(strategies_dict.keys()),
        key=f"{tab_key}_multi",
    )
    if st.button("▶ Run", type="primary", key=f"{tab_key}_run"):
        if not selected:
            st.warning("Select at least one strategy.")
        elif not task.strip():
            st.warning("Enter a task above first.")
        else:
            with st.spinner(f"Running {len(selected)} strategies in parallel…"):
                st.session_state[f"{tab_key}_results"] = \
                    run_strategies_in_parallel(runner, client, task, selected)
                st.session_state[f"{tab_key}_expected"] = expected_answer
    if f"{tab_key}_results" in st.session_state:
        render_results(
            st.session_state[f"{tab_key}_results"],
            st.session_state.get(f"{tab_key}_expected", ""),
            strategies_dict,
        )


def main():
    st.set_page_config(page_title="Prompt Engineering Lab",
                       page_icon="🧪", layout="wide")
    st.title("🧪 Prompt Engineering Lab")
    st.caption(
        "See how prompt, context, and harness engineering change what the "
        "model produces on the same task."
    )

    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        st.error(
            "`ANTHROPIC_API_KEY` is not set. Put it in your `.env` "
            "(`ANTHROPIC_API_KEY=sk-ant-...`) or Streamlit Cloud Secrets, "
            "then rerun."
        )
        st.stop()
    client = Anthropic(api_key=api_key)

    # ── Task input ──────────────────────────────────────────────────────────
    st.subheader("Task")
    example_key = st.selectbox(
        "Load an example (or pick custom)",
        options=["— custom —"] + list(EXAMPLE_TASKS.keys()),
    )
    if example_key == "— custom —":
        default_task, default_answer = "", ""
    else:
        default_task = EXAMPLE_TASKS[example_key]["task"]
        default_answer = EXAMPLE_TASKS[example_key]["answer"]

    task = st.text_area(
        "Task", value=default_task, height=100, key=f"task_{example_key}",
    )
    expected_answer = st.text_input(
        "Expected answer  (optional — used to grade responses)",
        value=default_answer, key=f"answer_{example_key}",
    )
    st.caption(
        f"Model: `{MODEL}` (Anthropic)  ·  each Run kicks off the "
        f"selected strategies in parallel."
    )

    # ── Three tabs, one per engineering dimension ───────────────────────────
    tab_p, tab_c, tab_h = st.tabs([
        "🖋  Prompt engineering",
        "📚  Context engineering",
        "🔧  Harness engineering",
    ])

    with tab_p:
        run_tab(
            "prompt", run_prompt, PROMPT_STRATEGIES, client, task,
            expected_answer,
            intro_text=(
                "**Varying:** the instruction and framing sent to the model.  \n"
                "**Held constant:** no few-shot examples, single API call at "
                "temperature 0.3."
            ),
        )

    with tab_c:
        run_tab(
            "context", run_context, CONTEXT_STRATEGIES, client, task,
            expected_answer,
            intro_text=(
                "**Varying:** worked examples included in the context "
                "before the task.  \n"
                "**Held constant:** step-by-step system prompt, single "
                "API call at temperature 0.3."
            ),
        )

    with tab_h:
        run_tab(
            "harness", run_harness, HARNESS_STRATEGIES, client, task,
            expected_answer,
            intro_text=(
                "**Varying:** the surrounding system — sampling temperature, "
                "number of samples, verification loops.  \n"
                "**Held constant:** the same base prompt and no examples."
            ),
        )


if __name__ == "__main__":
    main()
