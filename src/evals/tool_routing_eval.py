"""
tool_routing_eval.py — Objective evaluation of whether the agent calls the
CORRECT tool for a given question.

WHAT THIS MEASURES: eval 1 tested retrieval in isolation, this tests the
agent's decision-making — does it know WHEN to use the policy tool vs the
personal-data tools vs the manager tools? This is the failure mode that
can exist even when every individual tool works perfectly.

HOW IT WORKS: each test case has an expected tool name (or names, for
multi-tool questions). We run the question through the real agent, pull
every tool_call out of the full message history, and check if the
expected tool(s) appear in that list.

Usage:
    python src/evals/tool_routing_eval.py
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.agent import build_agent
from src.agent.session import current_session


# ── Fixed test set — question, who's logged in, and which tool(s) SHOULD
# get called. Some questions reasonably trigger more than one tool (like
# we saw in manual testing) — for those, listing multiple expected tools
# means "at least one of these should appear."
TEST_CASES = [
    {
        "question": "What is the company's dress code?",
        "login_as": "E004",
        "expected_tools": ["answer_hr_policy_question"],
    },
    {
        "question": "How many days of maternity leave do I get according to policy?",
        "login_as": "E004",
        "expected_tools": ["answer_hr_policy_question"],
    },
    {
        "question": "What's my current leave balance?",
        "login_as": "E004",
        "expected_tools": ["check_my_leave_balance"],
    },
    {
        "question": "Show me my leave request history.",
        "login_as": "E006",
        "expected_tools": ["check_my_leave_history"],
    },
    {
        "question": "Who do I report to?",
        "login_as": "E004",
        "expected_tools": ["find_my_manager"],
    },
    {
        "question": "What was my last performance review rating?",
        "login_as": "E004",
        "expected_tools": ["get_my_performance_review"],
    },
    {
        "question": "Who is on my team?",
        "login_as": "E003",
        "expected_tools": ["get_my_team"],
    },
    {
        "question": "Do I have any leave requests waiting for my approval?",
        "login_as": "E003",
        "expected_tools": ["get_my_pending_approvals"],
    },
    {
        "question": "What is the referral bonus amount per the policy?",
        "login_as": "E010",
        "expected_tools": ["answer_hr_policy_question"],
    },
    {
        "question": "What's my profile information?",
        "login_as": "E007",
        "expected_tools": ["get_my_profile"],
    },
]


def extract_tool_calls(result) -> list[str]:
    """Pull every tool name that was actually called during this run."""
    called_tools = []
    for message in result["messages"]:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                called_tools.append(call["name"])
    return called_tools


def run_eval(agent, delay_seconds: float = 15.0):
    """
    delay_seconds: pause between each question to stay under Groq's
    per-minute token rate limit (6000 TPM on the 8B model). Increased to
    15s after 8s still wasn't always enough headroom — token usage per
    call varies slightly depending on response length and retrieved
    chunk sizes, so we need more margin, not just the bare minimum.
    """
    results = []
    correct_count = 0
    total = len(TEST_CASES)

    for i, case in enumerate(TEST_CASES, 1):
        print(f"[{i}/{total}] Running: \"{case['question'][:50]}...\" (logged in as {case['login_as']})", flush=True)
        start = time.time()
        current_session.login(case["login_as"])

        result = agent.invoke({"messages": [{"role": "user", "content": case["question"]}]})
        elapsed = time.time() - start
        called_tools = extract_tool_calls(result)

        # Pass if AT LEAST ONE expected tool was actually called
        is_correct = any(t in called_tools for t in case["expected_tools"])
        if is_correct:
            correct_count += 1

        status = "✓" if is_correct else "✗"
        print(f"        {status} called: {called_tools}  (took {elapsed:.1f}s)", flush=True)

        if i < total:
            print(f"        (waiting {delay_seconds}s before next question to respect rate limit...)\n", flush=True)
            time.sleep(delay_seconds)

        results.append({
            "question": case["question"],
            "login_as": case["login_as"],
            "expected_tools": case["expected_tools"],
            "called_tools": called_tools,
            "correct": is_correct,
        })

    accuracy = correct_count / len(TEST_CASES)
    return results, accuracy


def print_report(results, accuracy):
    print("\n── Tool Routing Evaluation Report ──────────────────────────\n")

    for r in results:
        status = "✓ PASS" if r["correct"] else "✗ FAIL"
        print(f"{status}  [{r['login_as']}] {r['question']}")
        print(f"        expected: {r['expected_tools']}")
        print(f"        called:   {r['called_tools']}")
        print()

    print("──────────────────────────────────────────────────────────")
    print(f"RESULT: {sum(1 for r in results if r['correct'])}/{len(results)} correct "
          f"({accuracy*100:.1f}% tool-routing accuracy)")

    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n{len(failures)} failure(s) to investigate:")
        for f in failures:
            print(f"  - \"{f['question']}\" → expected {f['expected_tools']}, "
                  f"called {f['called_tools']}")
    else:
        print("\nAll test cases passed.")
    print("──────────────────────────────────────────────────────────\n")


def main():
    print("Building agent...")
    agent = build_agent()

    print(f"Running {len(TEST_CASES)} tool-routing test cases...\n")
    results, accuracy = run_eval(agent)

    print_report(results, accuracy)


if __name__ == "__main__":
    main()
