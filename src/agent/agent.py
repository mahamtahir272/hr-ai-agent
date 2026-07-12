"""
agent.py — The core HR AI agent. Decides, for each user message, whether
to answer from policy documents (RAG tool) or from the employee's own
data (SQL tools), and calls the right one automatically.

This uses LangChain's current create_agent API (LangChain 1.x). Older
tutorials use create_tool_calling_agent + AgentExecutor — that pattern
was removed/replaced in LangChain 1.x in favor of this single, simpler
create_agent() function, which returns a ready-to-invoke compiled graph.

The agent itself never touches the database or vector store directly —
it only ever calls the tools we already built and tested.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain.agents import create_agent
from langchain_groq import ChatGroq

import config
from src.tools.rag_tool import answer_hr_policy_question
from src.tools.sql_tool import SQL_TOOLS
from src.agent.session import current_session


# ── All tools the agent can choose from ──────────────────────────────────────
ALL_TOOLS = [answer_hr_policy_question] + SQL_TOOLS


# ── System prompt — trimmed to reduce tokens sent on every call ────────────
# (Kept short deliberately — this is sent in full on EVERY agent turn,
# alongside all tool descriptions, so verbosity here directly costs tokens
# and contributed to hitting Groq's per-minute rate limit on the 8B model.)
SYSTEM_PROMPT = """HR assistant for Acme Technologies.

- Policy questions (leave rules, benefits, conduct) -> answer_hr_policy_question
- Personal data (own balance, history, manager, review) -> matching check_my_*/get_my_*/find_my_* tool. These always use the logged-in user automatically.
- Manager-only actions (team, approvals) -> get_my_team / get_my_pending_approvals. Will be denied if user isn't a manager.

Rules: Never approve/reject leave yourself — only a human manager can. If user uses "my"/"I", prefer personal-data tools over policy tools. If no tool fits, say so honestly. Be concise."""


def build_agent():
    """Construct the agent using the current LangChain create_agent API."""
    llm = ChatGroq(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.GROQ_API_KEY,
        max_retries=2,  # default SDK retry behavior can silently sleep for a
                        # long time on 429s with no console output, which
                        # looks like a hang. Lower retries = fail faster and
                        # visibly, so a real rate-limit shows up as an error
                        # we can see, not a mysterious freeze.
    )

    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def ask(agent, question: str) -> str:
    """
    Run a single question through the agent and return the final answer.
    create_agent's compiled graph expects a "messages" list as input and
    returns the full updated state — the final answer is the last message.
    """
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final_message = result["messages"][-1]
    return final_message.content


if __name__ == "__main__":
    # Quick manual test — run: python src/agent/agent.py
    print("Logging in as E004 (Aisha Khan, employee)...\n")
    current_session.login("E004")

    hr_agent = build_agent()

    test_questions = [
        "How many days of annual leave do I get?",
        "What's my leave balance?",
        "Who is my manager?",
        "Can you approve my pending leave request?",
    ]

    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"USER: {q}")
        print('='*70)
        answer = ask(hr_agent, q)
        print(f"\nAGENT: {answer}")
