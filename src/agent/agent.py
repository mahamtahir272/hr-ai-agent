"""
agent.py — The core HR Operations Assistant agent.

WHAT THIS FILE DOES:
- Builds a LangChain agent with all available tools
- Provides ConversationSession for multi-turn memory
- Wires reflection (8B model) into every RAG-based answer
- Auto-escalates low-confidence answers via ticket + email tools

TOOLS AVAILABLE (16 total):
  RAG:     answer_hr_policy_question
  SQL:     get_my_profile, check_my_leave_balance, check_my_leave_history,
           find_my_manager, get_my_performance_review, get_my_team,
           get_my_pending_approvals
  Tickets: escalate_to_hr, check_my_tickets, view_open_hr_tickets
  Email:   notify_leave_submitted, notify_leave_decision,
           send_onboarding_email, notify_hr_ticket_created

MODEL ARCHITECTURE:
  Generation:  llama-3.3-70b-versatile (100K tokens/day)
  Reflection:  llama-3.1-8b-instant (500K tokens/day, separate quota)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain.agents import create_agent
from langchain_groq import ChatGroq

import config
from src.tools.rag_tool import answer_hr_policy_question
from src.tools.sql_tool import SQL_TOOLS
from src.tools.ticket_tool import TICKET_TOOLS, create_ticket
from src.tools.email_tool import EMAIL_TOOLS
from src.agent.session import current_session
from src.agent.reflection import reflect_on_answer, format_reflection_footer


# ── All tools the agent can choose from ──────────────────────────────────────
ALL_TOOLS = [answer_hr_policy_question] + SQL_TOOLS + TICKET_TOOLS + EMAIL_TOOLS

# How many past turns to keep in memory (each turn = 1 human + 1 AI message)
MAX_HISTORY_TURNS = 10


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """HR Operations Assistant for Acme Technologies.

- Policy questions (leave rules, benefits, conduct) -> answer_hr_policy_question
- Personal data (own balance, history, manager, review) -> check_my_*/get_my_*/find_my_* tools
- Manager actions (team, approvals) -> get_my_team / get_my_pending_approvals
- Ticket escalation (can't answer confidently) -> escalate_to_hr, then notify_hr_ticket_created
- Email notifications -> notify_leave_submitted / notify_leave_decision / send_onboarding_email
- View own tickets -> check_my_tickets

Rules:
- Never approve/reject leave yourself — human managers only
- After escalating a ticket, always call notify_hr_ticket_created to alert HR
- Email tools fire AFTER human decisions, never autonomously for approvals
- If user says "my"/"I", prefer personal-data tools over policy tools
- If no tool fits, say so honestly and offer to escalate"""


def build_agent():
    """Construct the agent using the current LangChain create_agent API."""
    llm = ChatGroq(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.GROQ_API_KEY,
        max_retries=2,
    )
    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


class ConversationSession:
    """
    Wraps the agent with conversation memory for one user session.

    Usage:
        session = ConversationSession(agent)
        response = session.chat("What's my leave balance?")
        response = session.chat("Can I take 5 more days?")
        session.clear()  # reset on logout
    """

    def __init__(self, agent, max_turns: int = MAX_HISTORY_TURNS):
        self.agent = agent
        self.max_turns = max_turns
        self.message_history = []

    def chat(self, user_message: str) -> str:
        """Send a message, get a response, maintain conversation history."""
        self.message_history.append({"role": "user", "content": user_message})

        # Trim to last N turns to avoid token bloat
        max_messages = self.max_turns * 2
        trimmed_history = self.message_history[-max_messages:]

        result = self.agent.invoke({"messages": trimmed_history})

        final_message = result["messages"][-1]
        response_text = final_message.content

        # ── Reflection step ────────────────────────────────────────────────
        # Only reflect on RAG-based answers (policy questions).
        # SQL tool answers are deterministic — no quality check needed.
        is_policy_answer = any(
            getattr(m, "tool_calls", None) and
            any(tc.get("name") == "answer_hr_policy_question"
                for tc in (m.tool_calls or []))
            for m in result["messages"]
        )

        if is_policy_answer:
            reflection = reflect_on_answer(
                question=user_message,
                draft_answer=response_text,
            )

            if reflection.should_escalate and current_session.is_logged_in():
                ticket = create_ticket(
                    query_text=user_message,
                    agent_response=response_text,
                    confidence_score=reflection.confidence,
                    category="policy",
                )
                response_text += (
                    f"\n\n---\n"
                    f"⚠️ I've automatically raised a support ticket "
                    f"(#{ticket['ticket_id']}) because I'm not fully "
                    f"confident in this answer. HR will follow up within "
                    f"2 business days."
                )
            else:
                response_text += format_reflection_footer(reflection)

        self.message_history.append({"role": "assistant", "content": response_text})
        return response_text

    def clear(self):
        """Reset conversation history — call this when user logs out."""
        self.message_history = []

    @property
    def turn_count(self) -> int:
        return len(self.message_history) // 2


def ask(agent, question: str) -> str:
    """Single-turn ask with no memory. Used by evals."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("HR Operations Assistant — startup test\n")
    print(f"Main model:       {config.LLM_MODEL}")
    print(f"Reflection model: {config.REFLECTION_MODEL}")
    print(f"Total tools:      {len(ALL_TOOLS)}")
    print(f"Tool names:       {[t.name for t in ALL_TOOLS]}\n")

    print("Logging in as E004 (Aisha Khan)...")
    current_session.login("E004")

    agent = build_agent()
    conversation = ConversationSession(agent)

    test_turns = [
        "What's my leave balance?",
        "Based on that, how many more annual leave days can I take this year?",
    ]

    for question in test_turns:
        print(f"\nUSER [{conversation.turn_count + 1}]: {question}")
        response = conversation.chat(question)
        print(f"AGENT: {response}")
        print("-" * 60)
