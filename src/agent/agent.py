"""
agent.py — The core HR AI agent with conversation memory.

Memory approach: LangChain's create_agent (1.x API) is stateless by design —
it takes a messages list and returns an updated messages list. We implement
memory by keeping the full message history and passing it back on every turn.
This is the correct pattern for this API version — no separate memory object
needed, the history IS the memory.

Max history: we keep the last N turns to avoid token bloat across long
conversations. Each "turn" = 1 human message + 1 AI response + any tool
call messages in between.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

import config
from src.tools.rag_tool import answer_hr_policy_question
from src.tools.sql_tool import SQL_TOOLS
from src.agent.session import current_session


# ── All tools the agent can choose from ──────────────────────────────────────
ALL_TOOLS = [answer_hr_policy_question] + SQL_TOOLS

# How many past turns to keep in memory. Each turn = human + AI messages.
# Keeping 10 turns = ~20 messages. Beyond this, older turns are dropped
# to avoid hitting token limits on long conversations.
MAX_HISTORY_TURNS = 10


# ── System prompt ─────────────────────────────────────────────────────────────
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
        response = session.chat("Can I take 5 more days?")  # agent remembers previous answer
        session.clear()  # reset history when user logs out
    """

    def __init__(self, agent, max_turns: int = MAX_HISTORY_TURNS):
        self.agent = agent
        self.max_turns = max_turns
        self.message_history = []   # running list of all messages this session

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response, maintaining conversation history.
        The full history is passed to the agent on every call — this is how
        the agent "remembers" previous turns without a separate memory object.
        """
        # Add the new user message to history
        self.message_history.append({"role": "user", "content": user_message})

        # Trim to last N turns to avoid token bloat
        # Each turn = 1 user message + 1+ assistant messages, so multiply by 2
        max_messages = self.max_turns * 2
        trimmed_history = self.message_history[-max_messages:]

        # Run the agent with full history
        result = self.agent.invoke({"messages": trimmed_history})

        # Extract the final text response
        final_message = result["messages"][-1]
        response_text = final_message.content

        # Add the assistant's response to our history for next turn
        self.message_history.append({"role": "assistant", "content": response_text})

        return response_text

    def clear(self):
        """Reset conversation history — call this when user logs out."""
        self.message_history = []

    @property
    def turn_count(self) -> int:
        """How many complete back-and-forth turns have happened."""
        return len(self.message_history) // 2


def ask(agent, question: str) -> str:
    """
    Single-turn ask — no memory. Used by evals and one-off tests.
    For interactive use, use ConversationSession instead.
    """
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    # Test multi-turn conversation memory
    print("Logging in as E004 (Aisha Khan)...\n")
    current_session.login("E004")

    agent = build_agent()
    conversation = ConversationSession(agent)

    # These questions are deliberately connected — the second one
    # only makes sense in the context of the first answer.
    test_turns = [
        "What's my leave balance?",
        "Based on that, how many more annual leave days can I take this year?",
        "What does the policy say about carrying forward unused leave?",
        "So would my unused days carry forward or be forfeited?",
    ]

    for question in test_turns:
        print(f"USER [{conversation.turn_count + 1}]: {question}")
        response = conversation.chat(question)
        print(f"AGENT: {response}\n")
        print("-" * 60)