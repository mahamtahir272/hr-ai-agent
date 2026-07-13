"""
ticket_tool.py — Automatically creates a support ticket in the database
when the agent cannot confidently answer an employee's query.

WHY THIS EXISTS: without this, unresolved queries simply disappear — the
agent says "I don't know" and the employee's issue goes nowhere. This tool
ensures every unresolved question becomes a trackable ticket that HR can
see, assign, and close. Zero manual steps from either the employee or HR.

WHEN THE AGENT CALLS THIS:
- The question is outside the agent's knowledge (payroll issues, legal
  questions, system access problems, etc.)
- The agent's confidence in its answer is low
- The employee explicitly asks to speak to HR

The agent decides when to escalate — this tool just executes it.
"""

import os
import sys
import sqlite3
import uuid
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.tools import tool
from src.agent.session import current_session
import config


def _get_connection():
    conn = sqlite3.connect(os.path.join(config.BASE_DIR, "data", "hr.db"))
    conn.row_factory = sqlite3.Row
    return conn


def create_ticket(
    query_text: str,
    agent_response: str,
    confidence_score: float,
    category: str,
    priority: str = "medium",
) -> dict:
    """
    Write a new ticket to the database. Called internally by the
    LangChain tool wrapper below. Returns the created ticket details.
    """
    employee_id = current_session.current_employee_id

    # Find the default HR Business Partner to assign to
    # In a real system this would be more sophisticated routing logic
    conn = _get_connection()
    cur = conn.cursor()

    # Try to find an HR Business Partner to assign the ticket to
    cur.execute("""
        SELECT employee_id FROM employees
        WHERE role LIKE '%HR%' AND employment_status = 'active'
        LIMIT 1
    """)
    hr_row = cur.fetchone()
    assigned_to = hr_row["employee_id"] if hr_row else None

    ticket_id = f"T{str(uuid.uuid4())[:8].upper()}"
    created_at = datetime.now().isoformat()

    cur.execute("""
        INSERT INTO tickets
        (ticket_id, employee_id, query_text, agent_response, confidence_score,
         category, priority, status, assigned_to, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
    """, (
        ticket_id, employee_id, query_text, agent_response,
        round(confidence_score, 2), category, priority,
        assigned_to, created_at
    ))

    conn.commit()
    conn.close()

    return {
        "ticket_id": ticket_id,
        "assigned_to": assigned_to,
        "created_at": created_at,
    }


def get_my_tickets_from_db() -> list[dict]:
    """Get all tickets for the currently logged-in employee."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT ticket_id, query_text, category, priority, status,
               created_at, resolved_at, resolution_notes
        FROM tickets
        WHERE employee_id = ?
        ORDER BY created_at DESC
    """, (current_session.current_employee_id,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_open_tickets_for_hr() -> list[dict]:
    """Get all open tickets — for HR staff to review."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT t.ticket_id, e.name as employee_name, t.query_text,
               t.category, t.priority, t.status, t.created_at,
               t.agent_response, t.confidence_score
        FROM tickets t
        JOIN employees e ON t.employee_id = e.employee_id
        WHERE t.status IN ('open', 'in_progress')
        ORDER BY
            CASE t.priority
                WHEN 'high' THEN 1
                WHEN 'medium' THEN 2
                WHEN 'low' THEN 3
            END,
            t.created_at ASC
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── LangChain tool wrappers ───────────────────────────────────────────────────

@tool
def escalate_to_hr(
    query_text: str,
    agent_response: str,
    confidence_score: float,
    category: str,
) -> str:
    """
    Create a support ticket for HR when the agent cannot confidently
    answer the employee's question. Use this when: the question is about
    payroll, salary, legal matters, system access, or anything outside
    HR policy and employee data. Also use when confidence is below 0.4.
    category must be one of: leave, policy, payroll, it_access, other.
    """
    if (err := current_session.login_check()):
        return err

    valid_categories = {"leave", "policy", "payroll", "it_access", "other"}
    if category not in valid_categories:
        category = "other"

    ticket = create_ticket(
        query_text=query_text,
        agent_response=agent_response,
        confidence_score=confidence_score,
        category=category,
        priority="high" if confidence_score < 0.3 else "medium",
    )

    return (
        f"I've created a support ticket (#{ticket['ticket_id']}) for your query. "
        f"An HR team member has been notified and will get back to you within "
        f"2 business days. Your ticket ID is {ticket['ticket_id']} — "
        f"you can use this to follow up with HR directly."
    )


@tool
def check_my_tickets() -> str:
    """
    Show the logged-in employee their own support ticket history —
    open, in-progress, and resolved tickets. Use when employee asks
    about their HR tickets or wants to follow up on a previous query.
    """
    if (err := current_session.login_check()):
        return err

    tickets = get_my_tickets_from_db()
    if not tickets:
        return "You have no support tickets on record."

    lines = ["Your support tickets:"]
    for t in tickets:
        resolved = f" — resolved: {t['resolved_at']}" if t["resolved_at"] else ""
        lines.append(
            f"  [{t['ticket_id']}] {t['category'].upper()} | "
            f"{t['priority']} priority | {t['status'].upper()}{resolved}\n"
            f"    Query: {t['query_text'][:80]}..."
        )
    return "\n".join(lines)


@tool
def view_open_hr_tickets() -> str:
    """
    MANAGER/HR-ONLY: view all open support tickets awaiting HR attention,
    sorted by priority then age. Use when an HR staff member asks to see
    pending tickets or wants to review escalated queries.
    """
    if (err := current_session.manager_check()):
        return err

    tickets = get_open_tickets_for_hr()
    if not tickets:
        return "No open support tickets at this time."

    lines = [f"Open HR support tickets ({len(tickets)} total):"]
    for t in tickets:
        lines.append(
            f"\n  [{t['ticket_id']}] {t['priority'].upper()} priority | "
            f"{t['category']} | raised by {t['employee_name']}"
        )
        lines.append(f"    Query: {t['query_text'][:100]}")
        lines.append(
            f"    Agent confidence: {t['confidence_score']} | "
            f"Created: {t['created_at'][:10]}"
        )
    return "\n".join(lines)


# Collected list for easy import into the agent
TICKET_TOOLS = [escalate_to_hr, check_my_tickets, view_open_hr_tickets]


if __name__ == "__main__":
    # Quick test — run: python src/tools/ticket_tool.py
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    print("── Testing ticket creation (logged in as E007) ──")
    current_session.login("E007")

    result = escalate_to_hr.invoke({
        "query_text": "My salary for last month was not credited to my account.",
        "agent_response": "I was unable to find specific information about salary payment issues in the available policy documents or employee data.",
        "confidence_score": 0.18,
        "category": "payroll",
    })
    print(result)

    print("\n── Checking tickets for E007 ──")
    print(check_my_tickets.invoke({}))

    print("\n── Logging in as manager E009 to view all open tickets ──")
    current_session.login("E009")
    print(view_open_hr_tickets.invoke({}))
