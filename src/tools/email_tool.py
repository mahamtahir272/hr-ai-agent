"""
email_tool.py — Automated email notification system.

DESIGN DECISION — MOCK vs REAL:
We mock the actual email send (print to console) instead of using SendGrid
because: (1) no card required, (2) behavior is identical for testing,
(3) swapping to real sending is a one-line change.

To switch to real SendGrid later:
1. pip install sendgrid
2. Add SENDGRID_API_KEY to .env
3. Replace _mock_send_email() with _real_send_email() below

Every email attempt — real or mocked — is logged to the notifications
table in hr.db. This gives HR a full audit trail of all automated
communications, which is a real production requirement.

WHEN EMAILS ARE SENT (all automated, zero manual steps):
- Leave request submitted → confirmation to employee
- Leave approved → notification to employee
- Leave rejected → notification to employee
- New employee joins → onboarding welcome email
- Ticket escalated → alert to assigned HR staff
- Performance review due → reminder to manager
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


def _log_notification(employee_id, notification_type, recipient_email,
                       subject, body, status="mocked"):
    """Write every email attempt to the notifications table."""
    conn = _get_connection()
    cur = conn.cursor()
    notification_id = f"N{str(uuid.uuid4())[:8].upper()}"
    cur.execute("""
        INSERT INTO notifications
        (notification_id, employee_id, notification_type, recipient_email,
         subject, body, status, sent_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        notification_id, employee_id, notification_type,
        recipient_email, subject, body, status,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()
    return notification_id


def _mock_send_email(recipient_email, subject, body):
    """
    Mock email sender — prints to console instead of actually sending.
    Replace this function body with real SendGrid code for production.
    """
    print(f"\n{'='*60}")
    print(f"📧 EMAIL SENT (mocked)")
    print(f"   To:      {recipient_email}")
    print(f"   Subject: {subject}")
    print(f"   Body:    {body[:200]}{'...' if len(body) > 200 else ''}")
    print(f"{'='*60}\n")
    return True


def send_notification(
    employee_id: str,
    notification_type: str,
    recipient_email: str,
    subject: str,
    body: str,
) -> dict:
    """
    Core notification sender — mocks the send and logs to database.
    Called by all the @tool functions below.
    Returns notification_id and status.
    """
    success = _mock_send_email(recipient_email, subject, body)
    status = "mocked" if success else "failed"
    notification_id = _log_notification(
        employee_id, notification_type, recipient_email, subject, body, status
    )
    return {"notification_id": notification_id, "status": status}


# ── Pre-built email templates ─────────────────────────────────────────────────

def _leave_submitted_email(employee_name, leave_type, start_date, end_date, days):
    return {
        "subject": f"Leave Request Submitted — {leave_type} ({days} days)",
        "body": (
            f"Hi {employee_name},\n\n"
            f"Your {leave_type} request for {days} day(s) "
            f"({start_date} to {end_date}) has been submitted successfully.\n\n"
            f"Your manager will review it within 2 working days. "
            f"You will receive a confirmation email once a decision is made.\n\n"
            f"Best regards,\nHR Team — Acme Technologies"
        )
    }


def _leave_approved_email(employee_name, leave_type, start_date, end_date, days, manager_name):
    return {
        "subject": f"✅ Leave Approved — {leave_type} ({days} days)",
        "body": (
            f"Hi {employee_name},\n\n"
            f"Great news! Your {leave_type} request for {days} day(s) "
            f"({start_date} to {end_date}) has been approved by {manager_name}.\n\n"
            f"Please ensure your work is handed over before your leave begins. "
            f"Enjoy your time off!\n\n"
            f"Best regards,\nHR Team — Acme Technologies"
        )
    }


def _leave_rejected_email(employee_name, leave_type, start_date, end_date, days, manager_name, reason=""):
    return {
        "subject": f"❌ Leave Request Not Approved — {leave_type}",
        "body": (
            f"Hi {employee_name},\n\n"
            f"Unfortunately, your {leave_type} request for {days} day(s) "
            f"({start_date} to {end_date}) was not approved by {manager_name}.\n\n"
            f"{'Reason: ' + reason + chr(10) + chr(10) if reason else ''}"
            f"Please speak with your manager or contact HR if you have questions.\n\n"
            f"Best regards,\nHR Team — Acme Technologies"
        )
    }


def _onboarding_email(employee_name, employee_id, role, manager_name, manager_email):
    return {
        "subject": f"Welcome to Acme Technologies, {employee_name.split()[0]}! 🎉",
        "body": (
            f"Hi {employee_name},\n\n"
            f"Welcome aboard! We're thrilled to have you join us.\n\n"
            f"Your employee ID is: {employee_id}\n"
            f"Your role: {role}\n"
            f"Your manager: {manager_name} ({manager_email})\n\n"
            f"Day 1 Schedule:\n"
            f"• 9:30 AM — HR orientation (HR conference room)\n"
            f"• 11:00 AM — IT setup and laptop handover\n"
            f"• 1:00 PM — Lunch with your team\n"
            f"• 2:30 PM — Meet your manager and buddy\n\n"
            f"Please bring a valid ID for verification.\n\n"
            f"Looking forward to seeing you!\n\n"
            f"Best regards,\nHR Team — Acme Technologies"
        )
    }


def _ticket_escalation_email(hr_name, hr_email, employee_name, ticket_id, query, category):
    return {
        "subject": f"[HR Ticket #{ticket_id}] New {category.upper()} query requires attention",
        "body": (
            f"Hi {hr_name},\n\n"
            f"A support ticket has been raised that requires your attention.\n\n"
            f"Ticket ID: {ticket_id}\n"
            f"Raised by: {employee_name}\n"
            f"Category: {category}\n"
            f"Query: {query}\n\n"
            f"Please review and respond within 2 business days.\n\n"
            f"Best regards,\nHR AI System — Acme Technologies"
        )
    }


# ── LangChain tool wrappers ───────────────────────────────────────────────────

@tool
def notify_leave_submitted(
    employee_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    total_days: int,
) -> str:
    """Send email confirmation to employee when they submit a leave request."""
    if (err := current_session.login_check()):
        return err

    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name, email FROM employees WHERE employee_id = ?", (employee_id,))
    emp = cur.fetchone()
    conn.close()

    if not emp:
        return f"Employee {employee_id} not found."

    template = _leave_submitted_email(
        emp["name"], leave_type, start_date, end_date, total_days
    )
    result = send_notification(
        employee_id, "leave_submitted", emp["email"],
        template["subject"], template["body"]
    )
    return f"Confirmation email sent to {emp['email']} (#{result['notification_id']})"


@tool
def notify_leave_decision(
    employee_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    total_days: int,
    decision: str,
    reason: str = "",
) -> str:
    """
    Send leave approval or rejection email to employee.
    decision must be 'approved' or 'rejected'.
    Use this AFTER a human manager has made the decision — never autonomously.
    """
    if (err := current_session.login_check()):
        return err

    if decision not in ("approved", "rejected"):
        return "decision must be 'approved' or 'rejected'."

    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT e.name, e.email, m.name as manager_name
        FROM employees e
        LEFT JOIN employees m ON e.manager_id = m.employee_id
        WHERE e.employee_id = ?
    """, (employee_id,))
    emp = cur.fetchone()
    conn.close()

    if not emp:
        return f"Employee {employee_id} not found."

    if decision == "approved":
        template = _leave_approved_email(
            emp["name"], leave_type, start_date, end_date,
            total_days, emp["manager_name"]
        )
        notif_type = "leave_approved"
    else:
        template = _leave_rejected_email(
            emp["name"], leave_type, start_date, end_date,
            total_days, emp["manager_name"], reason
        )
        notif_type = "leave_rejected"

    result = send_notification(
        employee_id, notif_type, emp["email"],
        template["subject"], template["body"]
    )
    return (
        f"Leave {decision} notification sent to {emp['email']} "
        f"(#{result['notification_id']})"
    )


@tool
def send_onboarding_email(employee_id: str) -> str:
    """
    Send welcome onboarding email to a new employee.
    Use when a new hire's record is created in the system.
    """
    if (err := current_session.manager_check()):
        return err

    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT e.employee_id, e.name, e.email, e.role,
               m.name as manager_name, m.email as manager_email
        FROM employees e
        LEFT JOIN employees m ON e.manager_id = m.employee_id
        WHERE e.employee_id = ?
    """, (employee_id,))
    emp = cur.fetchone()
    conn.close()

    if not emp:
        return f"Employee {employee_id} not found."

    template = _onboarding_email(
        emp["name"], emp["employee_id"], emp["role"],
        emp["manager_name"], emp["manager_email"]
    )
    result = send_notification(
        employee_id, "onboarding", emp["email"],
        template["subject"], template["body"]
    )
    return (
        f"Onboarding email sent to {emp['name']} at {emp['email']} "
        f"(#{result['notification_id']})"
    )


@tool
def notify_hr_ticket_created(ticket_id: str, employee_id: str, query: str, category: str) -> str:
    """
    Notify the assigned HR staff member when a support ticket is created.
    Called automatically after escalate_to_hr creates a ticket.
    """
    conn = _get_connection()
    cur = conn.cursor()

    # Get the employee who raised the ticket
    cur.execute("SELECT name FROM employees WHERE employee_id = ?", (employee_id,))
    emp = cur.fetchone()

    # Get the assigned HR person from the ticket
    cur.execute("""
        SELECT e.name, e.email FROM tickets t
        JOIN employees e ON t.assigned_to = e.employee_id
        WHERE t.ticket_id = ?
    """, (ticket_id,))
    hr = cur.fetchone()
    conn.close()

    if not hr:
        return "No HR staff assigned to this ticket — skipping email."

    template = _ticket_escalation_email(
        hr["name"], hr["email"],
        emp["name"] if emp else "Unknown Employee",
        ticket_id, query, category
    )
    result = send_notification(
        employee_id, "ticket_created", hr["email"],
        template["subject"], template["body"]
    )
    return (
        f"HR notification sent to {hr['name']} at {hr['email']} "
        f"(#{result['notification_id']})"
    )


# Collected list for easy import into the agent
EMAIL_TOOLS = [
    notify_leave_submitted,
    notify_leave_decision,
    send_onboarding_email,
    notify_hr_ticket_created,
]


if __name__ == "__main__":
    # Test all 4 email types — zero LLM calls, zero quota
    print("Testing email notification tool...\n")

    # Need a session for the tools that check login
    current_session.login("E003")   # Tariq Aziz — manager

    print("── Test 1: Leave submitted notification ──")
    result = notify_leave_submitted.invoke({
        "employee_id": "E006",
        "leave_type": "Annual Leave",
        "start_date": "2026-08-01",
        "end_date": "2026-08-05",
        "total_days": 5,
    })
    print(result)

    print("\n── Test 2: Leave approved notification ──")
    result = notify_leave_decision.invoke({
        "employee_id": "E006",
        "leave_type": "Annual Leave",
        "start_date": "2026-08-01",
        "end_date": "2026-08-05",
        "total_days": 5,
        "decision": "approved",
        "reason": "",
    })
    print(result)

    print("\n── Test 3: Leave rejected notification ──")
    result = notify_leave_decision.invoke({
        "employee_id": "E007",
        "leave_type": "Annual Leave",
        "start_date": "2026-08-10",
        "end_date": "2026-08-14",
        "total_days": 5,
        "decision": "rejected",
        "reason": "Critical sprint release scheduled during this period.",
    })
    print(result)

    print("\n── Test 4: Onboarding email ──")
    result = send_onboarding_email.invoke({"employee_id": "E008"})
    print(result)

    print("\n── Test 5: HR ticket notification ──")
    # Use the ticket we created earlier in ticket_tool test
    result = notify_hr_ticket_created.invoke({
        "ticket_id": "T003",
        "employee_id": "E007",
        "query": "Performance rating appeal process",
        "category": "policy",
    })
    print(result)

    print("\n── Checking notification log in database ──")
    conn = sqlite3.connect(os.path.join(config.BASE_DIR, "data", "hr.db"))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT notification_id, notification_type, recipient_email, status, sent_at
        FROM notifications
        ORDER BY sent_at DESC
        LIMIT 5
    """)
    for row in cur.fetchall():
        print(dict(row))
    conn.close()
