"""
sql_tool.py — Wraps employee_repo.py functions as LangChain tools, with
session-based access control enforced on every call.

KEY DESIGN CHANGE: tools no longer take employee_id as a free-text argument
that anyone could fill in with anyone else's ID. Instead, employee-facing
tools always operate on whoever is currently logged in (current_session),
and manager-facing tools check that the logged-in user actually manages
people before running. This closes the authorization gap — an employee
can no longer ask "show me E004's balance" while logged in as someone else.

IMPORTANT: every tool uses the non-raising login_check()/manager_check()
methods, NOT require_login()/require_manager(). An uncaught exception
inside a LangChain tool crashes the entire agent run rather than letting
the agent gracefully tell the user "I can't do that" — we found this the
hard way when a misrouted question hit a manager-only tool as a non-manager
and crashed the whole script instead of returning a clean denial message.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.tools import tool
from src.database.repositories import employee_repo
from src.agent.session import current_session


@tool
def get_my_profile() -> str:
    """Get logged-in user's own profile: name, role, dept, manager, status."""
    if (err := current_session.login_check()):
        return err
    emp = current_session.current_employee
    return (
        f"{emp['name']} ({emp['employee_id']}) is a {emp['role']} in "
        f"{emp['department_name']}. Reports to {emp['manager_name']}. "
        f"Joined on {emp['date_of_joining']}. Status: {emp['employment_status']}."
    )


@tool
def check_my_leave_balance() -> str:
    """Get logged-in user's leave balance (annual/sick/unpaid remaining days)."""
    if (err := current_session.login_check()):
        return err
    employee_id = current_session.current_employee_id
    balances = employee_repo.get_leave_balance(employee_id)
    if not balances:
        return "No leave balance records found for your account."

    lines = ["Your leave balance:"]
    for b in balances:
        lines.append(
            f"  - {b['leave_type_name']}: {b['remaining_days']} days remaining "
            f"(used {b['used_days']} of {b['entitled_days']} entitled)"
        )
    return "\n".join(lines)


@tool
def check_my_leave_history(status_filter: str = "") -> str:
    """Get logged-in user's leave request history. status_filter: pending/approved/rejected, or empty for all."""
    if (err := current_session.login_check()):
        return err
    employee_id = current_session.current_employee_id

    status = status_filter.strip().lower() if status_filter else None
    if status and status not in ("pending", "approved", "rejected"):
        status = None

    requests = employee_repo.get_leave_requests(employee_id, status=status)
    if not requests:
        filter_text = f" with status '{status}'" if status else ""
        return f"No leave requests found{filter_text}."

    lines = ["Your leave requests:"]
    for r in requests:
        lines.append(
            f"  - {r['leave_type_name']}: {r['start_date']} to {r['end_date']} "
            f"({r['total_days']} days) — {r['status'].upper()} — reason: {r['reason']}"
        )
    return "\n".join(lines)


@tool
def find_my_manager() -> str:
    """Find the logged-in user's manager (name, role, email)."""
    if (err := current_session.login_check()):
        return err
    manager = employee_repo.get_manager_of(current_session.current_employee_id)
    if not manager:
        return "You have no manager on record (you may be a top-level executive)."
    return f"{manager['name']} ({manager['employee_id']}), {manager['role']}, email: {manager['email']}"


@tool
def get_my_performance_review() -> str:
    """Get logged-in user's most recent performance review and rating."""
    if (err := current_session.login_check()):
        return err
    review = employee_repo.get_latest_performance_review(current_session.current_employee_id)
    if not review:
        return "No performance reviews found for your account."
    return (
        f"Your latest review ({review['review_period']}): {review['rating']}, "
        f"{review['goals_met']}% goals met.\n"
        f"Strengths: {review['strengths']}\n"
        f"Areas to improve: {review['areas_to_improve']}\n"
        f"Comments: {review['comments']}"
    )


# ── Manager-only tools — require current user to actually manage people ─────

@tool
def get_my_team() -> str:
    """MANAGER-ONLY: list direct reports. Denied if user isn't a manager."""
    if (err := current_session.manager_check()):
        return err
    reports = employee_repo.get_direct_reports(current_session.current_employee_id)
    if not reports:
        return "You have no direct reports."

    lines = ["Your direct reports:"]
    for r in reports:
        lines.append(f"  - {r['name']} ({r['employee_id']}), {r['role']} — {r['employment_status']}")
    return "\n".join(lines)


@tool
def get_my_pending_approvals() -> str:
    """MANAGER-ONLY: list pending leave requests awaiting approval. Surfacing only, agent cannot approve/reject. Denied if user isn't a manager."""
    if (err := current_session.manager_check()):
        return err
    pending = employee_repo.get_pending_leave_requests_for_manager(
        current_session.current_employee_id
    )
    if not pending:
        return "No pending leave requests awaiting your approval."

    lines = ["Pending leave requests for you to review:"]
    for p in pending:
        lines.append(
            f"  - {p['employee_name']} requested {p['leave_type_name']}: "
            f"{p['start_date']} to {p['end_date']} ({p['total_days']} days). "
            f"Reason: {p['reason']}. Applied on {p['applied_on']}."
        )
    lines.append("\nNote: These require manual review and approval by you in the "
                  "HR system — I cannot approve or reject leave requests automatically.")
    return "\n".join(lines)


# Collected list of all SQL-backed tools, for easy import into the agent
SQL_TOOLS = [
    get_my_profile,
    check_my_leave_balance,
    check_my_leave_history,
    find_my_manager,
    get_my_performance_review,
    get_my_team,
    get_my_pending_approvals,
]


if __name__ == "__main__":
    # Quick manual test — run: python src/tools/sql_tool.py
    print("── Logged in as E004 (Aisha Khan, employee, not a manager) ──")
    current_session.login("E004")
    print(check_my_leave_balance.invoke({}))
    print()

    print("Trying a manager-only tool as a non-manager (should NOT crash):")
    print(get_my_team.invoke({}))

    print("\n── Logging in as E003 (Tariq Aziz, manager) ──")
    current_session.login("E003")
    print(get_my_team.invoke({}))
    print()
    print(get_my_pending_approvals.invoke({}))
