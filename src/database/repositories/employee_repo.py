"""
employee_repo.py — Safe, parameterized database access functions.

IMPORTANT DESIGN DECISION: The agent never writes or sees raw SQL. It can
only call these predefined functions. This avoids SQL injection risk and
prevents the LLM from accidentally running a destructive query (e.g. DELETE,
DROP, UPDATE on the wrong row). Every function below only SELECTs data.

This is the "repository pattern" — database logic lives here, completely
separated from agent/tool logic. If we ever swap SQLite for PostgreSQL,
only this file changes.
"""

import os
import sys
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import config


def _get_connection():
    """Open a connection with row access by column name."""
    conn = sqlite3.connect(config.BASE_DIR + "/data/hr.db")
    conn.row_factory = sqlite3.Row
    return conn


def get_employee_by_id(employee_id: str) -> dict | None:
    """Fetch full employee record by ID. Returns None if not found."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT e.employee_id, e.name, e.email, e.role, d.department_name,
               e.manager_id, m.name as manager_name, e.date_of_joining,
               e.employment_status, e.salary_band
        FROM employees e
        JOIN departments d ON e.department_id = d.department_id
        LEFT JOIN employees m ON e.manager_id = m.employee_id
        WHERE e.employee_id = ?
    """, (employee_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_employee_by_email(email: str) -> dict | None:
    """Fetch employee by email — useful for login/identification."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("SELECT employee_id FROM employees WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return get_employee_by_id(row["employee_id"])


def get_leave_balance(employee_id: str, year: int = 2025) -> list[dict]:
    """Get all leave type balances for an employee for a given year."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT lt.leave_type_name, lb.entitled_days, lb.used_days, lb.remaining_days
        FROM leave_balances lb
        JOIN leave_types lt ON lb.leave_type_id = lt.leave_type_id
        WHERE lb.employee_id = ? AND lb.year = ?
        ORDER BY lt.leave_type_name
    """, (employee_id, year))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_leave_requests(employee_id: str, status: str | None = None) -> list[dict]:
    """
    Get leave requests for an employee, optionally filtered by status
    (pending/approved/rejected). Returns most recent first.
    """
    conn = _get_connection()
    cur = conn.cursor()
    if status:
        cur.execute("""
            SELECT lr.request_id, lt.leave_type_name, lr.start_date, lr.end_date,
                   lr.total_days, lr.reason, lr.status, lr.applied_on
            FROM leave_requests lr
            JOIN leave_types lt ON lr.leave_type_id = lt.leave_type_id
            WHERE lr.employee_id = ? AND lr.status = ?
            ORDER BY lr.applied_on DESC
        """, (employee_id, status))
    else:
        cur.execute("""
            SELECT lr.request_id, lt.leave_type_name, lr.start_date, lr.end_date,
                   lr.total_days, lr.reason, lr.status, lr.applied_on
            FROM leave_requests lr
            JOIN leave_types lt ON lr.leave_type_id = lt.leave_type_id
            WHERE lr.employee_id = ?
            ORDER BY lr.applied_on DESC
        """, (employee_id,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_manager_of(employee_id: str) -> dict | None:
    """Get the manager's details for a given employee."""
    emp = get_employee_by_id(employee_id)
    if not emp or not emp.get("manager_id"):
        return None
    return get_employee_by_id(emp["manager_id"])


def get_direct_reports(manager_id: str) -> list[dict]:
    """Get all employees who report directly to a given manager."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT employee_id, name, role, employment_status
        FROM employees
        WHERE manager_id = ?
        ORDER BY name
    """, (manager_id,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_performance_review(employee_id: str) -> dict | None:
    """Get the most recent performance review for an employee."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT review_period, rating, goals_met, strengths, areas_to_improve, comments
        FROM performance_reviews
        WHERE employee_id = ?
        ORDER BY reviewed_on DESC
        LIMIT 1
    """, (employee_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_pending_leave_requests_for_manager(manager_id: str) -> list[dict]:
    """
    Get all pending leave requests awaiting approval from a specific manager.
    NOTE: This is for DISPLAY/surfacing only — the agent should never
    auto-approve or auto-reject. A human manager makes that decision.
    """
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT lr.request_id, e.name as employee_name, lt.leave_type_name,
               lr.start_date, lr.end_date, lr.total_days, lr.reason, lr.applied_on
        FROM leave_requests lr
        JOIN employees e ON lr.employee_id = e.employee_id
        JOIN leave_types lt ON lr.leave_type_id = lt.leave_type_id
        WHERE e.manager_id = ? AND lr.status = 'pending'
        ORDER BY lr.applied_on ASC
    """, (manager_id,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    # Quick manual test — run: python src/database/repositories/employee_repo.py
    print("── Employee lookup (E004) ──")
    print(get_employee_by_id("E004"))

    print("\n── Leave balance (E004, 2025) ──")
    for b in get_leave_balance("E004"):
        print(b)

    print("\n── Leave requests (E006) ──")
    for r in get_leave_requests("E006"):
        print(r)

    print("\n── Manager of E004 ──")
    print(get_manager_of("E004"))

    print("\n── Direct reports of E003 ──")
    for r in get_direct_reports("E003"):
        print(r)

    print("\n── Latest performance review (E004) ──")
    print(get_latest_performance_review("E004"))

    print("\n── Pending leave requests for manager E003 ──")
    for r in get_pending_leave_requests_for_manager("E003"):
        print(r)
