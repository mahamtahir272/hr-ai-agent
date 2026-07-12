"""
session.py — Tracks the identity of the currently logged-in user for one
conversation session.

WHY THIS EXISTS: Without this, any user could ask "show me E004's leave
balance" regardless of who they actually are — that's an authorization
bug, not an accuracy bug. This module is the single source of truth for
"who is asking right now," and every tool call gets checked against it.

This is intentionally simple for the project's scope — a real production
system would use proper auth (JWT, OAuth) instead of a manually-set
employee_id. We're keeping the concept correct without over-engineering
the implementation.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.repositories import employee_repo


class Session:
    """Holds the identity of the current user for this conversation."""

    def __init__(self):
        self.current_employee_id: str | None = None
        self.current_employee: dict | None = None

    def login(self, employee_id: str) -> dict:
        """
        Log in as a given employee ID. In a real system this would be
        replaced by proper authentication (SSO, JWT token, etc.) — here
        we simulate it by directly setting which employee is 'logged in'.
        """
        employee = employee_repo.get_employee_by_id(employee_id)
        if not employee:
            raise ValueError(f"No employee found with ID {employee_id}. Cannot log in.")

        self.current_employee_id = employee_id
        self.current_employee = employee
        return employee

    def logout(self):
        self.current_employee_id = None
        self.current_employee = None

    def is_logged_in(self) -> bool:
        return self.current_employee_id is not None

    def is_manager(self) -> bool:
        """
        Check if the current user manages anyone. Used to gate
        manager-only tools like get_pending_approvals, get_team_members.
        """
        if not self.is_logged_in():
            return False
        reports = employee_repo.get_direct_reports(self.current_employee_id)
        return len(reports) > 0

    def require_login(self):
        if not self.is_logged_in():
            raise PermissionError("No user is logged in. Cannot perform this action.")

    def require_manager(self):
        self.require_login()
        if not self.is_manager():
            raise PermissionError(
                f"{self.current_employee['name']} does not manage any employees "
                f"and cannot access manager-only tools."
            )

    def login_check(self) -> str | None:
        """
        Non-raising version of require_login(), safe to call inside a
        LangChain @tool function. Returns an error message string if not
        logged in, or None if everything is fine. Tools should do:
            if (err := current_session.login_check()): return err
        instead of calling require_login() directly, since an uncaught
        PermissionError inside a tool call crashes the entire agent run
        rather than gracefully returning a message to the user.
        """
        if not self.is_logged_in():
            return "No user is logged in. Cannot perform this action."
        return None

    def manager_check(self) -> str | None:
        """Non-raising version of require_manager(). See login_check() docstring."""
        login_error = self.login_check()
        if login_error:
            return login_error
        if not self.is_manager():
            return (
                f"Access denied: {self.current_employee['name']} does not manage "
                f"any employees and cannot access manager-only tools."
            )
        return None


# A single global session for this app run. In a real web app, this would
# instead be a per-user session stored in Redis or a similar store, keyed
# by login token — not a single global object shared by everyone.
current_session = Session()


if __name__ == "__main__":
    # Quick manual test — run: python src/agent/session.py
    print("── Logging in as E004 (Aisha Khan, not a manager) ──")
    current_session.login("E004")
    print(f"Logged in: {current_session.current_employee['name']}")
    print(f"Is manager: {current_session.is_manager()}")

    try:
        current_session.require_manager()
    except PermissionError as e:
        print(f"Correctly blocked: {e}")

    print("\n── Logging in as E003 (Tariq Aziz, IS a manager) ──")
    current_session.login("E003")
    print(f"Logged in: {current_session.current_employee['name']}")
    print(f"Is manager: {current_session.is_manager()}")
    current_session.require_manager()
    print("Correctly allowed — no exception raised.")
