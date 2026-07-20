"""
main.py — FastAPI backend for the HR Operations Assistant.

ENDPOINTS:
  POST /auth/login          — log in as an employee (sets session)
  POST /auth/logout         — clear session
  GET  /auth/me             — get current logged-in employee info

  POST /chat                — send a message, get agent response
  GET  /chat/history        — get conversation history for current session

  GET  /employee/profile    — current employee's profile
  GET  /employee/leave      — current employee's leave balance
  GET  /employee/tickets    — current employee's support tickets

  GET  /hr/tickets          — all open tickets (manager/HR only)
  GET  /hr/jobs             — all open job postings
  GET  /hr/screenings       — all resume screening results
  POST /hr/screen           — screen a single resume
  POST /hr/screen-batch     — screen multiple resumes at once

  GET  /health              — health check

DESIGN DECISIONS:
- Sessions stored in-memory per server instance (simple, correct for dev)
- No JWT/OAuth — simplified auth for a portfolio project, noted in README
- Conversation history tied to session_id passed in request headers
- CORS enabled for local frontend development
"""

import os
import sys
import uuid
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from src.agent.agent import build_agent, ConversationSession
from src.agent.session import Session
from src.database.repositories import employee_repo
from src.tools.resume_screening_tool import screen_resume, screen_batch, _list_open_jobs, _get_job_posting
from src.tools.ticket_tool import get_open_tickets_for_hr


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HR Operations Assistant API",
    description="AI-powered HR assistant with RAG, tool-calling, reflection, and automated workflows.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-memory session store ───────────────────────────────────────────────────
# Maps session_id → {"session": Session, "conversation": ConversationSession}
# In production this would be Redis with TTL expiry
_sessions: dict = {}
_agent = None  # built once on startup, reused across requests


def _get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def _get_session(x_session_id: Optional[str] = Header(None)) -> dict:
    """Dependency — validates session_id from request header."""
    if not x_session_id or x_session_id not in _sessions:
        raise HTTPException(status_code=401, detail="Not authenticated. POST /auth/login first.")
    return _sessions[x_session_id]


# ── Request/Response models ───────────────────────────────────────────────────

class LoginRequest(BaseModel):
    employee_id: str

class LoginResponse(BaseModel):
    session_id: str
    employee_id: str
    name: str
    role: str
    department: str
    is_manager: bool

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    turn_count: int

class ScreenRequest(BaseModel):
    candidate_name: str
    candidate_email: str
    job_id: str
    resume_text: str

class BatchCandidate(BaseModel):
    name: str
    email: str
    resume: str

class BatchScreenRequest(BaseModel):
    job_id: str
    candidates: list[BatchCandidate]


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """
    Log in as an employee. Returns a session_id to include in
    subsequent requests as the X-Session-Id header.
    """
    emp = employee_repo.get_employee_by_id(req.employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail=f"Employee {req.employee_id} not found.")

    if emp["employment_status"] == "resigned":
        raise HTTPException(status_code=403, detail="This employee account is no longer active.")

    # Create a new session
    session_id = str(uuid.uuid4())
    user_session = Session()
    user_session.login(req.employee_id)

    conversation = ConversationSession(_get_agent())

    _sessions[session_id] = {
        "session": user_session,
        "conversation": conversation,
        "employee": emp,
    }

    return LoginResponse(
        session_id=session_id,
        employee_id=emp["employee_id"],
        name=emp["name"],
        role=emp["role"],
        department=emp["department_name"],
        is_manager=user_session.is_manager(),
    )


@app.post("/auth/logout")
def logout(x_session_id: Optional[str] = Header(None)):
    """Clear the session and conversation history."""
    if x_session_id and x_session_id in _sessions:
        _sessions.pop(x_session_id)
    return {"message": "Logged out successfully."}


@app.get("/auth/me")
def get_me(session_data: dict = Depends(_get_session)):
    """Get current logged-in employee's details."""
    emp = session_data["employee"]
    user_session = session_data["session"]
    return {
        "employee_id": emp["employee_id"],
        "name": emp["name"],
        "role": emp["role"],
        "department": emp["department_name"],
        "is_manager": user_session.is_manager(),
    }


# ── Chat endpoints ────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, session_data: dict = Depends(_get_session)):
    """
    Send a message to the HR agent. Returns the agent's response.
    Maintains conversation history within the session.
    The agent has access to all 18 tools and will route automatically.
    """
    # Set the global session to this user's session before invoking agent
    # This ensures tool calls (SQL, tickets, etc.) use the correct user
    from src.agent.session import current_session
    user_session = session_data["session"]
    current_session.current_employee_id = user_session.current_employee_id
    current_session.current_employee = user_session.current_employee

    conversation = session_data["conversation"]

    try:
        response = conversation.chat(req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    return ChatResponse(
        response=response,
        turn_count=conversation.turn_count,
    )


@app.get("/chat/history")
def get_history(session_data: dict = Depends(_get_session)):
    """Get full conversation history for the current session."""
    conversation = session_data["conversation"]
    return {
        "history": conversation.message_history,
        "turn_count": conversation.turn_count,
    }


@app.delete("/chat/history")
def clear_history(session_data: dict = Depends(_get_session)):
    """Clear conversation history (start fresh without logging out)."""
    session_data["conversation"].clear()
    return {"message": "Conversation history cleared."}


# ── Employee endpoints ────────────────────────────────────────────────────────

@app.get("/employee/profile")
def get_profile(session_data: dict = Depends(_get_session)):
    """Get current employee's full profile."""
    emp_id = session_data["session"].current_employee_id
    emp = employee_repo.get_employee_by_id(emp_id)
    return emp


@app.get("/employee/leave")
def get_leave(session_data: dict = Depends(_get_session)):
    """Get current employee's leave balances."""
    emp_id = session_data["session"].current_employee_id
    return employee_repo.get_leave_balance(emp_id)


@app.get("/employee/tickets")
def get_tickets(session_data: dict = Depends(_get_session)):
    """Get current employee's support ticket history."""
    from src.tools.ticket_tool import get_my_tickets_from_db
    from src.agent.session import current_session
    user_session = session_data["session"]
    current_session.current_employee_id = user_session.current_employee_id
    current_session.current_employee = user_session.current_employee
    return get_my_tickets_from_db()


# ── HR/Manager endpoints ──────────────────────────────────────────────────────

def _require_manager(session_data: dict):
    """Helper to enforce manager-only access on HR endpoints."""
    if not session_data["session"].is_manager():
        raise HTTPException(
            status_code=403,
            detail="Access denied. This endpoint requires manager or HR access."
        )


@app.get("/hr/tickets")
def get_hr_tickets(session_data: dict = Depends(_get_session)):
    """MANAGER/HR ONLY: Get all open support tickets sorted by priority."""
    _require_manager(session_data)
    return get_open_tickets_for_hr()


@app.get("/hr/jobs")
def get_jobs(session_data: dict = Depends(_get_session)):
    """Get all open job postings."""
    return _list_open_jobs()


@app.get("/hr/screenings")
def get_screenings(
    job_id: Optional[str] = None,
    session_data: dict = Depends(_get_session)
):
    """MANAGER/HR ONLY: Get resume screening results, optionally filtered by job_id."""
    _require_manager(session_data)
    import sqlite3
    conn = sqlite3.connect(os.path.join(config.BASE_DIR, "data", "hr.db"))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if job_id:
        cur.execute("""
            SELECT rs.*, jp.title as job_title
            FROM resume_screenings rs
            JOIN job_postings jp ON rs.job_id = jp.job_id
            WHERE rs.job_id = ?
            ORDER BY rs.match_score DESC
        """, (job_id,))
    else:
        cur.execute("""
            SELECT rs.*, jp.title as job_title
            FROM resume_screenings rs
            JOIN job_postings jp ON rs.job_id = jp.job_id
            ORDER BY rs.screened_at DESC
            LIMIT 20
        """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


@app.post("/hr/screen")
def screen_single(req: ScreenRequest, session_data: dict = Depends(_get_session)):
    """MANAGER/HR ONLY: Screen a single candidate's resume."""
    _require_manager(session_data)
    result = screen_resume(
        req.candidate_name, req.candidate_email,
        req.job_id, req.resume_text
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/hr/screen-batch")
def screen_multiple(req: BatchScreenRequest, session_data: dict = Depends(_get_session)):
    """MANAGER/HR ONLY: Screen multiple candidates and return ranked leaderboard."""
    _require_manager(session_data)
    candidates = [{"name": c.name, "email": c.email, "resume": c.resume}
                  for c in req.candidates]
    results = screen_batch(req.job_id, candidates)
    return {"job_id": req.job_id, "total": len(results), "ranked": results}


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick health check — confirms API is running."""
    return {
        "status": "ok",
        "model": config.LLM_MODEL,
        "reflection_model": config.REFLECTION_MODEL,
        "active_sessions": len(_sessions),
    }


if __name__ == "__main__":
    import uvicorn
    print(f"Starting HR Operations Assistant API...")
    print(f"Main model:       {config.LLM_MODEL}")
    print(f"Reflection model: {config.REFLECTION_MODEL}")
    print(f"Docs:             http://localhost:8000/docs")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
