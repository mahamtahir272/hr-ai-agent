"""
resume_screening_tool.py — Improved resume screening with:
1. Multi-dimensional scoring (skills 50%, experience 30%, overall 20%)
2. Semantic skill matching via embeddings (catches synonyms)
3. Structured JSON output from LLM (no fragile keyword parsing)
4. Experience level check against JD requirements
5. Batch screening with ranked leaderboard
"""

import os
import sys
import sqlite3
import uuid
import json
import re
from datetime import datetime
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from src.agent.session import current_session
import config


# ── Module-level embedding model (loaded once, reused) ───────────────────────
_embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def _get_connection():
    conn = sqlite3.connect(os.path.join(config.BASE_DIR, "data", "hr.db"))
    conn.row_factory = sqlite3.Row
    return conn


def _cosine_similarity(vec_a, vec_b):
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Improvement 1: Multi-dimensional scoring ──────────────────────────────────
def _compute_multidim_score(jd: dict, resume_text: str) -> dict:
    """
    Compute three separate similarity scores and combine with weights.
    Skills match weighted highest since that's the most important signal.
    """
    # Build focused text sections for each dimension
    skills_jd_text = f"Required skills: {jd['skills_required']}"
    exp_jd_text = f"Responsibilities: {jd['responsibilities']} Experience: {jd['experience_min']}-{jd['experience_max']} years"
    overall_jd_text = f"{jd['title']} {jd['skills_required']} {jd['responsibilities']}"

    # Embed all at once to reuse compute
    vectors = _embeddings.embed_documents([
        skills_jd_text, exp_jd_text, overall_jd_text, resume_text
    ])

    skills_score = _cosine_similarity(vectors[0], vectors[3])
    exp_score = _cosine_similarity(vectors[1], vectors[3])
    overall_score = _cosine_similarity(vectors[2], vectors[3])

    # Weighted composite: skills 50%, experience fit 30%, overall 20%
    composite = (skills_score * 0.50) + (exp_score * 0.30) + (overall_score * 0.20)

    return {
        "skills_score": round(skills_score, 3),
        "experience_score": round(exp_score, 3),
        "overall_score": round(overall_score, 3),
        "composite_score": round(composite, 3),
    }


# ── Improvement 2: Semantic skill matching ────────────────────────────────────
def _semantic_skill_match(required_skills: list[str], resume_text: str, threshold: float = 0.45) -> dict:
    """
    Embed each required skill individually and find the most semantically
    similar sentence in the resume. Far more robust than string contains —
    catches 'Postgres' when JD says 'PostgreSQL', 'relational DB' for 'SQL', etc.
    """
    # Split resume into sentences for granular matching
    sentences = [s.strip() for s in re.split(r'[.\n]', resume_text) if len(s.strip()) > 10]
    if not sentences:
        sentences = [resume_text]

    # Embed all resume sentences once
    sentence_vecs = _embeddings.embed_documents(sentences)

    matched = []
    missing = []

    for skill in required_skills:
        skill = skill.strip()
        if not skill:
            continue

        # First try exact/substring match (fast)
        if skill.lower() in resume_text.lower():
            matched.append(skill)
            continue

        # Fallback: semantic similarity between skill and each resume sentence
        skill_vec = _embeddings.embed_query(skill)
        best_score = max(_cosine_similarity(skill_vec, sv) for sv in sentence_vecs)

        if best_score >= threshold:
            matched.append(f"{skill} (semantic)")
        else:
            missing.append(skill)

    return {"matched": matched, "missing": missing}


# ── Improvement 3: Structured JSON output from LLM ───────────────────────────
def _llm_structured_evaluation(jd: dict, candidate_name: str, resume_text: str,
                                 scores: dict, skills_matched: list, skills_missing: list,
                                 experience_fit: str) -> dict:
    """
    Ask the LLM to return structured JSON — no keyword parsing, no fragile text matching.
    """
    llm = ChatGroq(
        model=config.LLM_MODEL,
        temperature=0.1,
        api_key=config.GROQ_API_KEY,
        max_retries=2,
    )

    prompt = f"""You are a senior technical recruiter. Evaluate this candidate.

ROLE: {jd['title']} ({jd['department_name']})
Required Skills: {jd['skills_required']}
Experience: {jd['experience_min']}-{jd['experience_max']} years
Responsibilities: {jd['responsibilities']}

CANDIDATE: {candidate_name}
Resume: {resume_text}

SCORING CONTEXT:
- Skills match score: {scores['skills_score']:.0%}
- Experience match score: {scores['experience_score']:.0%}
- Composite score: {scores['composite_score']:.0%}
- Skills matched: {', '.join(skills_matched) or 'none'}
- Skills missing: {', '.join(skills_missing) or 'none'}
- Experience fit: {experience_fit}

Respond ONLY with a valid JSON object, no markdown, no explanation:
{{
  "recommendation": "Shortlist" or "Hold" or "Reject",
  "confidence": <float 0.0-1.0>,
  "strengths": ["<strength 1>", "<strength 2>"],
  "concerns": ["<concern 1>"],
  "summary": "<2-3 sentence overall assessment>",
  "interview_questions": ["<question 1>", "<question 2>", "<question 3>"]
}}

Shortlist: strong match on skills + experience. Hold: partial match, worth second look.
Reject: significant gaps in required skills or experience."""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if LLM doesn't return clean JSON
        rec = "Hold"
        if scores["composite_score"] >= 0.7:
            rec = "Shortlist"
        elif scores["composite_score"] < 0.45:
            rec = "Reject"
        return {
            "recommendation": rec,
            "confidence": scores["composite_score"],
            "strengths": ["Could not parse LLM evaluation"],
            "concerns": ["Evaluation parsing failed — review manually"],
            "summary": response.content[:300],
            "interview_questions": [],
        }


# ── Improvement 4: Experience level check ────────────────────────────────────
def _check_experience_fit(resume_text: str, exp_min: int, exp_max: int) -> str:
    """
    Extract years of experience from resume text and compare to JD requirements.
    Uses simple regex patterns — works for most standard resume formats.
    """
    patterns = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
        r'(\d+)\+?\s*yrs?\s+(?:of\s+)?experience',
        r'experience\s+of\s+(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s+\w+\s+experience',          # "3 years ML experience"
        r'(\d+)\+?\s*years?\s+\w+\s+\w+\s+experience',   # "3 years ML engineering experience"
        r'(\d{4})\s*[-–]\s*(?:present|current|now)',       # year ranges
    ]

    years_found = []
    for pattern in patterns:
        matches = re.findall(pattern, resume_text.lower())
        for m in matches:
            try:
                yr = int(m)
                if yr > 1900:  # it's a calendar year, not a duration
                    yr = datetime.now().year - yr
                if 0 < yr < 40:
                    years_found.append(yr)
            except ValueError:
                pass

    if not years_found:
        return f"Experience not clearly stated (JD requires {exp_min}-{exp_max} years)"

    max_exp = max(years_found)

    if max_exp < exp_min:
        return f"Under-experienced: ~{max_exp} years found, minimum {exp_min} required"
    elif max_exp > exp_max + 3:
        return f"Possibly overqualified: ~{max_exp} years found, role needs {exp_min}-{exp_max}"
    else:
        return f"Good fit: ~{max_exp} years experience matches {exp_min}-{exp_max} requirement"


def _get_job_posting(job_id: str) -> dict | None:
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT j.*, d.department_name
        FROM job_postings j
        JOIN departments d ON j.department_id = d.department_id
        WHERE j.job_id = ?
    """, (job_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def _save_screening_result(candidate_name, candidate_email, job_id, resume_text,
                            composite_score, skills_matched, skills_missing,
                            llm_evaluation, recommendation) -> str:
    conn = _get_connection()
    cur = conn.cursor()
    screening_id = f"RS{str(uuid.uuid4())[:8].upper()}"
    cur.execute("""
        INSERT INTO resume_screenings
        (screening_id, candidate_name, candidate_email, job_id, resume_text,
         match_score, skills_matched, skills_missing, llm_evaluation,
         recommendation, screened_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        screening_id, candidate_name, candidate_email, job_id, resume_text,
        round(composite_score, 3),
        ", ".join(skills_matched),
        ", ".join(skills_missing),
        llm_evaluation, recommendation,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()
    return screening_id


# ── Core screening function ───────────────────────────────────────────────────
def screen_resume(candidate_name: str, candidate_email: str,
                  job_id: str, resume_text: str) -> dict:
    """
    Full improved screening pipeline. Returns complete results dict.
    """
    job = _get_job_posting(job_id)
    if not job:
        return {"error": f"Job posting {job_id} not found."}

    required_skills = [s.strip() for s in job["skills_required"].split(",")]

    # 1. Multi-dimensional scoring
    scores = _compute_multidim_score(job, resume_text)

    # 2. Semantic skill matching
    skill_results = _semantic_skill_match(required_skills, resume_text)

    # 3. Experience level check
    experience_fit = _check_experience_fit(
        resume_text, job["experience_min"], job["experience_max"]
    )

    # 4. Structured LLM evaluation
    eval_result = _llm_structured_evaluation(
        job, candidate_name, resume_text,
        scores, skill_results["matched"], skill_results["missing"],
        experience_fit
    )

    recommendation = eval_result.get("recommendation", "Hold")
    llm_summary = json.dumps(eval_result, indent=2)

    # 5. Save to database
    screening_id = _save_screening_result(
        candidate_name, candidate_email, job_id, resume_text,
        scores["composite_score"],
        skill_results["matched"], skill_results["missing"],
        llm_summary, recommendation
    )

    return {
        "screening_id": screening_id,
        "candidate_name": candidate_name,
        "job_title": job["title"],
        "scores": scores,
        "experience_fit": experience_fit,
        "skills_matched": skill_results["matched"],
        "skills_missing": skill_results["missing"],
        "recommendation": recommendation,
        "confidence": eval_result.get("confidence", scores["composite_score"]),
        "strengths": eval_result.get("strengths", []),
        "concerns": eval_result.get("concerns", []),
        "summary": eval_result.get("summary", ""),
        "interview_questions": eval_result.get("interview_questions", []),
    }


# ── Improvement 5: Batch screening with ranked leaderboard ───────────────────
def screen_batch(job_id: str, candidates: list[dict]) -> list[dict]:
    """
    Screen multiple candidates for the same role and return ranked results.
    candidates = [{"name": str, "email": str, "resume": str}, ...]
    """
    results = []
    for c in candidates:
        result = screen_resume(c["name"], c["email"], job_id, c["resume"])
        if "error" not in result:
            results.append(result)

    # Sort by composite score descending
    results.sort(key=lambda x: x["scores"]["composite_score"], reverse=True)
    return results


def _format_screening_result(result: dict) -> str:
    """Format a screening result for display."""
    scores = result["scores"]
    return (
        f"Resume Screening — #{result['screening_id']}\n"
        f"Candidate:     {result['candidate_name']}\n"
        f"Role:          {result['job_title']}\n"
        f"Recommendation: {result['recommendation']}\n\n"
        f"MATCH SCORES (how well candidate fits the role):\n"
        f"  Skills match:     {scores['skills_score']:.0%}\n"
        f"  Experience match: {scores['experience_score']:.0%}\n"
        f"  Overall match:    {scores['overall_score']:.0%}\n"
        f"  Composite:        {scores['composite_score']:.0%}  ← primary signal\n\n"
        f"Experience fit: {result['experience_fit']}\n\n"
        f"Skills matched: {', '.join(result['skills_matched']) or 'None'}\n"
        f"Skills missing: {', '.join(result['skills_missing']) or 'None'}\n\n"
        f"Strengths:\n" +
        "\n".join(f"  • {s}" for s in result["strengths"]) +
        f"\n\nConcerns:\n" +
        "\n".join(f"  • {c}" for c in result["concerns"]) +
        f"\n\nSummary: {result['summary']}\n\n"
        f"Suggested interview questions:\n" +
        "\n".join(f"  {i+1}. {q}" for i, q in enumerate(result["interview_questions"]))
    )


# ── LangChain tool wrappers ───────────────────────────────────────────────────

@tool
def screen_candidate_resume(
    candidate_name: str,
    candidate_email: str,
    job_id: str,
    resume_text: str,
) -> str:
    """
    MANAGER/HR-ONLY: Screen one candidate's resume against a job posting.
    Uses multi-dimensional embedding scoring + structured LLM evaluation.
    Returns detailed scores, skills analysis, and Shortlist/Hold/Reject recommendation.
    """
    if (err := current_session.manager_check()):
        return err
    result = screen_resume(candidate_name, candidate_email, job_id, resume_text)
    if "error" in result:
        return result["error"]
    return _format_screening_result(result)


@tool
def screen_multiple_candidates(job_id: str, candidates_json: str) -> str:
    """
    MANAGER/HR-ONLY: Screen multiple candidates at once and return a ranked
    leaderboard. candidates_json must be a JSON array:
    [{"name": "...", "email": "...", "resume": "..."}, ...]
    Results are sorted by composite score — best match first.
    """
    if (err := current_session.manager_check()):
        return err

    try:
        candidates = json.loads(candidates_json)
    except json.JSONDecodeError:
        return "Invalid candidates_json format. Must be a valid JSON array."

    results = screen_batch(job_id, candidates)
    if not results:
        return "No candidates could be screened."

    lines = [f"Candidate Ranking — {len(results)} screened\n{'='*50}\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"#{i} {r['candidate_name']} — {r['recommendation']}\n"
            f"   Composite: {r['scores']['composite_score']:.0%} | "
            f"Skills: {r['scores']['skills_score']:.0%} | "
            f"Experience: {r['scores']['experience_score']:.0%}\n"
            f"   {r['experience_fit']}\n"
            f"   Matched: {', '.join(r['skills_matched']) or 'none'}\n"
            f"   Missing: {', '.join(r['skills_missing']) or 'none'}\n"
            f"   {r['summary']}\n"
        )
    return "\n".join(lines)


def _list_open_jobs() -> list[dict]:
    """Return all currently open job postings as dictionaries."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT j.job_id, j.title, d.department_name,
               j.experience_min, j.experience_max, j.skills_required
        FROM job_postings j
        JOIN departments d ON j.department_id = d.department_id
        WHERE j.status = 'open'
        ORDER BY j.posted_on DESC
    """)
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


@tool
def list_open_positions() -> str:
    """List all currently open job postings with skills and experience requirements."""
    rows = _list_open_jobs()
    if not rows:
        return "No open positions at this time."
    lines = [f"Open positions ({len(rows)} total):\n"]
    for j in rows:
        lines.append(
            f"  [{j['job_id']}] {j['title']} — {j['department_name']}\n"
            f"         {j['experience_min']}-{j['experience_max']} yrs | "
            f"{j['skills_required']}\n"
        )
    return "\n".join(lines)


@tool
def get_screening_results(job_id: str = "") -> str:
    """
    MANAGER/HR-ONLY: Get resume screening results sorted by score.
    Filter by job_id or leave empty for all recent screenings.
    """
    if (err := current_session.manager_check()):
        return err
    conn = _get_connection()
    cur = conn.cursor()
    if job_id:
        cur.execute("""
            SELECT rs.screening_id, rs.candidate_name, rs.candidate_email,
                   jp.title, rs.match_score, rs.recommendation,
                   rs.skills_matched, rs.skills_missing
            FROM resume_screenings rs
            JOIN job_postings jp ON rs.job_id = jp.job_id
            WHERE rs.job_id = ?
            ORDER BY rs.match_score DESC
        """, (job_id,))
    else:
        cur.execute("""
            SELECT rs.screening_id, rs.candidate_name, rs.candidate_email,
                   jp.title, rs.match_score, rs.recommendation,
                   rs.skills_matched, rs.skills_missing
            FROM resume_screenings rs
            JOIN job_postings jp ON rs.job_id = jp.job_id
            ORDER BY rs.screened_at DESC LIMIT 10
        """)
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return "No screening results found."
    lines = [f"Screening results ({len(rows)} found):\n"]
    for r in rows:
        lines.append(
            f"  [{r['screening_id']}] {r['candidate_name']} → {r['title']}\n"
            f"         Score: {r['match_score']:.0%} | {r['recommendation']}\n"
            f"         Matched: {r['skills_matched'] or 'none'}\n"
        )
    return "\n".join(lines)


SCREENING_TOOLS = [
    screen_candidate_resume,
    screen_multiple_candidates,
    list_open_positions,
    get_screening_results,
]


if __name__ == "__main__":
    print("Testing improved resume screening...\n")
    current_session.login("E009")

    print("── Test 1: List open positions ──")
    print(list_open_positions.invoke({}))

    print("\n── Test 2: Strong candidate for ML Engineer (J002) ──")
    r1 = screen_candidate_resume.invoke({
        "candidate_name": "Usman Tariq",
        "candidate_email": "usman@gmail.com",
        "job_id": "J002",
        "resume_text": (
            "3 years ML engineering experience. Expert in Python, PyTorch, "
            "HuggingFace transformers, and RAG pipeline development using LangChain. "
            "Built production document QA system with Chroma vector database. "
            "Published NLP research. LLM fine-tuning on domain-specific datasets. "
            "OpenAI API integration, prompt engineering, model evaluation."
        ),
    })
    print(r1)

    print("\n── Test 3: Weak candidate for same role ──")
    r2 = screen_candidate_resume.invoke({
        "candidate_name": "Random Applicant",
        "candidate_email": "random@gmail.com",
        "job_id": "J002",
        "resume_text": (
            "2 years data entry and Excel reporting. "
            "Basic Python scripting for file automation. "
            "Interested in learning AI and machine learning."
        ),
    })
    print(r2)

    print("\n── Test 4: Batch screening leaderboard ──")
    candidates = json.dumps([
        {
            "name": "Usman Tariq",
            "email": "usman@gmail.com",
            "resume": "3 years ML engineering. Python, PyTorch, HuggingFace, RAG, LangChain, vector databases, LLM fine-tuning."
        },
        {
            "name": "Amna Shah",
            "email": "amna@gmail.com",
            "resume": "2.5 years ML. Python, TensorFlow, some NLP work. Built chatbot using OpenAI API. Experience with data pipelines."
        },
        {
            "name": "Random Applicant",
            "email": "random@gmail.com",
            "resume": "Data entry, Excel, basic Python scripting. No ML experience."
        },
    ])
    print(screen_multiple_candidates.invoke({"job_id": "J002", "candidates_json": candidates}))

    print("\n── Test 5: View screening results ──")
    print(get_screening_results.invoke({"job_id": "J002"}))
