"""
setup_db.py — Run this ONCE to create and seed the entire HR database.
Creates hr.db in the data/ folder with all 10 tables and realistic synthetic data.

Usage:
    python src/database/setup_db.py
"""

import sqlite3
import os
import sys
from datetime import date, timedelta
import random

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH  = os.path.join(BASE_DIR, "data", "hr.db")
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def d(date_str): return date_str          # just for readability in seed data
def days_ago(n): return (date.today() - timedelta(days=n)).isoformat()
def days_from_now(n): return (date.today() + timedelta(days=n)).isoformat()


def create_tables(cur):
    """Create all 10 tables with foreign key constraints."""

    cur.executescript("""
    PRAGMA foreign_keys = ON;

    -- ── 1. DEPARTMENTS ────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS departments (
        department_id     TEXT PRIMARY KEY,
        department_name   TEXT NOT NULL,
        hod_employee_id   TEXT,           -- FK set after employees inserted
        created_at        TEXT DEFAULT (date('now'))
    );

    -- ── 2. EMPLOYEES ──────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS employees (
        employee_id       TEXT PRIMARY KEY,
        name              TEXT NOT NULL,
        email             TEXT UNIQUE NOT NULL,
        role              TEXT NOT NULL,
        department_id     TEXT NOT NULL REFERENCES departments(department_id),
        manager_id        TEXT REFERENCES employees(employee_id),
        date_of_joining   TEXT NOT NULL,
        employment_status TEXT NOT NULL DEFAULT 'active',   -- active | on_leave | resigned
        salary_band       TEXT NOT NULL,                    -- L1 | L2 | L3 | L4 | L5
        created_at        TEXT DEFAULT (date('now'))
    );

    -- ── 3. LEAVE TYPES ────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS leave_types (
        leave_type_id        TEXT PRIMARY KEY,
        leave_type_name      TEXT NOT NULL,
        annual_entitlement   INTEGER NOT NULL,
        carry_forward_limit  INTEGER NOT NULL DEFAULT 0,
        requires_document    INTEGER NOT NULL DEFAULT 0,    -- 0=false 1=true
        description          TEXT
    );

    -- ── 4. LEAVE BALANCES ─────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS leave_balances (
        balance_id        TEXT PRIMARY KEY,
        employee_id       TEXT NOT NULL REFERENCES employees(employee_id),
        leave_type_id     TEXT NOT NULL REFERENCES leave_types(leave_type_id),
        year              INTEGER NOT NULL,
        entitled_days     INTEGER NOT NULL,
        used_days         INTEGER NOT NULL DEFAULT 0,
        remaining_days    INTEGER NOT NULL,
        UNIQUE(employee_id, leave_type_id, year)
    );

    -- ── 5. LEAVE REQUESTS ─────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS leave_requests (
        request_id        TEXT PRIMARY KEY,
        employee_id       TEXT NOT NULL REFERENCES employees(employee_id),
        leave_type_id     TEXT NOT NULL REFERENCES leave_types(leave_type_id),
        start_date        TEXT NOT NULL,
        end_date          TEXT NOT NULL,
        total_days        INTEGER NOT NULL,
        reason            TEXT,
        status            TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|rejected
        approved_by       TEXT REFERENCES employees(employee_id),
        applied_on        TEXT DEFAULT (date('now')),
        actioned_on       TEXT
    );

    -- ── 6. PERFORMANCE REVIEWS ────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS performance_reviews (
        review_id         TEXT PRIMARY KEY,
        employee_id       TEXT NOT NULL REFERENCES employees(employee_id),
        reviewer_id       TEXT NOT NULL REFERENCES employees(employee_id),
        review_period     TEXT NOT NULL,    -- e.g. "H1-2025"
        rating            TEXT NOT NULL,    -- Exceptional|Exceeds|Meets|Below|PIP
        goals_met         INTEGER NOT NULL, -- percentage 0-100
        strengths         TEXT,
        areas_to_improve  TEXT,
        comments          TEXT,
        reviewed_on       TEXT NOT NULL
    );

    -- ── 7. JOB POSTINGS ───────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS job_postings (
        job_id            TEXT PRIMARY KEY,
        title             TEXT NOT NULL,
        department_id     TEXT NOT NULL REFERENCES departments(department_id),
        experience_min    INTEGER NOT NULL,  -- years
        experience_max    INTEGER NOT NULL,
        salary_band       TEXT NOT NULL,
        skills_required   TEXT NOT NULL,     -- comma-separated
        responsibilities  TEXT NOT NULL,
        status            TEXT NOT NULL DEFAULT 'open',  -- open|closed|on_hold
        posted_on         TEXT NOT NULL,
        closed_on         TEXT
    );

    -- ── 8. RESUME SCREENINGS ──────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS resume_screenings (
        screening_id      TEXT PRIMARY KEY,
        candidate_name    TEXT NOT NULL,
        candidate_email   TEXT NOT NULL,
        job_id            TEXT NOT NULL REFERENCES job_postings(job_id),
        resume_text       TEXT NOT NULL,
        match_score       REAL NOT NULL,         -- 0.0 to 1.0
        skills_matched    TEXT,                  -- comma-separated
        skills_missing    TEXT,                  -- comma-separated
        llm_evaluation    TEXT NOT NULL,         -- LLM's structured summary
        recommendation    TEXT NOT NULL,         -- Shortlist|Hold|Reject
        screened_at       TEXT DEFAULT (datetime('now'))
    );

    -- ── 9. NOTIFICATIONS ──────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS notifications (
        notification_id   TEXT PRIMARY KEY,
        employee_id       TEXT REFERENCES employees(employee_id),
        notification_type TEXT NOT NULL,   -- leave_approved|leave_rejected|
                                           -- onboarding|ticket_created|review_due
        recipient_email   TEXT NOT NULL,
        subject           TEXT NOT NULL,
        body              TEXT NOT NULL,
        status            TEXT NOT NULL DEFAULT 'sent',   -- sent|failed|mocked
        sent_at           TEXT DEFAULT (datetime('now'))
    );

    -- ── 10. TICKETS (escalations) ─────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS tickets (
        ticket_id         TEXT PRIMARY KEY,
        employee_id       TEXT REFERENCES employees(employee_id),
        query_text        TEXT NOT NULL,        -- what the employee asked
        agent_response    TEXT,                 -- what agent tried to answer
        confidence_score  REAL,                 -- agent's self-reported confidence
        category          TEXT,                 -- leave|policy|payroll|other
        priority          TEXT DEFAULT 'medium', -- low|medium|high
        status            TEXT DEFAULT 'open',  -- open|in_progress|resolved|closed
        assigned_to       TEXT REFERENCES employees(employee_id),
        resolution_notes  TEXT,
        created_at        TEXT DEFAULT (datetime('now')),
        resolved_at       TEXT
    );
    """)
    print("✓ All 10 tables created.")


def seed_departments(cur):
    departments = [
        ("D001", "Engineering"),
        ("D002", "Human Resources"),
        ("D003", "Sales"),
        ("D004", "Marketing"),
        ("D005", "Finance"),
        ("D006", "Product"),
        ("D007", "Operations"),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO departments (department_id, department_name) VALUES (?,?)",
        departments
    )
    print(f"✓ {len(departments)} departments seeded.")


def seed_employees(cur):
    # (employee_id, name, email, role, dept_id, manager_id, joining_date, status, salary_band)
    employees = [
        # ── Executive ──────────────────────────────────────────────────────
        ("E001", "Ayesha Noor",     "ayesha.noor@acmetech.com",     "Chief Operating Officer",    "D007", None,  "2017-04-01", "active", "L5"),
        ("E002", "Kamran Baig",     "kamran.baig@acmetech.com",     "Chief Technology Officer",   "D001", None,  "2016-11-15", "active", "L5"),

        # ── Engineering ────────────────────────────────────────────────────
        ("E003", "Tariq Aziz",      "tariq.aziz@acmetech.com",      "Engineering Manager",        "D001", "E002","2019-05-30", "active", "L4"),
        ("E004", "Aisha Khan",      "aisha.khan@acmetech.com",      "Senior Software Engineer",   "D001", "E003","2022-03-14", "active", "L3"),
        ("E005", "Bilal Raza",      "bilal.raza@acmetech.com",      "DevOps Engineer",            "D001", "E003","2023-09-18", "active", "L2"),
        ("E006", "Rohan Mehta",     "rohan.mehta@acmetech.com",     "Software Engineer",          "D001", "E003","2023-07-01", "active", "L2"),
        ("E007", "Omar Farooq",     "omar.farooq@acmetech.com",     "QA Engineer",                "D001", "E003","2024-02-29", "active", "L2"),
        ("E008", "Zara Hussain",    "zara.hussain@acmetech.com",    "Frontend Engineer",          "D001", "E003","2024-06-10", "active", "L1"),

        # ── HR ─────────────────────────────────────────────────────────────
        ("E009", "Mehwish Saleem",  "mehwish.saleem@acmetech.com",  "HR Director",                "D002", "E001","2018-02-14", "active", "L4"),
        ("E010", "Fatima Sheikh",   "fatima.sheikh@acmetech.com",   "HR Business Partner",        "D002", "E009","2021-11-20", "active", "L3"),
        ("E011", "Zainab Iqbal",    "zainab.iqbal@acmetech.com",    "Recruiter",                  "D002", "E009","2023-04-11", "active", "L2"),

        # ── Sales ──────────────────────────────────────────────────────────
        ("E012", "Adeel Hussain",   "adeel.hussain@acmetech.com",   "Sales Manager",              "D003", "E001","2020-10-12", "active", "L4"),
        ("E013", "Daniyal Ahmed",   "daniyal.ahmed@acmetech.com",   "Account Executive",          "D003", "E012","2024-01-10", "active", "L2"),
        ("E014", "Sana Mirza",      "sana.mirza@acmetech.com",      "Account Executive",          "D003", "E012","2024-03-15", "active", "L2"),

        # ── Marketing ──────────────────────────────────────────────────────
        ("E015", "Nida Malik",      "nida.malik@acmetech.com",      "Marketing Director",         "D004", "E001","2019-09-09", "active", "L4"),
        ("E016", "Sara Yousuf",     "sara.yousuf@acmetech.com",     "Marketing Manager",          "D004", "E015","2020-06-05", "active", "L3"),
        ("E017", "Ali Hassan",      "ali.hassan@acmetech.com",      "Content Strategist",         "D004", "E016","2023-01-23", "active", "L2"),

        # ── Finance ────────────────────────────────────────────────────────
        ("E018", "Kashif Javed",    "kashif.javed@acmetech.com",    "Finance Manager",            "D005", "E001","2021-01-25", "active", "L4"),
        ("E019", "Hina Tariq",      "hina.tariq@acmetech.com",      "Financial Analyst",          "D005", "E018","2022-08-22", "active", "L3"),
        ("E020", "Usman Qureshi",   "usman.qureshi@acmetech.com",   "Accounts Executive",         "D005", "E018","2024-05-01", "active", "L2"),

        # ── Product ────────────────────────────────────────────────────────
        ("E021", "Rabia Asif",      "rabia.asif@acmetech.com",      "Product Manager",            "D006", "E002","2021-07-19", "active", "L4"),
        ("E022", "Hamza Butt",      "hamza.butt@acmetech.com",      "Associate Product Manager",  "D006", "E021","2023-10-02", "active", "L2"),

        # ── Operations ─────────────────────────────────────────────────────
        ("E023", "Saima Riaz",      "saima.riaz@acmetech.com",      "Operations Lead",            "D007", "E001","2020-03-08", "active", "L3"),
        ("E024", "Faisal Chaudhry", "faisal.chaudhry@acmetech.com", "Office Administrator",       "D007", "E023","2022-12-01", "active", "L2"),

        # ── On leave / resigned (for realism) ─────────────────────────────
        ("E025", "Maria Abbasi",    "maria.abbasi@acmetech.com",    "Software Engineer",          "D001", "E003","2022-05-16", "on_leave", "L2"),
        ("E026", "Junaid Malik",    "junaid.malik@acmetech.com",    "Sales Executive",            "D003", "E012","2023-02-20", "resigned","L1"),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO employees
        (employee_id,name,email,role,department_id,manager_id,date_of_joining,employment_status,salary_band)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, employees)

    # Update HODs in departments
    hod_map = [
        ("E002","D001"), ("E009","D002"), ("E012","D003"),
        ("E015","D004"), ("E018","D005"), ("E021","D006"), ("E001","D007"),
    ]
    cur.executemany(
        "UPDATE departments SET hod_employee_id=? WHERE department_id=?", hod_map
    )
    print(f"✓ {len(employees)} employees seeded.")


def seed_leave_types(cur):
    leave_types = [
        ("LT001", "Annual Leave",     18, 6, 0, "Paid leave for personal use. 5 days advance notice required."),
        ("LT002", "Sick Leave",       10, 0, 1, "Paid sick leave. Medical certificate required for 3+ consecutive days."),
        ("LT003", "Maternity Leave",  182,0, 0, "26 weeks fully paid maternity leave."),
        ("LT004", "Paternity Leave",  14, 0, 0, "2 weeks fully paid paternity leave within 3 months of birth."),
        ("LT005", "Bereavement Leave",5,  0, 0, "Immediate family only: spouse, parent, child, sibling."),
        ("LT006", "Unpaid Leave",     30, 0, 0, "Subject to manager and HR approval. Over 15 days needs VP sign-off."),
        ("LT007", "Work From Home",   104,0, 0, "Up to 2 days per week. Manager approval required."),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO leave_types
        (leave_type_id,leave_type_name,annual_entitlement,carry_forward_limit,requires_document,description)
        VALUES (?,?,?,?,?,?)
    """, leave_types)
    print(f"✓ {len(leave_types)} leave types seeded.")


def seed_leave_balances(cur):
    # Active employees get balances for 2025
    active_employees = [
        "E001","E002","E003","E004","E005","E006","E007","E008",
        "E009","E010","E011","E012","E013","E014","E015","E016",
        "E017","E018","E019","E020","E021","E022","E023","E024","E025"
    ]
    # (leave_type_id, entitlement)
    standard_leaves = [("LT001",18), ("LT002",10), ("LT006",30)]

    balances = []
    bid = 1
    random.seed(42)  # reproducible
    for emp in active_employees:
        for lt_id, entitled in standard_leaves:
            used = random.randint(0, min(entitled, 12))
            remaining = entitled - used
            balances.append((
                f"B{bid:04d}", emp, lt_id, 2025,
                entitled, used, remaining
            ))
            bid += 1

    # Add maternity leave for E025 (currently on leave)
    balances.append(("B9901","E025","LT003",2025,182,90,92))

    cur.executemany("""
        INSERT OR IGNORE INTO leave_balances
        (balance_id,employee_id,leave_type_id,year,entitled_days,used_days,remaining_days)
        VALUES (?,?,?,?,?,?,?)
    """, balances)
    print(f"✓ {len(balances)} leave balance records seeded.")


def seed_leave_requests(cur):
    requests = [
        # approved requests
        ("LR001","E004","LT001",days_ago(60), days_ago(55), 5, "Family vacation",        "approved","E003",days_ago(62),days_ago(61)),
        ("LR002","E006","LT002",days_ago(30), days_ago(28), 3, "Fever and flu",           "approved","E003",days_ago(31),days_ago(30)),
        ("LR003","E013","LT001",days_ago(20), days_ago(16), 5, "Wedding ceremony",        "approved","E012",days_ago(22),days_ago(21)),
        ("LR004","E019","LT001",days_ago(10), days_ago(8),  3, "Personal work",           "approved","E018",days_ago(12),days_ago(11)),
        ("LR005","E017","LT002",days_ago(5),  days_ago(4),  2, "Not feeling well",        "approved","E016",days_ago(6), days_ago(5)),
        ("LR006","E008","LT001",days_ago(45), days_ago(41), 5, "Eid holidays",            "approved","E003",days_ago(47),days_ago(46)),
        ("LR007","E011","LT005",days_ago(90), days_ago(86), 5, "Bereavement",             "approved","E009",days_ago(91),days_ago(91)),
        ("LR008","E022","LT004",days_ago(15), days_ago(2),  14,"Paternity leave",         "approved","E021",days_ago(20),days_ago(19)),
        ("LR009","E005","LT001",days_ago(70), days_ago(66), 5, "Travelling abroad",       "approved","E003",days_ago(72),days_ago(71)),
        ("LR010","E020","LT002",days_ago(3),  days_ago(2),  2, "Migraine",                "approved","E018",days_ago(4), days_ago(3)),

        # pending requests (agent can act on these)
        ("LR011","E007","LT001",days_from_now(5), days_from_now(9),  5, "Personal trip",  "pending", None,  days_ago(1), None),
        ("LR012","E014","LT002",days_from_now(1), days_from_now(2),  2, "Doctor visit",   "pending", None,  days_ago(0), None),
        ("LR013","E017","LT001",days_from_now(10),days_from_now(14), 5, "Vacation",       "pending", None,  days_ago(0), None),

        # rejected request (for realism)
        ("LR014","E006","LT001",days_ago(40), days_ago(36), 5, "Holiday",                 "rejected","E003",days_ago(42),days_ago(41)),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO leave_requests
        (request_id,employee_id,leave_type_id,start_date,end_date,total_days,
         reason,status,approved_by,applied_on,actioned_on)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, requests)
    print(f"✓ {len(requests)} leave requests seeded.")


def seed_performance_reviews(cur):
    reviews = [
        ("PR001","E004","E003","H1-2025","Exceeds Expectations",    88,"Strong problem-solving, good mentoring",    "Should improve documentation",         "Aisha has shown consistent delivery this half.",       days_ago(30)),
        ("PR002","E005","E003","H1-2025","Meets Expectations",      75,"Reliable infra work, good on-call record", "Needs to improve communication in standups","Bilal keeps the infra stable. Communication can improve.",days_ago(30)),
        ("PR003","E006","E003","H1-2025","Meets Expectations",      70,"Completes assigned tickets on time",        "Needs to take ownership of larger features","Rohan is consistent but plays it safe.",               days_ago(30)),
        ("PR004","E007","E003","H1-2025","Below Expectations",      55,"Thorough in testing",                       "Speed of test case delivery needs work","Omar needs to pick up velocity significantly.",        days_ago(30)),
        ("PR005","E010","E009","H1-2025","Exceeds Expectations",    90,"Excellent employee relations, empathetic",  "Could delegate more to junior HR staff","Fatima is a cornerstone of the HR team.",             days_ago(28)),
        ("PR006","E013","E012","H1-2025","Exceptional",             95,"Highest sales in Q1, client retention 98%","Work-life balance concerns raised",    "Daniyal has exceeded every target set.",               days_ago(25)),
        ("PR007","E016","E015","H1-2025","Meets Expectations",      72,"Good campaign execution",                   "Data-driven decision making needs work","Sara runs solid campaigns but needs more analytics.", days_ago(22)),
        ("PR008","E019","E018","H1-2025","Exceeds Expectations",    85,"Accurate forecasting, attention to detail", "Presentation skills to leadership",    "Hina's financial models have been highly accurate.",   days_ago(20)),
        ("PR009","E021","E002","H1-2025","Exceptional",             92,"Clear product vision, cross-team alignment","Could involve engineering earlier",    "Rabia is driving the product roadmap excellently.",    days_ago(18)),
        ("PR010","E008","E003","H1-2025","Meets Expectations",      68,"Fast learner, good UI work",                "Needs more testing discipline",        "Zara is new but showing good progress.",               days_ago(15)),

        # Older reviews (H2-2024) for history
        ("PR011","E004","E003","H2-2024","Meets Expectations",      78,"Good delivery",                             "Communication with stakeholders",       "Solid half for Aisha.",                               days_ago(210)),
        ("PR012","E013","E012","H2-2024","Exceeds Expectations",    87,"Strong pipeline",                          "Needs to document sales calls better",  "Daniyal is on a strong trajectory.",                  days_ago(205)),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO performance_reviews
        (review_id,employee_id,reviewer_id,review_period,rating,goals_met,
         strengths,areas_to_improve,comments,reviewed_on)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, reviews)
    print(f"✓ {len(reviews)} performance reviews seeded.")


def seed_job_postings(cur):
    postings = [
        ("J001","Senior Backend Engineer",  "D001",4,7,"L3",
         "Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, REST APIs",
         "Design and build scalable backend services. Lead code reviews. Mentor junior engineers. Collaborate with product on technical feasibility.",
         "open", days_ago(15), None),

        ("J002","ML Engineer",              "D001",2,5,"L3",
         "Python, LangChain, OpenAI API, RAG, Vector Databases, PyTorch, HuggingFace",
         "Build and maintain LLM-powered features. Implement RAG pipelines. Fine-tune models. Evaluate model performance and set up monitoring.",
         "open", days_ago(10), None),

        ("J003","HR Business Partner",      "D002",3,6,"L3",
         "HR Operations, HRIS, Employee Relations, Performance Management, Talent Acquisition",
         "Partner with business units on HR strategy. Handle employee grievances. Drive performance review cycles. Support talent acquisition.",
         "open", days_ago(20), None),

        ("J004","Account Executive",        "D003",1,3,"L2",
         "B2B Sales, CRM, Salesforce, Negotiation, Client Relationship Management",
         "Manage a portfolio of SMB accounts. Meet quarterly revenue targets. Run discovery calls and product demos. Maintain pipeline hygiene in CRM.",
         "open", days_ago(8), None),

        ("J005","Product Manager",          "D006",3,6,"L4",
         "Product Roadmap, Agile, JIRA, User Research, Data Analysis, Stakeholder Management",
         "Own the product roadmap for a core feature area. Work with engineering and design. Define success metrics. Run sprint planning.",
         "on_hold", days_ago(30), None),

        ("J006","DevOps Engineer",          "D001",2,4,"L2",
         "AWS, Terraform, CI/CD, Docker, Kubernetes, Linux, Bash, Monitoring",
         "Manage cloud infrastructure. Build and maintain CI/CD pipelines. Implement observability. Reduce deployment friction for engineering teams.",
         "closed", days_ago(60), days_ago(5)),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO job_postings
        (job_id,title,department_id,experience_min,experience_max,salary_band,
         skills_required,responsibilities,status,posted_on,closed_on)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, postings)
    print(f"✓ {len(postings)} job postings seeded.")


def seed_resume_screenings(cur):
    screenings = [
        ("RS001","Ahmed Siddiqui",  "ahmed.siddiqui@gmail.com",  "J001",
         "5 years backend experience. Python, FastAPI, PostgreSQL, Docker. Led migration of monolith to microservices at previous company. B.Sc Computer Science FAST NUCES.",
         0.87,
         "Python, FastAPI, PostgreSQL, Docker, REST APIs",
         "Redis, Kubernetes",
         "Strong match. Candidate has directly relevant experience with the core stack. Microservices migration experience is a strong plus. Missing Redis and Kubernetes but these are learnable.",
         "Shortlist", days_ago(5)),

        ("RS002","Priya Sharma",    "priya.sharma@outlook.com",  "J001",
         "3 years backend development. Django, MySQL, AWS basics. Some Docker experience. Currently working at a startup.",
         0.52,
         "Python, Docker",
         "FastAPI, PostgreSQL, Redis, Kubernetes, REST APIs",
         "Partial match. Candidate knows Python but uses Django not FastAPI, MySQL not PostgreSQL. AWS is listed but not detailed. Experience level is below the 4-year minimum.",
         "Hold", days_ago(4)),

        ("RS003","Usman Tariq",     "usman.tariq@hotmail.com",   "J002",
         "2.5 years in ML. Experience with PyTorch, HuggingFace, fine-tuning LLMs. Built a RAG system for document QA at previous role. Published 1 paper on NLP.",
         0.91,
         "Python, PyTorch, HuggingFace, RAG, LLM fine-tuning",
         "LangChain, Vector Databases",
         "Excellent match. Direct RAG experience is highly relevant. LLM fine-tuning background is a strong differentiator. LangChain and vector DB exposure missing but candidate clearly has the foundation.",
         "Shortlist", days_ago(3)),

        ("RS004","Fatima Zahra",    "fatima.z@gmail.com",        "J003",
         "4 years in HR. Experience with employee relations, performance management, HRIS (BambooHR). Handled a team of 200 employees. MBA HR from IBA.",
         0.83,
         "HR Operations, Employee Relations, Performance Management, HRIS",
         "Talent Acquisition",
         "Strong match for the HRBP role. Hands-on experience with a similar-sized employee base. HRIS experience is directly relevant. Talent acquisition experience is weaker but not a dealbreaker.",
         "Shortlist", days_ago(2)),

        ("RS005","Zaid Khan",       "zaid.khan@gmail.com",       "J004",
         "1 year sales experience. Fresh MBA. Some CRM exposure during internship at a FMCG company. No B2B experience.",
         0.38,
         "CRM",
         "B2B Sales, Salesforce, Negotiation, Client Relationship Management",
         "Weak match. Candidate is early-career with no B2B experience. FMCG sales context is quite different from the target role. Could be considered for a junior pipeline if one opens.",
         "Reject", days_ago(1)),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO resume_screenings
        (screening_id,candidate_name,candidate_email,job_id,resume_text,
         match_score,skills_matched,skills_missing,llm_evaluation,recommendation,screened_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, screenings)
    print(f"✓ {len(screenings)} resume screenings seeded.")


def seed_notifications(cur):
    notifications = [
        ("N001","E004","leave_approved",   "aisha.khan@acmetech.com",
         "Your Annual Leave request has been approved",
         "Hi Aisha, your leave from 5 days has been approved by Tariq Aziz. Enjoy your time off!",
         "sent", days_ago(61)),
        ("N002","E006","leave_approved",   "rohan.mehta@acmetech.com",
         "Your Sick Leave request has been approved",
         "Hi Rohan, your sick leave for 3 days has been approved. Please rest and recover.",
         "sent", days_ago(30)),
        ("N003","E006","leave_rejected",   "rohan.mehta@acmetech.com",
         "Your Annual Leave request was not approved",
         "Hi Rohan, your leave request for 5 days was not approved due to a critical release. Please reapply for a different date.",
         "sent", days_ago(41)),
        ("N004","E008","onboarding",       "zara.hussain@acmetech.com",
         "Welcome to Acme Technologies — Your first day details",
         "Hi Zara, welcome aboard! Your employee ID is E008. IT setup is on Day 1 at 10am. Your buddy is Aisha Khan.",
         "sent", days_ago(180)),
        ("N005","E022","leave_approved",   "hamza.butt@acmetech.com",
         "Your Paternity Leave request has been approved",
         "Hi Hamza, your paternity leave for 14 days has been approved. Congratulations!",
         "sent", days_ago(19)),
        ("N006","E007","ticket_created",   "omar.farooq@acmetech.com",
         "Your HR query has been escalated — Ticket #T003",
         "Hi Omar, your question could not be answered automatically. A ticket has been raised and HR will respond within 2 business days.",
         "sent", days_ago(7)),
        ("N007","E013","leave_approved",   "daniyal.ahmed@acmetech.com",
         "Your Annual Leave request has been approved",
         "Hi Daniyal, your leave for 5 days has been approved by Adeel Hussain.",
         "sent", days_ago(21)),
        ("N008",None,  "review_due",       "tariq.aziz@acmetech.com",
         "Reminder: H2-2025 Performance Reviews due in 2 weeks",
         "Hi Tariq, the H2-2025 review cycle opens in 2 weeks. Please ensure your team completes self-assessments by the deadline.",
         "sent", days_ago(14)),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO notifications
        (notification_id,employee_id,notification_type,recipient_email,
         subject,body,status,sent_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, notifications)
    print(f"✓ {len(notifications)} notifications seeded.")


def seed_tickets(cur):
    tickets = [
        ("T001","E006","How do I apply for unpaid leave if I have exhausted all paid leave?",
         "Unpaid leave can be applied through the HR portal. You need manager and HR approval. Requests over 15 days need VP sign-off.",
         0.82,"leave","low","resolved","E010",
         "Answered via policy document. Employee confirmed resolution.",
         days_ago(25), days_ago(24)),

        ("T002","E019","Can I get my salary slip for the last 6 months for a bank loan application?",
         "I was unable to find specific information about salary slip requests in the available documents.",
         0.21,"payroll","high","resolved","E010",
         "HR shared salary slips directly with employee via secure email.",
         days_ago(18), days_ago(16)),

        ("T003","E007","I believe my performance rating was unfair. What is the appeal process?",
         "I found information about the performance review process but could not find specific details about an appeal process.",
         0.30,"policy","high","in_progress","E009",
         None,
         days_ago(7), None),

        ("T004","E014","What is the process for claiming the referral bonus I submitted 4 months ago?",
         "The referral bonus policy states it is paid after 3 months of the referred candidate's employment. I could not verify the specific referral status.",
         0.45,"payroll","medium","open","E010",
         None,
         days_ago(3), None),

        ("T005","E017","Is there a budget for attending an external AI conference next month?",
         "Employees are entitled to INR 20,000 annual learning budget subject to manager approval per the benefits policy.",
         0.78,"policy","low","resolved","E010",
         "Confirmed via policy. Employee directed to submit through manager approval flow.",
         days_ago(10), days_ago(9)),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO tickets
        (ticket_id,employee_id,query_text,agent_response,confidence_score,
         category,priority,status,assigned_to,resolution_notes,created_at,resolved_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, tickets)
    print(f"✓ {len(tickets)} tickets seeded.")


def print_summary(cur):
    tables = [
        "departments","employees","leave_types","leave_balances",
        "leave_requests","performance_reviews","job_postings",
        "resume_screenings","notifications","tickets"
    ]
    print("\n── Database summary ──────────────────────────────")
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        count = cur.fetchone()[0]
        print(f"  {t:<25} {count:>3} rows")
    print(f"\n  Database saved to: {DB_PATH}")
    print("──────────────────────────────────────────────────")


def main():
    if os.path.exists(DB_PATH):
        print(f"Database already exists at {DB_PATH}")
        answer = input("Recreate from scratch? This will delete all data. (y/n): ").strip().lower()
        if answer != "y":
            print("Aborted. Existing database kept.")
            sys.exit(0)
        os.remove(DB_PATH)
        print("Old database removed.")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    print("\nCreating tables...")
    create_tables(cur)

    print("\nSeeding data...")
    seed_departments(cur)
    seed_employees(cur)
    seed_leave_types(cur)
    seed_leave_balances(cur)
    seed_leave_requests(cur)
    seed_performance_reviews(cur)
    seed_job_postings(cur)
    seed_resume_screenings(cur)
    seed_notifications(cur)
    seed_tickets(cur)

    conn.commit()
    conn.close()

    print_summary(sqlite3.connect(DB_PATH).cursor())
    print("\n✓ Database setup complete. You can now run the agent.\n")


if __name__ == "__main__":
    main()
