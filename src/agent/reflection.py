"""
reflection.py — Self-critique layer that evaluates the agent's draft answer
before it reaches the user.

WHY THIS EXISTS: without reflection, the agent returns whatever answer it
generates, even if that answer is vague, unsupported, or outside its
knowledge. Reflection adds a second LLM call that acts as a critic —
it reads the question, the draft answer, and the source context, then
decides: is this answer good enough, or should it be escalated to HR?

This is what makes the ticket escalation genuinely agentic rather than
just a database write. The agent doesn't need to be told explicitly when
to escalate — the reflection step identifies weak answers automatically.

COST: reflection adds one extra LLM call per agent response that goes
through the RAG tool. SQL tool responses (personal data lookups) are
deterministic and don't need reflection — if the database returned a
leave balance, that balance is correct by definition.
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import config


# Reflection uses a smaller, faster model for the critic step to save tokens.
# The critic's job is simpler than the generator's — it just scores, doesn't create.
# We use the same model here but this is where you'd swap to a cheaper model
# in a production system with a paid tier.
def _get_reflection_llm():
    return ChatGroq(
        model=config.LLM_MODEL,
        temperature=0.0,      # zero temperature for consistent, deterministic scoring
        api_key=config.GROQ_API_KEY,
        max_retries=2,
    )


REFLECTION_PROMPT = ChatPromptTemplate.from_template("""
You are a quality reviewer for an HR assistant. Evaluate whether the draft
answer below adequately addresses the employee's question.

Employee question: {question}

Context used (from policy documents or employee data):
{context}

Draft answer: {draft_answer}

Evaluate the draft answer on these criteria:
1. Is it directly responsive to the question asked?
2. Is it fully supported by the context provided (no hallucinated facts)?
3. Is it specific enough to be actionable (not vague or evasive)?
4. Is the question within the HR assistant's scope (policy, leave, employee data)?

Respond ONLY with a JSON object, no other text:
{{
  "confidence": <float between 0.0 and 1.0>,
  "quality": "<excellent|good|acceptable|poor>",
  "reason": "<one sentence explaining the score>",
  "should_escalate": <true|false>,
  "escalation_reason": "<why escalation is needed, or null if not needed>"
}}

Scoring guide:
- 0.8-1.0: Answer is specific, well-supported, directly addresses the question
- 0.6-0.8: Answer is mostly good but slightly vague or missing a detail
- 0.4-0.6: Answer is partially responsive but misses key aspects
- 0.0-0.4: Answer is vague, unsupported, off-topic, or the question is outside scope

Set should_escalate to true if confidence < 0.5 OR the question is clearly
outside the HR assistant's scope (salary disputes, legal matters, IT issues, etc.)
""")


class ReflectionResult:
    """Structured result from the reflection step."""

    def __init__(self, confidence: float, quality: str, reason: str,
                 should_escalate: bool, escalation_reason: str | None):
        self.confidence = confidence
        self.quality = quality
        self.reason = reason
        self.should_escalate = should_escalate
        self.escalation_reason = escalation_reason

    def __repr__(self):
        return (f"ReflectionResult(confidence={self.confidence}, "
                f"quality='{self.quality}', escalate={self.should_escalate})")


def reflect_on_answer(
    question: str,
    draft_answer: str,
    context: str = "No specific context retrieved.",
) -> ReflectionResult:
    """
    Evaluate a draft answer using a second LLM call.

    Returns a ReflectionResult with:
    - confidence: 0-1 score of answer quality
    - quality: human-readable label
    - reason: one-line explanation
    - should_escalate: whether HR escalation is recommended
    - escalation_reason: why, if applicable

    This is called AFTER the agent generates a draft answer but BEFORE
    it's returned to the user.
    """
    llm = _get_reflection_llm()

    prompt = REFLECTION_PROMPT.format(
        question=question,
        context=context[:1500],  # trim context to avoid token bloat
        draft_answer=draft_answer,
    )

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown code fences if the model wraps JSON in them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        return ReflectionResult(
            confidence=float(parsed.get("confidence", 0.5)),
            quality=parsed.get("quality", "acceptable"),
            reason=parsed.get("reason", "No reason provided."),
            should_escalate=bool(parsed.get("should_escalate", False)),
            escalation_reason=parsed.get("escalation_reason"),
        )

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # If reflection itself fails, default to a cautious middle-ground result
        # rather than crashing — a failed reflection shouldn't break the whole pipeline
        return ReflectionResult(
            confidence=0.5,
            quality="acceptable",
            reason=f"Reflection parsing failed: {e}",
            should_escalate=False,
            escalation_reason=None,
        )


def format_reflection_footer(result: ReflectionResult) -> str:
    """
    Format a small footer to append to the agent's final answer,
    showing the reflection result. Visible to the user for transparency.
    In a production UI you might hide this or show it differently.
    """
    if result.should_escalate:
        return (
            f"\n\n---\n"
            f"⚠️ This answer has been flagged for HR review "
            f"(confidence: {result.confidence:.0%}). "
            f"A support ticket will be created automatically."
        )
    elif result.confidence >= 0.8:
        return f"\n\n---\n✓ Answer verified (confidence: {result.confidence:.0%})"
    else:
        return (
            f"\n\n---\n"
            f"ℹ️ Answer confidence: {result.confidence:.0%}. "
            f"If unsatisfied, you can ask HR directly."
        )


if __name__ == "__main__":
    # Quick test — run: python src/agent/reflection.py
    print("── Test 1: Good answer (should NOT escalate) ──")
    result1 = reflect_on_answer(
        question="How many days of annual leave do I get?",
        draft_answer="You are entitled to 18 days of paid annual leave per calendar year, accrued at 1.5 days per month.",
        context="1. ANNUAL LEAVE: All full-time employees are entitled to 18 days of paid annual leave per calendar year, accrued at a rate of 1.5 days per month."
    )
    print(result1)
    print(f"Escalate: {result1.should_escalate}")
    print(f"Reason: {result1.reason}")

    print("\n── Test 2: Out-of-scope question (SHOULD escalate) ──")
    result2 = reflect_on_answer(
        question="Why was my salary not credited last month?",
        draft_answer="I was unable to find specific information about salary payment issues in the available policy documents.",
        context="No specific context retrieved."
    )
    print(result2)
    print(f"Escalate: {result2.should_escalate}")
    print(f"Reason: {result2.reason}")

    print("\n── Reflection footers ──")
    print(format_reflection_footer(result1))
    print(format_reflection_footer(result2))
