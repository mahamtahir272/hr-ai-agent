"""
reflection.py — Self-critique layer using a separate smaller model.

TWO-MODEL ARCHITECTURE:
- Main agent: LLM_MODEL (llama-3.3-70b-versatile) — 100K tokens/day
- Reflection: REFLECTION_MODEL (llama-3.1-8b-instant) — 500K tokens/day (separate quota)

By splitting models, reflection doesn't eat into the main agent's scarce 70B budget.
The 8B model is reliable enough for scoring (simpler task than tool-calling).
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import config


def _get_reflection_llm():
    return ChatGroq(
        model=config.REFLECTION_MODEL,
        temperature=config.REFLECTION_TEMPERATURE,
        api_key=config.GROQ_API_KEY,
        max_retries=2,
    )


REFLECTION_PROMPT = ChatPromptTemplate.from_template("""
You are a quality reviewer for an HR assistant. Evaluate the draft answer below.

Employee question: {question}

Context used: {context}

Draft answer: {draft_answer}

Respond ONLY with a JSON object, no other text:
{{
  "confidence": <float 0.0-1.0>,
  "quality": "<excellent|good|acceptable|poor>",
  "reason": "<one sentence>",
  "should_escalate": <true|false>,
  "escalation_reason": "<reason or null>"
}}

Scoring: 0.8-1.0=excellent, 0.6-0.8=good, 0.4-0.6=acceptable, 0.0-0.4=poor.
Set should_escalate=true if confidence < 0.5 or question is outside HR scope
(salary disputes, legal matters, IT access issues, etc.)
""")


class ReflectionResult:
    def __init__(self, confidence, quality, reason, should_escalate, escalation_reason):
        self.confidence = confidence
        self.quality = quality
        self.reason = reason
        self.should_escalate = should_escalate
        self.escalation_reason = escalation_reason

    def __repr__(self):
        return (f"ReflectionResult(confidence={self.confidence}, "
                f"quality='{self.quality}', escalate={self.should_escalate})")


def reflect_on_answer(question, draft_answer, context="No specific context retrieved."):
    """Evaluate a draft answer using the 8B reflection model (separate quota from 70B agent)."""
    llm = _get_reflection_llm()
    prompt = REFLECTION_PROMPT.format(
        question=question,
        context=context[:1500],
        draft_answer=draft_answer,
    )
    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        return ReflectionResult(
            confidence=float(parsed.get("confidence", 0.5)),
            quality=parsed.get("quality", "acceptable"),
            reason=parsed.get("reason", "No reason provided."),
            should_escalate=bool(parsed.get("should_escalate", False)),
            escalation_reason=parsed.get("escalation_reason"),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return ReflectionResult(0.5, "acceptable", f"Reflection failed: {e}", False, None)


def format_reflection_footer(result):
    if result.should_escalate:
        return (f"\n\n---\n⚠️ Flagged for HR review (confidence: {result.confidence:.0%}). "
                f"A support ticket will be created automatically.")
    elif result.confidence >= 0.8:
        return f"\n\n---\n✓ Answer verified (confidence: {result.confidence:.0%})"
    else:
        return (f"\n\n---\nℹ️ Answer confidence: {result.confidence:.0%}. "
                f"If unsatisfied, contact HR directly.")


if __name__ == "__main__":
    print(f"Reflection model: {config.REFLECTION_MODEL} (8B — separate 500K/day quota)")
    print(f"Main agent model: {config.LLM_MODEL} (70B — 100K/day quota)\n")

    print("── Test 1: Good answer (should NOT escalate) ──")
    r1 = reflect_on_answer(
        "How many days of annual leave do I get?",
        "You are entitled to 18 days of paid annual leave per calendar year.",
        "ANNUAL LEAVE: All full-time employees are entitled to 18 days per calendar year."
    )
    print(r1)
    print(f"Escalate: {r1.should_escalate} | Reason: {r1.reason}")

    print("\n── Test 2: Out-of-scope (SHOULD escalate) ──")
    r2 = reflect_on_answer(
        "Why was my salary not credited last month?",
        "I was unable to find specific information about salary payment issues.",
        "No specific context retrieved."
    )
    print(r2)
    print(f"Escalate: {r2.should_escalate} | Reason: {r2.reason}")

    print("\n── Footers ──")
    print(format_reflection_footer(r1))
    print(format_reflection_footer(r2))
