"""
retrieval_eval.py — Objective, deterministic evaluation of RAG retrieval
accuracy.

WHAT THIS MEASURES: for each question, does the system retrieve a chunk
from the CORRECT source document? This is a fixed test set with known
correct answers — the result is a real number, not a subjective opinion.

This is intentionally narrow: it only checks "right document," not
"right specific sentence" or "is the final LLM answer correct." Checking
the exact right document is still a meaningful, honest signal of
retrieval quality without requiring manual chunk-level labeling.

Usage:
    python src/evals/retrieval_eval.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config


# ── Fixed test set — 20 questions, each with a known correct source file ───
# These were written by reading the actual policy documents, so the
# "expected_source" values are ground truth, not guesses.
TEST_CASES = [
    # leave_policy.txt
    {"question": "How many days of annual leave do I get?", "expected_source": "leave_policy.txt"},
    {"question": "What is the sick leave entitlement?", "expected_source": "leave_policy.txt"},
    {"question": "How long is maternity leave?", "expected_source": "leave_policy.txt"},
    {"question": "How many weeks of paternity leave can I take?", "expected_source": "leave_policy.txt"},
    {"question": "Can I work from home every day?", "expected_source": "leave_policy.txt"},
    {"question": "How many days of bereavement leave are given?", "expected_source": "leave_policy.txt"},
    {"question": "What happens if unpaid leave exceeds 15 days?", "expected_source": "leave_policy.txt"},
    {"question": "How many public holidays does the company observe?", "expected_source": "leave_policy.txt"},
    {"question": "Can unused annual leave be encashed?", "expected_source": "leave_policy.txt"},
    {"question": "Who approves my leave request first?", "expected_source": "leave_policy.txt"},

    # code_of_conduct.txt
    {"question": "What is the company's policy on harassment?", "expected_source": "code_of_conduct.txt"},
    {"question": "How do I report a POSH complaint?", "expected_source": "code_of_conduct.txt"},
    {"question": "Can I share confidential client data after I resign?", "expected_source": "code_of_conduct.txt"},
    {"question": "What is the dress code?", "expected_source": "code_of_conduct.txt"},
    {"question": "What are the standard working hours?", "expected_source": "code_of_conduct.txt"},
    {"question": "What happens if I install unauthorized software on my work laptop?", "expected_source": "code_of_conduct.txt"},
    {"question": "Am I protected if I report fraud as a whistleblower?", "expected_source": "code_of_conduct.txt"},

    # benefits_onboarding_policy.txt
    {"question": "What is the referral bonus amount?", "expected_source": "benefits_onboarding_policy.txt"},
    {"question": "How long is the probation period for new hires?", "expected_source": "benefits_onboarding_policy.txt"},
    {"question": "What is the notice period after confirmation?", "expected_source": "benefits_onboarding_policy.txt"},
]


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="hr_policies",
    )


def run_eval(vector_store, k: int = 2):
    """
    For each test case, retrieve top-k chunks and check if AT LEAST ONE
    came from the expected source document. Returns detailed results plus
    an overall accuracy score.
    """
    results = []
    correct_count = 0

    for case in TEST_CASES:
        question = case["question"]
        expected = case["expected_source"]

        retrieved = vector_store.similarity_search(question, k=k)
        retrieved_sources = [doc.metadata.get("source", "unknown") for doc in retrieved]

        is_correct = expected in retrieved_sources
        if is_correct:
            correct_count += 1

        results.append({
            "question": question,
            "expected_source": expected,
            "retrieved_sources": retrieved_sources,
            "correct": is_correct,
        })

    accuracy = correct_count / len(TEST_CASES)
    return results, accuracy


def print_report(results, accuracy):
    print("\n── Retrieval Evaluation Report ─────────────────────────────\n")

    failures = [r for r in results if not r["correct"]]

    for r in results:
        status = "✓ PASS" if r["correct"] else "✗ FAIL"
        print(f"{status}  {r['question']}")
        print(f"        expected: {r['expected_source']}")
        print(f"        got:      {r['retrieved_sources']}")
        print()

    print("──────────────────────────────────────────────────────────")
    print(f"RESULT: {sum(1 for r in results if r['correct'])}/{len(results)} correct "
          f"({accuracy*100:.1f}% retrieval accuracy)")

    if failures:
        print(f"\n{len(failures)} failure(s) to investigate:")
        for f in failures:
            print(f"  - \"{f['question']}\" → expected {f['expected_source']}, "
                  f"got {f['retrieved_sources']}")
    else:
        print("\nAll test cases passed.")
    print("──────────────────────────────────────────────────────────\n")


def main():
    print("Loading vector store...")
    vector_store = load_vector_store()

    print(f"Running {len(TEST_CASES)} test cases (k=2 chunks per query)...")
    results, accuracy = run_eval(vector_store, k=2)

    print_report(results, accuracy)


if __name__ == "__main__":
    main()
