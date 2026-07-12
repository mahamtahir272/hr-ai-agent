"""
rag_tool.py — Wraps the Chroma vector store + LLM into a single callable
function: ask a policy question, get back an answer with cited sources.

This is what the agent calls when a user asks about HR policies
(leave rules, code of conduct, benefits, etc.)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import config

# ── Load the vector store once, reused across calls ─────────────────────────
_embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
_vector_store = Chroma(
    persist_directory=config.CHROMA_DIR,
    embedding_function=_embeddings,
    collection_name="hr_policies",
)

_llm = ChatGroq(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE,
    api_key=config.GROQ_API_KEY,
)

_PROMPT = ChatPromptTemplate.from_template("""
You are an HR policy assistant. Answer the employee's question using ONLY
the context below. If the context does not contain the answer, say clearly
that you don't have that information rather than guessing.

Context from company policy documents:
{context}

Employee question: {question}

Answer concisely and accurately. Cite specific numbers/rules from the context
where relevant.
""")


def answer_policy_question(question: str, k: int = 4) -> dict:
    """
    Retrieve relevant policy chunks and generate a grounded answer.

    Returns:
        {
            "answer": str,
            "sources": list[str],          # filenames the answer came from
            "confidence": float,           # 0-1, based on retrieval score
            "retrieved_chunks": list[str]  # raw chunks, for debugging
        }
    """
    # Step 1: retrieve relevant chunks with similarity scores
    # Note: Chroma returns DISTANCE here (lower = more similar), not a 0-1
    # relevance score. We convert it below for an approximate confidence value.
    results = _vector_store.similarity_search_with_score(question, k=k)

    if not results:
        return {
            "answer": "I couldn't find any relevant policy information for that question.",
            "sources": [],
            "confidence": 0.0,
            "retrieved_chunks": [],
        }

    chunks = [doc.page_content for doc, score in results]
    sources = list(set(doc.metadata.get("source", "unknown") for doc, score in results))

    # Convert distance to an approximate 0-1 confidence (lower distance = higher confidence)
    # We use only the TOP result's distance, not an average across all k chunks.
    # Averaging dilutes the signal — if chunk #1 is a perfect match but chunk #4
    # is irrelevant noise, averaging makes a great retrieval look mediocre.
    # This is still a rough heuristic, not a calibrated probability — it reflects
    # retrieval similarity, NOT whether the final answer is actually correct.
    top_distance = results[0][1]
    approx_confidence = round(max(0.0, min(1.0, 1 - (top_distance / 2))), 2)

    # Step 2: build context and ask the LLM to answer using only that context
    context = "\n\n---\n\n".join(chunks)
    prompt = _PROMPT.format(context=context, question=question)
    response = _llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources,
        "confidence": approx_confidence,
        "retrieved_chunks": chunks,
    }


from langchain_core.tools import tool

@tool
def answer_hr_policy_question(question: str) -> str:
    """
    Answer HR policy questions (leave, conduct, benefits, dress code, notice period).
    Not for personal data like 'my balance'. Returns answer with source citations.
    """
    result = answer_policy_question(question)
    answer_with_sources = (
        f"{result['answer']}\n\n"
        f"(Source: {', '.join(result['sources'])} | "
        f"retrieval confidence: {result['confidence']})"
    )
    return answer_with_sources


if __name__ == "__main__":
    # Quick manual test — run: python src/tools/rag_tool.py
    test_questions = [
        "How many days of annual leave do I get?",
        "Can I work from home every day?",
        "What is the company's policy on harassment?",
        "What is the referral bonus amount?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        result = answer_policy_question(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Confidence: {result['confidence']}")
        print("-" * 60)
