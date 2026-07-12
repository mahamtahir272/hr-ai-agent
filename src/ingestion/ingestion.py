"""
ingestion.py — Loads HR policy documents, chunks them, embeds them using a
free local model, and stores them in a Chroma vector database.

Run this once to build the vector store. Re-run anytime you add/change
documents in data/docs/ — it rebuilds the index from scratch.

Usage:
    python src/ingestion/ingestion.py
"""

import os
import sys

# Allow importing config.py from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config


def load_documents():
    """Load every .txt file in data/docs/ and attach metadata (filename)."""
    documents = []
    docs_dir = config.DOCS_DIR

    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    txt_files = [f for f in os.listdir(docs_dir) if f.endswith(".txt")]
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {docs_dir}")

    print(f"Found {len(txt_files)} document(s) to ingest:")
    for filename in txt_files:
        filepath = os.path.join(docs_dir, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        loaded = loader.load()

        # Strip header/title lines (e.g. "LEAVE POLICY - ACME...", "Effective
        # Date:", "Document Version:") so they never become their own
        # meaningless chunk that pollutes retrieval results.
        for doc in loaded:
            lines = doc.page_content.split("\n")
            cleaned_lines = [
                line for line in lines
                if not (
                    line.strip().endswith("LTD")
                    or line.strip().startswith("Effective Date:")
                    or line.strip().startswith("Document Version:")
                )
            ]
            doc.page_content = "\n".join(cleaned_lines).strip()

            # Tag each document with a clean source name for citations later
            doc.metadata["source"] = filename
            doc.metadata["doc_type"] = "hr_policy"

        documents.extend(loaded)
        print(f"  ✓ {filename}  ({len(loaded[0].page_content)} characters)")

    return documents


def split_documents(documents):
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # tries paragraph first, then sentence
    )
    chunks = splitter.split_documents(documents)

    # Add a chunk index to metadata — useful for debugging retrieval later
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"\nSplit into {len(chunks)} chunks "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks):
    """Embed chunks with a free local model and persist to Chroma."""
    print(f"\nLoading embedding model: {config.EMBEDDING_MODEL}")
    print("(First run downloads the model — may take 1-2 minutes)")

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    print("\nEmbedding chunks and writing to Chroma...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR,
        collection_name="hr_policies",
    )

    print(f"✓ Vector store saved to: {config.CHROMA_DIR}")
    return vector_store


def test_retrieval(vector_store):
    """Run a few sample queries to sanity-check retrieval quality."""
    test_queries = [
        "How many days of annual leave do I get?",
        "What is the maternity leave policy?",
        "What happens if I take unpaid leave for more than 15 days?",
        "What is the dress code?",
    ]

    print("\n── Testing retrieval with sample queries ──────────────")
    for query in test_queries:
        results = vector_store.similarity_search(query, k=2)
        print(f"\nQuery: {query}")
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"  [{i}] from {source}: \"{preview}...\"")
    print("\n────────────────────────────────────────────────────────")


def main():
    print("Starting ingestion pipeline...\n")

    documents = load_documents()
    chunks = split_documents(documents)
    vector_store = build_vector_store(chunks)
    test_retrieval(vector_store)

    print("\n✓ Ingestion complete. The RAG knowledge base is ready.")


if __name__ == "__main__":
    main()