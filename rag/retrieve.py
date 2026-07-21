"""
TradeGuard AI — RAG Layer (rule retrieval)
---------------------------------------------
Loads CBA rule chunks, builds a TF-IDF index (no internet needed), and
exposes retrieve_rules(query, k) -> ranked chunks with citations.

Swap-out note: this uses TF-IDF instead of a neural embedding model
because this dev environment has no internet access to download model
weights. The interface (retrieve_rules) is identical to what a
sentence-transformers or Azure OpenAI embeddings version would expose —
only build_index()'s internals would change. That's a deliberate
architecture choice so the swap is a one-file change later.

Run:  python3 rag/retrieve.py
"""

import csv
import pickle
import sqlite3
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
DB_PATH = BASE / "etl" / "tradeguard.db"
INDEX_PATH = Path(__file__).resolve().parent / "rule_index.pkl"


def load_chunks():
    with open(RAW / "cba_rules_seed.csv", newline="") as f:
        return list(csv.DictReader(f))


def build_index():
    """Load rule chunks into the DB and build a TF-IDF index over their text."""
    chunks = load_chunks()

    conn = sqlite3.connect(DB_PATH)
    for c in chunks:
        conn.execute(
            "INSERT OR REPLACE INTO rule_chunks (chunk_id, source_doc, section, topic, text, embedding) "
            "VALUES (?,?,?,?,?,NULL)",
            (c["chunk_id"], c["source_doc"], c["section"], c["topic"], c["text"]),
        )
    conn.commit()
    conn.close()

    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}, f)

    print(f"Indexed {len(chunks)} rule chunks -> {INDEX_PATH}")
    return vectorizer, matrix, chunks


def retrieve_rules(query: str, k: int = 3) -> list:
    """Tool: given a natural-language query, return the top-k most relevant rule chunks."""
    if not INDEX_PATH.exists():
        build_index()
    with open(INDEX_PATH, "rb") as f:
        idx = pickle.load(f)

    q_vec = idx["vectorizer"].transform([query])
    sims = cosine_similarity(q_vec, idx["matrix"])[0]
    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]

    results = []
    for i in ranked:
        c = idx["chunks"][i]
        results.append({
            "chunk_id": c["chunk_id"],
            "citation": f"{c['source_doc']} — {c['section']}",
            "topic": c["topic"],
            "text": c["text"],
            "relevance": round(float(sims[i]), 3),
        })
    return results


if __name__ == "__main__":
    build_index()

    test_queries = [
        "Can a team over the second apron combine two players in a trade?",
        "What happens if a team uses the mid-level exception?",
        "How much salary can a team take back if they are over the first apron?",
    ]
    for q in test_queries:
        print(f"\nQUERY: {q}")
        for r in retrieve_rules(q, k=2):
            print(f"  [{r['relevance']}] {r['citation']}: {r['text'][:110]}...")
