# backend/rag_engine.py
"""
RAG Engine (Retrieval-Augmented Generation)
-------------------------------------------
1. build_index_from_file()   → process a document, add chunks to FAISS
2. load_index()              → load persisted FAISS index at Flask startup
3. retrieve()                → find top-k relevant chunks for a query
4. is_confident()            → check if retrieval score meets the threshold
5. generate_answer()         → generate a grounded LLM response
"""

import os
import json
import numpy as np
import faiss
from typing import List, Tuple
from openai import OpenAI

from config import (
    GROQ_API_KEY,
    CHAT_MODEL,
    MAX_TOKENS,
    FAISS_INDEX_DIR,
    CHUNKS_METADATA_FILE,
    TOP_K_CHUNKS,
    CONFIDENCE_THRESHOLD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BUSINESS_NAME,
    BUSINESS_HOURS,
)
from document_processor import extract_text, chunk_text
import re
import math

# ─── Module-level state ───────────────────────────────────────────────────────
# The FAISS index and chunk list live in memory for the lifetime of the Flask process.
# They are loaded from disk at startup and updated in memory when new documents are added.

_client: OpenAI     = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
_index:  faiss.Index = None       # The FAISS vector index (None until first document)
_chunks: List[str]   = []         # Text chunks — parallel to index vectors

# TF-IDF variables for local, offline embeddings
TFIDF_METADATA_FILE  = os.path.join(FAISS_INDEX_DIR, "tfidf_metadata.json")
_tfidf_vocab: dict   = {}         # Term to index mapping
_tfidf_idf: dict     = {}         # Term to IDF score mapping


# ─── Public API ───────────────────────────────────────────────────────────────

def load_index():
    """
    Load a previously saved FAISS index from disk.
    Must be called once at Flask startup (before the first request arrives).
    If no index exists yet (no documents uploaded), this is a no-op.
    """
    global _index, _chunks, _tfidf_vocab, _tfidf_idf

    index_path = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    if not os.path.exists(index_path):
        print("[RAG] No saved index found. Upload a document to create one.")
        return

    # Clear old OpenAI-dimensional index if TF-IDF metadata doesn't exist
    if not os.path.exists(TFIDF_METADATA_FILE):
        print("[RAG] Old index format detected (missing TF-IDF metadata). Deleting old files to trigger rebuilding...")
        try:
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(CHUNKS_METADATA_FILE):
                os.remove(CHUNKS_METADATA_FILE)
        except Exception as e:
            print(f"[RAG] Error deleting old index files: {e}")
        return

    _index = faiss.read_index(index_path)
    with open(CHUNKS_METADATA_FILE, "r", encoding="utf-8") as f:
        _chunks = json.load(f)

    if os.path.exists(TFIDF_METADATA_FILE):
        with open(TFIDF_METADATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            _tfidf_vocab = data.get("vocab", {})
            _tfidf_idf = data.get("idf", {})

    print(f"[RAG] Index loaded: {_index.ntotal} vectors, {len(_chunks)} chunks")


def build_index_from_file(filepath: str):
    """
    Process a new uploaded document:
      1. Extract text
      2. Chunk it
      3. Fit local TF-IDF vocabulary across all cumulative chunks
      4. Generate local embeddings for all chunks
      5. Add to a new FAISS index matching the vocabulary size
      6. Persist index, chunks, and TF-IDF tables to disk
    """
    global _index, _chunks, _tfidf_vocab, _tfidf_idf

    # Step 1 & 2: Extract and chunk
    raw_text   = extract_text(filepath)
    new_chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)

    if not new_chunks:
        raise ValueError("Document is empty after text extraction.")

    # Combine existing chunks with new chunks
    combined_chunks = _chunks + new_chunks
    print(f"[RAG] Processing '{os.path.basename(filepath)}': {len(new_chunks)} chunks. Total chunks: {len(combined_chunks)}")

    # Fit local TF-IDF on combined chunks
    doc_freq = {}
    for chunk in combined_chunks:
        terms = set(_tfidf_tokenize(chunk))
        for term in terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    _tfidf_vocab = {}
    _tfidf_idf = {}
    for term, freq in doc_freq.items():
        if freq > 0:
            _tfidf_vocab[term] = len(_tfidf_vocab)
            _tfidf_idf[term] = math.log((1 + len(combined_chunks)) / (1 + freq)) + 1

    # Step 3 & 4: Embed all chunks using the new TF-IDF vocabulary
    dim = len(_tfidf_vocab)
    embeddings = []
    for chunk in combined_chunks:
        vector = _tfidf_transform(chunk)
        embeddings.append(vector)
    vectors = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)

    # Recreate the FAISS index with the new dimension
    _index = faiss.IndexFlatIP(dim)

    # Add vectors and update in-memory chunk list
    _index.add(vectors)
    _chunks = combined_chunks

    # Step 5: Persist to disk
    _save_index()
    print(f"[RAG] Index now has {_index.ntotal} total vectors")


def retrieve(query: str) -> Tuple[List[str], float]:
    """
    Find the top-k most relevant chunks for a query.

    Returns:
        (chunks: List[str], top_score: float)
        top_score is the cosine similarity of the best match (0.0–1.0).
        Returns ([], 0.0) if no index exists yet.
    """
    if _index is None or len(_chunks) == 0 or len(_tfidf_vocab) == 0:
        return [], 0.0

    try:
        # Embed and normalize the query using local TF-IDF
        q_vec = np.array([_tfidf_transform(query)], dtype="float32")
        faiss.normalize_L2(q_vec)

        # Search (k = min to avoid exceeding available vectors)
        k = min(TOP_K_CHUNKS, len(_chunks))
        scores, ix = _index.search(q_vec, k)

        # Map indices back to text chunks (guard against -1 which FAISS uses for padding)
        top_chunks = [_chunks[i] for i in ix[0] if 0 <= i < len(_chunks)]
        top_score  = float(scores[0][0]) if scores.size > 0 else 0.0

        # Scale sparse TF-IDF score to match dense confidence threshold (0.35)
        scaled_score = min(1.0, top_score * 2.5)

        return top_chunks, scaled_score
    except Exception as e:
        print(f"[RAG] Vector search failed: {e}")
        return _fallback_keyword_search(query)


def is_confident(score: float) -> bool:
    """
    Returns True if the retrieval score is above the confidence threshold.
    Score of 0.35 means the retrieved chunk has 35% semantic similarity to the query.
    Below this, the bot checks if the message is casual chat (respond conversationally)
    or a genuine support question (escalate as a knowledge gap).
    """
    return score >= CONFIDENCE_THRESHOLD


def generate_answer(
    query: str,
    context_chunks: List[str],
    history: List[dict],
) -> str:
    """
    Generate a customer-facing answer using retrieved context chunks.

    The system prompt instructs the model to answer ONLY from the provided
    context — not from its general training knowledge. This is the RAG 'grounding'.

    history : recent messages from the session (for multi-turn context)
    """
    # Combine retrieved chunks into a single context block
    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = f"""You are a friendly and professional customer support assistant for {BUSINESS_NAME}, a home internet and router setup service operating in Mumbai, Kalyan, and Thane.

STRICT RULE: Answer the customer's question ONLY using the information in the KNOWLEDGE BASE CONTEXT below.
- If the answer is clearly in the context, provide it concisely and helpfully (2–4 sentences).
- If the answer is not in the context, say: \"I don't have that specific information right now. A member of our team will follow up with you shortly.\"
- Do NOT invent prices, services, or policies.
- Do NOT say anything that contradicts the knowledge base.
- Be warm, clear, and avoid technical jargon unless the customer uses it first.

Business hours: {BUSINESS_HOURS}

KNOWLEDGE BASE CONTEXT:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    # Include recent history (last 6 messages)
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    # Current user query
    messages.append({"role": "user", "content": query})

    try:
        response = _client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[RAG] LLM generation failed: {e}")
        return _fallback_generate_answer(query, context_chunks)


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _tfidf_tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _tfidf_transform(text: str) -> np.ndarray:
    vector = np.zeros(len(_tfidf_vocab), dtype="float32")
    terms = _tfidf_tokenize(text)
    term_counts = {}
    for term in terms:
        if term in _tfidf_vocab:
            term_counts[term] = term_counts.get(term, 0) + 1

    for term, count in term_counts.items():
        idx = _tfidf_vocab[term]
        tf = count / len(terms) if terms else 0
        vector[idx] = tf * _tfidf_idf[term]

    return vector


def _save_index():
    """Persist FAISS index, chunk metadata, and TF-IDF mappings to disk."""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    faiss.write_index(_index, os.path.join(FAISS_INDEX_DIR, "index.faiss"))
    with open(CHUNKS_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(_chunks, f, ensure_ascii=False)
    with open(TFIDF_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"vocab": _tfidf_vocab, "idf": _tfidf_idf}, f, ensure_ascii=False)


def _fallback_keyword_search(query: str) -> Tuple[List[str], float]:
    """
    Search chunks using simple keyword matching if vector search is offline.
    """
    if not _chunks:
        return [], 0.0
        
    query_words = [w.lower() for w in query.split() if len(w) > 3]
    if not query_words:
        return [], 0.0
        
    matches = []
    for chunk in _chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for word in query_words if word in chunk_lower)
        if score > 0:
            # Calculate a pseudo similarity score (clamped between 0.0 and 1.0)
            normalized_score = min(0.9, score / len(query_words))
            matches.append((chunk, normalized_score))
            
    if not matches:
        return [], 0.0
        
    # Sort matches by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    top_matches = matches[:TOP_K_CHUNKS]
    
    top_chunks = [m[0] for m in top_matches]
    top_score = top_matches[0][1]
    
    return top_chunks, top_score


def _fallback_generate_answer(query: str, context_chunks: List[str]) -> str:
    # Try to find a line in the chunks that answers the question
    for chunk in context_chunks:
        lines = chunk.split("\n")
        for i, line in enumerate(lines):
            if "a:" in line.lower() or "answer:" in line.lower():
                return line.strip()
            if i > 0 and ("q:" in lines[i-1].lower() or "question:" in lines[i-1].lower()):
                return line.strip()
                
    # Default message if we can't extract a clean line
    return (
        "I don't have that specific information right now. "
        "A member of our team will follow up with you shortly."
    )

