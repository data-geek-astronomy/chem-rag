"""
embed.py - Embedding generation and Chroma vector store management.

Handles:
- Batch embedding via Gemini API
- Storing chunks in ChromaDB with metadata
- Hybrid search (vector similarity + keyword BM25-style filtering)
"""

import os
import time
from pathlib import Path

import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from google import genai
from google.genai import types as genai_types
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

from ingest import Chunk

console = Console()

# Chroma collection name
COLLECTION_NAME = "chem_properties"


class GeminiEmbeddingFunction(EmbeddingFunction):
    """ChromaDB-compatible embedding function using Gemini text-embedding-004."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-004"):
        self._client = genai.Client(api_key=api_key)
        self._model = model_name

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            result = self._client.models.embed_content(
                model=self._model,
                contents=text,
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings


def get_chroma_client(persist_dir: str = "./data/chroma_db") -> chromadb.PersistentClient:
    """Get or create a persistent Chroma client."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_collection(client: chromadb.PersistentClient, embedding_model: str) -> chromadb.Collection:
    """Get or create the Chroma collection with Gemini embeddings."""
    gemini_ef = GeminiEmbeddingFunction(
        api_key=os.environ["GEMINI_API_KEY"],
        model_name=embedding_model,
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=gemini_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def embed_and_store(
    chunks: list[Chunk],
    collection: chromadb.Collection,
    batch_size: int = 100,
) -> int:
    """
    Embed chunks and store in Chroma.
    Skips chunks already in the collection (idempotent).
    Returns number of new chunks added.
    """
    if not chunks:
        console.print("[yellow]No chunks to embed.[/yellow]")
        return 0

    # Check which chunk IDs already exist
    existing_ids = set()
    try:
        existing = collection.get(ids=[c.chunk_id for c in chunks])
        existing_ids = set(existing["ids"])
    except Exception:
        pass

    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

    if not new_chunks:
        console.print("[green]All chunks already indexed. Nothing to add.[/green]")
        return 0

    console.print(f"[cyan]Embedding {len(new_chunks)} new chunks (skipping {len(existing_ids)} existing)...[/cyan]")

    added = 0
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]

        ids = [c.chunk_id for c in batch]
        documents = [c.text for c in batch]
        metadatas = [
            {
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "type": c.metadata.get("type", "text"),
            }
            for c in batch
        ]

        # Chroma calls the embedding function automatically
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        added += len(batch)

        console.print(f"  Stored batch {i // batch_size + 1} ({added}/{len(new_chunks)} chunks)")

        # Rate limit buffer
        if i + batch_size < len(new_chunks):
            time.sleep(0.5)

    console.print(f"[bold green]Indexed {added} new chunks. Collection total: {collection.count()}[/bold green]")
    return added


def vector_search(
    query: str,
    collection: chromadb.Collection,
    n_results: int = 10,
    filter_source: str | None = None,
) -> list[dict]:
    """
    Pure vector similarity search.
    Returns list of {text, source_file, page_number, score}.
    """
    where = {"source_file": filter_source} if filter_source else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "source_file": meta.get("source_file", ""),
                "page_number": meta.get("page_number", 0),
                "chunk_type": meta.get("type", "text"),
                "score": round(1 - dist, 4),  # cosine distance → similarity
            }
        )
    return hits


def hybrid_search(
    query: str,
    collection: chromadb.Collection,
    n_results: int = 10,
    keyword_boost_terms: list[str] | None = None,
) -> list[dict]:
    """
    Hybrid search: vector similarity + keyword re-ranking.

    Since Chroma doesn't natively support BM25, we:
    1. Retrieve 3x candidates via vector search
    2. Re-rank by boosting chunks containing keyword terms
    3. Return top n_results

    This mirrors the RRF approach used in Azure AI Search.
    """
    candidates = vector_search(query, collection, n_results=n_results * 3)

    if not keyword_boost_terms:
        # Auto-extract keywords from query (simple but effective)
        stopwords = {"the", "a", "an", "of", "for", "and", "or", "is", "in", "at", "to", "what", "give", "me"}
        keyword_boost_terms = [
            w.lower() for w in query.split() if w.lower() not in stopwords and len(w) > 2
        ]

    # Score = vector_score + keyword_bonus
    keyword_boost = 0.15
    for hit in candidates:
        text_lower = hit["text"].lower()
        bonus = sum(keyword_boost for kw in keyword_boost_terms if kw in text_lower)
        hit["hybrid_score"] = round(hit["score"] + bonus, 4)

    # Sort by hybrid score, return top n
    candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return candidates[:n_results]


def get_collection_stats(collection: chromadb.Collection) -> dict:
    """Return basic stats about the current index."""
    count = collection.count()
    if count == 0:
        return {"total_chunks": 0, "sources": []}

    # Sample to get unique sources
    sample = collection.get(limit=min(count, 1000), include=["metadatas"])
    sources = list({m["source_file"] for m in sample["metadatas"]})

    return {"total_chunks": count, "sources": sorted(sources)}
