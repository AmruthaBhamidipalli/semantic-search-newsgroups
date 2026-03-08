"""
Part 4 - FastAPI Service
========================
Endpoints:
  POST   /query        - semantic search with cache check
  GET    /cache/stats  - cache statistics  
  DELETE /cache        - flush cache and reset stats
  GET    /health       - health check
"""

import os
import sys
import numpy as np
import joblib
import chromadb
from typing import Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Fix import path so cache module is found ──────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cache.semantic_cache import SemanticCache

# ── Paths ─────────────────────────────────────────────────────────────────────
CHROMA_DIR  = os.path.join(ROOT, "embeddings", "chroma_db")
CLUSTER_DIR = os.path.join(ROOT, "clustering")
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION  = "newsgroups_corpus"
N_RESULTS   = 5

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Semantic Search API",
    description="20 Newsgroups semantic search with fuzzy clustering and semantic cache",
    version="1.0.0",
)

# ── Global state (loaded once at startup) ─────────────────────────────────────
_embed_model = None
_gmm         = None
_pca         = None
_collection  = None
_cache       = None
_k           = None


@app.on_event("startup")
async def startup():
    """
    Load all heavy objects once when server starts.
    Stored as module-level globals so every request reuses them.
    """
    global _embed_model, _gmm, _pca, _collection, _cache, _k

    print("🚀  Loading models and artifacts ...")

    # Embedding model
    _embed_model = SentenceTransformer(EMBED_MODEL)
    print("    Embedding model loaded ✅")

    # GMM + PCA from Part 2
    _gmm  = joblib.load(os.path.join(CLUSTER_DIR, "gmm_model.joblib"))
    _pca  = joblib.load(os.path.join(CLUSTER_DIR, "pca_model.joblib"))
    meta  = joblib.load(os.path.join(CLUSTER_DIR, "cluster_meta.joblib"))
    _k    = meta["k"]
    print(f"    GMM loaded: K={_k} clusters ✅")

    # ChromaDB
    client      = chromadb.PersistentClient(path=CHROMA_DIR)
    _collection = client.get_collection(COLLECTION)
    print(f"    ChromaDB: {_collection.count():,} documents ✅")

    # Semantic cache
    _cache = SemanticCache(default_threshold=0.85)
    print("    Semantic cache ready ✅")
    print(f"\n    🟢  Service ready on http://127.0.0.1:8000")
    print(f"    📖  Interactive docs at http://127.0.0.1:8000/docs\n")


# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query    : str
    threshold: Optional[float] = None


class QueryResponse(BaseModel):
    query           : str
    cache_hit       : bool
    matched_query   : Optional[str]
    similarity_score: Optional[float]
    result          : Any
    dominant_cluster: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def embed_query(text: str) -> np.ndarray:
    vec = _embed_model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec[0]


def assign_cluster(query_embedding: np.ndarray):
    reduced         = _pca.transform(query_embedding.reshape(1, -1))
    soft_assignment = _gmm.predict_proba(reduced)[0]
    dominant        = int(soft_assignment.argmax())
    return dominant, soft_assignment


def search_corpus(query_embedding: np.ndarray, n_results: int = N_RESULTS):
    results = _collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "document"  : doc[:400],
            "category"  : meta.get("category_name", "unknown"),
            "cluster_id": meta.get("cluster_id", -1),
            "similarity": round(1 - dist, 4),
        })
    return hits


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest):
    """
    1. Embed the query
    2. Assign to cluster via GMM
    3. Check semantic cache
    4. Hit  → return cached result
    5. Miss → search ChromaDB, store in cache, return result
    """
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="Query must be non-empty")

    query_text = body.query.strip()

    # Embed + cluster
    query_embedding          = embed_query(query_text)
    dominant_cluster, soft   = assign_cluster(query_embedding)

    # Cache lookup
    hit = _cache.lookup(query_embedding, soft, threshold=body.threshold)

    if hit is not None:
        cached_entry, score = hit
        return QueryResponse(
            query            = query_text,
            cache_hit        = True,
            matched_query    = cached_entry.query_text,
            similarity_score = round(score, 4),
            result           = cached_entry.result,
            dominant_cluster = dominant_cluster,
        )

    # Cache miss - search corpus
    result = search_corpus(query_embedding)

    # Store in cache
    _cache.store(
        query_text      = query_text,
        query_embedding = query_embedding,
        result          = result,
        soft_assignment = soft,
    )

    return QueryResponse(
        query            = query_text,
        cache_hit        = False,
        matched_query    = None,
        similarity_score = None,
        result           = result,
        dominant_cluster = dominant_cluster,
    )


@app.get("/cache/stats")
async def cache_stats():
    """Returns current cache statistics."""
    return JSONResponse(content=_cache.stats())


@app.delete("/cache")
async def flush_cache():
    """Flushes entire cache and resets all counters."""
    _cache.flush()
    return JSONResponse(content={"message": "Cache flushed. All stats reset to zero."})


@app.get("/health")
async def health():
    return {
        "status"       : "ok",
        "corpus_size"  : _collection.count(),
        "clusters"     : _k,
        "cache_entries": _cache.stats()["total_entries"],
    }
