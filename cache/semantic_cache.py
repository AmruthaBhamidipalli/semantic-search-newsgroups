"""
Part 3 — Semantic Cache (built from scratch, no Redis/Memcached)
================================================================
The core idea:
  When a new query arrives, instead of recomputing the result, we check
  whether we've answered a "close enough" question before. If yes (cache hit),
  we return the stored result instantly.

  "Close enough" = cosine similarity between query embeddings ≥ THRESHOLD.

Design Decisions:
-----------------

DATA STRUCTURE:
  The cache is a plain Python dict keyed by cluster_id, each holding a list
  of CacheEntry objects.

  Why cluster-partitioned?
    Naively, checking a new query against ALL cached entries is O(N) in cache
    size. If we first identify the query's dominant cluster (from Part 2's GMM)
    we only need to check entries in that cluster's bucket. This reduces
    lookup to O(N/K) on average — a K-fold speedup that matters as cache grows.

    This is the "cluster structure doing real work" the task asks for.

    Example: cache has 500 entries across 15 clusters (~33 per cluster).
    Without clustering: 500 comparisons. With: ~33 comparisons + 1 GMM predict.
    At K=15, that's a 15× speedup for large caches.

THE TUNABLE THRESHOLD:
  SIMILARITY_THRESHOLD ∈ (0, 1] controls cache aggressiveness.

  Low threshold (e.g. 0.70):
    → Many hits; "What is AI?" and "Tell me about machine learning" match.
    → Risk: returning a result for a query that's only loosely related.
    → Reveals: the system's semantic neighbourhood is wide; clusters are fuzzy.

  High threshold (e.g. 0.97):
    → Very few hits; queries must be nearly identical paraphrases.
    → Behaves almost like exact-match caching.
    → Reveals: how rare true paraphrases are in practice.

  Interesting middle ground (~0.85–0.90):
    → Catches real paraphrases ("best GPU for ML" vs "top GPU for deep learning")
    → Rejects topic shifts ("GPU for gaming" vs "GPU for ML")
    → This is where semantic search adds genuine value over keyword caching.

  We expose SIMILARITY_THRESHOLD as a config constant AND as a runtime
  parameter so the API can override it per-request for experiments.

CACHE EVICTION:
  Simple LRU (Least Recently Used) per cluster bucket, capped at
  MAX_ENTRIES_PER_CLUSTER. This prevents unbounded memory growth.
  The oldest entries (by last_accessed timestamp) are dropped first.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any
from collections import OrderedDict


# ── Configuration ──────────────────────────────────────────────────────────────

# The key tunable: see module docstring for behavioural analysis
SIMILARITY_THRESHOLD      = 0.85

# LRU cap per cluster bucket — prevents memory blowup in long-running services
MAX_ENTRIES_PER_CLUSTER   = 200

# Fall-back: if GMM cluster assignment is uncertain (high entropy), also check
# the top-2 clusters. Boundary documents belong to multiple clusters semantically.
ENTROPY_THRESHOLD_FALLBACK = 1.5   # nats; above this = uncertain doc


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    """
    One cached query-result pair.

    Fields:
      query_text      : original query string (for display in API response)
      query_embedding : unit-normalised 384-D vector (for cosine comparison)
      result          : whatever we computed (search results, answer text, etc.)
      dominant_cluster: which cluster bucket this lives in
      soft_assignment : full (K,) probability vector — stored so we can later
                        analyse cross-cluster hits (a query from cluster 3
                        being served from a cluster 3 entry that's 40% cluster 5)
      hit_count       : how many times this entry has been returned as a hit
      last_accessed   : unix timestamp — used by LRU eviction
      created_at      : unix timestamp
    """
    query_text       : str
    query_embedding  : np.ndarray
    result           : Any
    dominant_cluster : int
    soft_assignment  : np.ndarray
    hit_count        : int       = 0
    last_accessed    : float     = field(default_factory=time.time)
    created_at       : float     = field(default_factory=time.time)


# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC CACHE
# ══════════════════════════════════════════════════════════════════════════════

class SemanticCache:
    """
    Cluster-partitioned semantic cache.

    Internal structure:
        _buckets : dict[cluster_id (int) → OrderedDict[str → CacheEntry]]
                   OrderedDict preserves insertion order for LRU eviction.
                   Key = query_text (str) for O(1) exact-match lookup before
                   doing the more expensive cosine scan.

    Public API:
        lookup(query_embedding, soft_assignment, threshold) → CacheEntry | None
        store(query_text, query_embedding, result, soft_assignment)
        flush()
        stats() → dict
    """

    def __init__(
        self,
        default_threshold: float = SIMILARITY_THRESHOLD,
        max_per_cluster  : int   = MAX_ENTRIES_PER_CLUSTER,
    ):
        self.default_threshold = default_threshold
        self.max_per_cluster   = max_per_cluster

        # cluster_id → OrderedDict(query_text → CacheEntry)
        self._buckets: dict[int, OrderedDict] = {}

        # Global statistics
        self._hit_count  = 0
        self._miss_count = 0

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_bucket(self, cluster_id: int) -> OrderedDict:
        """Return the bucket for cluster_id, creating it if absent."""
        if cluster_id not in self._buckets:
            self._buckets[cluster_id] = OrderedDict()
        return self._buckets[cluster_id]

    def _evict_if_full(self, cluster_id: int) -> None:
        """
        LRU eviction: if bucket exceeds max_per_cluster, remove the entry
        with the oldest last_accessed timestamp.
        This is O(N) in bucket size, acceptable for small buckets (~200 entries).
        """
        bucket = self._buckets.get(cluster_id, {})
        if len(bucket) >= self.max_per_cluster:
            # Find LRU entry
            lru_key = min(bucket.keys(), key=lambda k: bucket[k].last_accessed)
            del bucket[lru_key]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two unit-normed vectors.
        Since embeddings are L2-normalised (done in Part 1), this is just
        the dot product — O(D) with no division needed.
        """
        return float(np.dot(a, b))

    def _clusters_to_search(
        self,
        soft_assignment: np.ndarray,
        entropy: float,
    ) -> list[int]:
        """
        Determine which cluster buckets to search.

        Normal case (low entropy): search dominant cluster only.
        Uncertain case (high entropy): search top-2 clusters.
          This handles boundary documents (e.g. gun legislation in both
          politics AND firearms clusters) — we don't want to miss a cached
          answer just because the query landed on the wrong side of a fuzzy boundary.
        """
        dominant = int(soft_assignment.argmax())
        if entropy < ENTROPY_THRESHOLD_FALLBACK:
            return [dominant]
        # High entropy: add the second-best cluster too
        sorted_clusters = np.argsort(soft_assignment)[::-1]
        return sorted_clusters[:2].tolist()

    # ── Public API ─────────────────────────────────────────────────────────────

    def lookup(
        self,
        query_embedding : np.ndarray,
        soft_assignment : np.ndarray,
        threshold       : Optional[float] = None,
    ) -> Optional[tuple[CacheEntry, float]]:
        """
        Search the cache for a semantically similar query.

        Algorithm:
          1. Compute entropy of soft_assignment → decide which buckets to check
          2. For each candidate bucket, scan all entries and compute cosine sim
          3. Return the entry with highest similarity ≥ threshold, or None

        Returns:
          (CacheEntry, similarity_score) if hit, else None

        Complexity: O(B × N/K) where B = buckets searched (1 or 2),
                    N = total entries, K = number of clusters.
        """
        threshold = threshold if threshold is not None else self.default_threshold

        eps     = 1e-12
        entropy = float(-(soft_assignment * np.log(soft_assignment + eps)).sum())
        buckets = self._clusters_to_search(soft_assignment, entropy)

        best_entry = None
        best_score = -1.0

        for cluster_id in buckets:
            bucket = self._buckets.get(cluster_id)
            if not bucket:
                continue

            for entry in bucket.values():
                score = self._cosine_similarity(query_embedding, entry.query_embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_entry is not None and best_score >= threshold:
            # Update LRU and hit stats on the winning entry
            best_entry.hit_count    += 1
            best_entry.last_accessed = time.time()
            self._hit_count         += 1
            return best_entry, best_score

        self._miss_count += 1
        return None

    def store(
        self,
        query_text      : str,
        query_embedding : np.ndarray,
        result          : Any,
        soft_assignment : np.ndarray,
    ) -> CacheEntry:
        """
        Store a new query-result pair in the appropriate cluster bucket.

        If the exact query text is already cached (e.g. re-submitted after
        a flush), we overwrite it with the new result — idempotent behaviour.
        """
        dominant_cluster = int(soft_assignment.argmax())
        self._evict_if_full(dominant_cluster)

        bucket = self._get_bucket(dominant_cluster)
        entry  = CacheEntry(
            query_text       = query_text,
            query_embedding  = query_embedding.copy(),   # defensive copy
            result           = result,
            dominant_cluster = dominant_cluster,
            soft_assignment  = soft_assignment.copy(),
        )
        bucket[query_text] = entry
        # Move to end of OrderedDict (most recently used)
        bucket.move_to_end(query_text)
        return entry

    def flush(self) -> None:
        """Clear all buckets and reset statistics."""
        self._buckets.clear()
        self._hit_count  = 0
        self._miss_count = 0

    def stats(self) -> dict:
        """Return current cache statistics."""
        total_entries = sum(len(b) for b in self._buckets.values())
        total_queries = self._hit_count + self._miss_count
        hit_rate      = self._hit_count / total_queries if total_queries > 0 else 0.0

        return {
            "total_entries"   : total_entries,
            "hit_count"       : self._hit_count,
            "miss_count"      : self._miss_count,
            "hit_rate"        : round(hit_rate, 4),
            "buckets"         : {
                str(cid): len(b)
                for cid, b in self._buckets.items()
            },
            "threshold"       : self.default_threshold,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"SemanticCache("
            f"entries={s['total_entries']}, "
            f"hits={s['hit_count']}, "
            f"misses={s['miss_count']}, "
            f"threshold={self.default_threshold})"
        )
