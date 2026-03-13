# Semantic Search System — 20 Newsgroups

A lightweight semantic search system built on the 20 Newsgroups 
dataset (~20,000 news posts). Features fuzzy clustering, a 
cluster-aware semantic cache, and a FastAPI service.

---

## What This System Does

- Embeds 19,997 news documents into vectors using AI
- Groups them into 11 semantic clusters using fuzzy clustering
- Answers natural language queries by finding similar documents
- Caches similar queries so repeated searches are instant
- A question like "NASA missions" matches "space exploration" 
  automatically — no exact words needed

---

## Project Structure
```
semantic_search/
├── scripts/
│   ├── 01_embed_corpus.py   # Clean, embed, store in ChromaDB
│   └── 02_cluster.py        # GMM fuzzy clustering + BIC selection
├── cache/
│   └── semantic_cache.py    # Semantic cache built from scratch
├── api/
│   └── main.py              # FastAPI service
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## How to Run

### Step 1 — Setup environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### Step 2 — Build corpus embeddings (~15 min)
```bash
python scripts/01_embed_corpus.py
```

### Step 3 — Run fuzzy clustering (~5 min)
```bash
python scripts/02_cluster.py
```

### Step 4 — Start API
```bash
uvicorn api.main:app --port 8000
```

### Step 5 — Open interactive docs
```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### POST /query
```json
{
  "query": "Tell me about space exploration",
  "threshold": 0.85
}
```
Response:
```json
{
  "query": "Tell me about space exploration",
  "cache_hit": true,
  "matched_query": null,
  "similarity_score": null,
  "result": [...],
  "dominant_cluster": 3
}
```

### GET /cache/stats
```json
{
  "total_entries": 2,
  "hit_count": 1,
  "miss_count": 1,
  "hit_rate": 0.5,
  "threshold": 0.85
}
```

### DELETE /cache
Flushes entire cache and resets all stats.

---

## Design Decisions

### Embedding Model: all-MiniLM-L6-v2
- 384-dimensional vectors, fast, no API key needed
- Trained on 1B+ sentence pairs including news text
- 5x faster than larger models with only 3% quality loss

### Vector Database: ChromaDB
- Local persistent storage, no server needed
- Native cosine similarity and metadata filtering
- Stores cluster_id per document for fast lookup

### Clustering: Gaussian Mixture Models (GMM)
- NOT K-Means — GMM gives probability distributions
- A post about gun legislation scores 40% politics, 
  40% firearms — not forced into one category
- K=11 selected by BIC evidence, not guesswork
- BIC proved K=11 fits better than K=20 (the label count)

### Semantic Cache (built from scratch)
- No Redis, no Memcached — pure Python
- Cluster-partitioned: O(N/K) lookup, not O(N)
- Similarity threshold 0.85 catches real paraphrases
- LRU eviction keeps memory bounded

### Similarity Threshold Exploration
| Threshold | Behaviour |
|-----------|-----------|
| 0.70 | Aggressive — wide matches, risk of false hits |
| 0.85 | Balanced — catches true paraphrases (default) |
| 0.95 | Conservative — near-identical queries only |

---

## Cluster Results (K=11)

| Cluster | Top Categories | Semantic Theme |
|---------|---------------|----------------|
| 0 | misc.forsale | Buying and selling |
| 1 | politics.guns, politics.misc | Politics & firearms |
| 2 | sport.hockey, sport.baseball | Sports |
| 3 | sci.space, sci.crypt | Science |
| 4 | politics.mideast | Middle East politics |
| 5 | religion.christian | Christianity |
| 6 | alt.atheism | Atheism & religion debate |
| 7 | comp.windows, comp.graphics | Computers & software |
| 8 | comp.hardware (IBM + Mac) | Computer hardware |
| 9 | sci.med | Medicine |
| 10 | rec.motorcycles, rec.autos | Vehicles |

---

## Docker
```bash
docker build -t semantic-search .
docker run -p 8000:8000 semantic-search
```
```

---
