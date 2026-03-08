# Semantic Search System — 20 Newsgroups

A lightweight semantic search system with fuzzy clustering, 
semantic cache, and FastAPI service.

## How to Run

1. Create venv:
   python -m venv venv
   venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Build corpus embeddings (~15 min):
   python scripts/01_embed_corpus.py

4. Run fuzzy clustering (~5 min):
   python scripts/02_cluster.py

5. Start API:
   uvicorn api.main:app --port 8000

6. Open browser:
   http://127.0.0.1:8000/docs

## API Endpoints

- POST /query — semantic search with cache
- GET /cache/stats — cache statistics  
- DELETE /cache — flush cache

## Design Decisions

- Embedding: all-MiniLM-L6-v2 (fast, 384-dim, no API key)
- Vector DB: ChromaDB (local, no server needed)
- Clustering: GMM with BIC evidence (K=11, not K=20)
- Cache: cluster-partitioned dict, threshold=0.85
- No Redis — built from scratch

## Docker

docker build -t semantic-search .
docker run -p 8000:8000 semantic-search
