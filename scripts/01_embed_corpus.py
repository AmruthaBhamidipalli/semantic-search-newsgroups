"""
Part 1 — Embedding & Vector Database Setup
==========================================
Reads the 20 Newsgroups dataset from LOCAL files (data/20_newsgroups/)

Design Decisions:
-----------------
EMBEDDING MODEL: all-MiniLM-L6-v2
  - 384-dim, fast, trained on 1B+ sentence pairs, good for news text
  - Runs fully locally, no API key needed

VECTOR STORE: ChromaDB
  - Local persistent storage, no server needed
  - cluster_id stored per doc enables fast filtered lookup in Part 3

CLEANING:
  - Remove email headers (From/Subject/Organization) — author noise
  - Remove quoted reply lines — off-topic content bleed
  - Drop posts under 50 chars — no semantic signal
  - Remove non-ASCII — encoding noise from old Usenet posts
"""

import re
import os
import sys
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
import joblib

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "20_newsgroups")
EMB_DIR     = os.path.join(BASE_DIR, "embeddings")
CHROMA_DIR  = os.path.join(BASE_DIR, "embeddings", "chroma_db")
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

EMBED_MODEL  = "all-MiniLM-L6-v2"
COLLECTION   = "newsgroups_corpus"
BATCH_SIZE   = 64
MIN_DOC_LEN  = 50


def load_dataset():
    print("📥  Loading dataset from local files ...")
    print(f"    Looking in: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print(f"\n❌  ERROR: Could not find {DATA_DIR}")
        print("    Please make sure you:")
        print("    1. Put 20_newsgroups.tar.gz inside the data/ folder")
        print("    2. Extracted it with: tar -xzf 20_newsgroups.tar.gz")
        sys.exit(1)

    texts, labels, label_names, label_map = [], [], [], {}

    categories = sorted([c for c in os.listdir(DATA_DIR)
                         if os.path.isdir(os.path.join(DATA_DIR, c))])
    print(f"    Found {len(categories)} categories: {categories[:5]} ...")

    for category in categories:
        if category not in label_map:
            label_map[category] = len(label_names)
            label_names.append(category)
        cat_id   = label_map[category]
        cat_path = os.path.join(DATA_DIR, category)
        for filename in os.listdir(cat_path):
            filepath = os.path.join(cat_path, filename)
            if not os.path.isfile(filepath):
                continue
            try:
                with open(filepath, "r", encoding="latin-1") as f:
                    content = f.read()
                texts.append(content)
                labels.append(cat_id)
            except Exception:
                continue

    print(f"    Loaded {len(texts):,} raw documents across {len(label_names)} categories")
    return texts, labels, label_names


_PATTERNS = [
    re.compile(r"^From:.*$",         re.M),
    re.compile(r"^Subject:.*$",      re.M),
    re.compile(r"^Organization:.*$", re.M),
    re.compile(r"^Lines:.*$",        re.M),
    re.compile(r"^Message-ID:.*$",   re.M),
    re.compile(r"^NNTP-Posting.*$",  re.M),
    re.compile(r"^Path:.*$",         re.M),
    re.compile(r"^Xref:.*$",         re.M),
    re.compile(r"^>.*$",             re.M),
    re.compile(r"[\w\.-]+@[\w\.-]+", re.M),
    re.compile(r"^http\S+",          re.M),
    re.compile(r"[\-=_*#]{3,}",      re.M),
    re.compile(r"[^\x00-\x7F]+"),
    re.compile(r"\s{2,}",            re.S),
]

def clean_text(raw):
    text = raw
    for pat in _PATTERNS:
        text = pat.sub(" ", text)
    return text.strip()

def clean_corpus(raw_texts, labels):
    print("🧹  Cleaning corpus ...")
    cleaned, kept_labels, kept_indices, dropped = [], [], [], 0
    for idx, (raw, label) in enumerate(tqdm(zip(raw_texts, labels),
                                             total=len(raw_texts), desc="  cleaning")):
        text = clean_text(raw)
        if len(text) < MIN_DOC_LEN:
            dropped += 1
            continue
        cleaned.append(text)
        kept_labels.append(label)
        kept_indices.append(idx)
    print(f"    Kept: {len(cleaned):,}  |  Dropped: {dropped:,}")
    return cleaned, kept_labels, kept_indices


def embed_corpus(texts):
    print(f"\n🔢  Embedding {len(texts):,} documents ...")
    print("    This will take 10-20 minutes on CPU. Please wait ...")
    model      = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, batch_size=BATCH_SIZE,
                               normalize_embeddings=True, show_progress_bar=True)
    print(f"    Done! Shape: {embeddings.shape}")
    return embeddings


def store_in_chroma(texts, embeddings, labels, label_names, kept_indices):
    print(f"\n💾  Saving to ChromaDB ...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION,
                                           metadata={"hnsw:space": "cosine"})
    total   = len(texts)
    n_batch = (total + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(n_batch), desc="  storing"):
        start = i * BATCH_SIZE
        end   = min(start + BATCH_SIZE, total)
        collection.add(
            ids        = [str(kept_indices[j]) for j in range(start, end)],
            embeddings = embeddings[start:end].tolist(),
            documents  = texts[start:end],
            metadatas  = [{"category_id": int(labels[j]),
                           "category_name": label_names[labels[j]],
                           "doc_index": int(kept_indices[j]),
                           "cluster_id": -1}
                          for j in range(start, end)],
        )
    print(f"    Stored {collection.count():,} documents ✅")


def save_artifacts(embeddings, cleaned_texts, labels, kept_indices, label_names):
    np.save(os.path.join(EMB_DIR, "embeddings.npy"), embeddings)
    joblib.dump({"texts": cleaned_texts, "labels": labels,
                 "kept_indices": kept_indices, "label_names": label_names},
                os.path.join(EMB_DIR, "corpus_meta.joblib"))
    print(f"✅  Saved embeddings.npy and corpus_meta.joblib")


def main():
    raw_texts, labels, label_names = load_dataset()
    cleaned_texts, kept_labels, kept_indices = clean_corpus(raw_texts, labels)
    embeddings = embed_corpus(cleaned_texts)
    store_in_chroma(cleaned_texts, embeddings, kept_labels, label_names, kept_indices)
    save_artifacts(embeddings, cleaned_texts, kept_labels, kept_indices, label_names)
    print("\n🎉  Part 1 COMPLETE!  Next: python scripts/02_cluster.py")

if __name__ == "__main__":
    main()
