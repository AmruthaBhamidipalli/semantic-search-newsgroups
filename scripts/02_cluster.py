print("Starting Part 2 - Fuzzy Clustering...")
print("Importing libraries...")

import os
import sys
import numpy as np
import joblib

print("  numpy, joblib OK")

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

print("  sklearn OK")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("  matplotlib OK")

import chromadb

print("  chromadb OK")
print("All imports successful!")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.getcwd()
EMB_DIR     = os.path.join(BASE_DIR, "embeddings")
CLUSTER_DIR = os.path.join(BASE_DIR, "clustering")
CHROMA_DIR  = os.path.join(BASE_DIR, "embeddings", "chroma_db")
os.makedirs(CLUSTER_DIR, exist_ok=True)

COLLECTION  = "newsgroups_corpus"
PCA_DIMS    = 64
K_MIN       = 8
K_MAX       = 25
BATCH_SIZE  = 512


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════
print("\n📂  Loading embeddings from Part 1 ...")
embeddings = np.load(os.path.join(EMB_DIR, "embeddings.npy"))
meta       = joblib.load(os.path.join(EMB_DIR, "corpus_meta.joblib"))
print(f"    Embeddings shape : {embeddings.shape}")
print(f"    Documents        : {len(meta['texts']):,}")


# ══════════════════════════════════════════════════════════════════════════
# 2. PCA
# ══════════════════════════════════════════════════════════════════════════
print(f"\n📉  PCA: {embeddings.shape[1]}D → {PCA_DIMS}D ...")
pca     = PCA(n_components=PCA_DIMS, random_state=42)
reduced = pca.fit_transform(embeddings)
var     = pca.explained_variance_ratio_.sum()
print(f"    Variance retained: {var:.1%}")
print(f"    Reduced shape    : {reduced.shape}")


# ══════════════════════════════════════════════════════════════════════════
# 3. SELECT K WITH BIC
# ══════════════════════════════════════════════════════════════════════════
print(f"\n🔍  Searching K from {K_MIN} to {K_MAX} using BIC ...")
print("    (This takes 3-7 minutes, please wait...)")

bics, aics, ks = [], [], []

for k in range(K_MIN, K_MAX + 1):
    print(f"    Trying K={k}...", end=" ", flush=True)
    gm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        max_iter=100,
        n_init=2,
        random_state=42,
    )
    gm.fit(reduced)
    b = gm.bic(reduced)
    a = gm.aic(reduced)
    bics.append(b)
    aics.append(a)
    ks.append(k)
    print(f"BIC={b:.0f}")

# Find best K = where BIC drops the most
bic_arr = np.array(bics)
drops   = bic_arr[:-1] - bic_arr[1:]
best_k  = ks[int(np.argmax(drops))]
print(f"\n    ✅  Selected K = {best_k}  (steepest BIC drop)")

# Save BIC plot
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ks, bics, "o-", label="BIC", color="steelblue")
ax.plot(ks, aics, "s--", label="AIC", color="coral")
ax.axvline(x=best_k, color="green", linestyle=":", label=f"Selected K={best_k}")
ax.set_xlabel("Number of clusters K")
ax.set_ylabel("Score (lower = better)")
ax.set_title("GMM Model Selection: BIC & AIC vs K")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(CLUSTER_DIR, "bic_aic_plot.png")
fig.savefig(plot_path, dpi=150)
plt.close()
print(f"    Saved BIC plot → {plot_path}")


# ══════════════════════════════════════════════════════════════════════════
# 4. FIT FINAL GMM
# ══════════════════════════════════════════════════════════════════════════
print(f"\n🎯  Fitting final GMM with K={best_k} ...")
gmm = GaussianMixture(
    n_components=best_k,
    covariance_type="diag",
    max_iter=300,
    n_init=5,
    random_state=42,
    verbose=1,
)
gmm.fit(reduced)
soft_labels      = gmm.predict_proba(reduced)   # shape: (N, K)
hard_assignments = soft_labels.argmax(axis=1)   # dominant cluster per doc

print(f"    soft_labels shape : {soft_labels.shape}")
print(f"    Log-likelihood    : {gmm.score(reduced):.4f}")


# ══════════════════════════════════════════════════════════════════════════
# 5. CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print(f"\n📊  Cluster Analysis (K={best_k})")
print("=" * 60)

texts       = meta["texts"]
labels      = meta["labels"]
label_names = meta["label_names"]

eps     = 1e-12
entropy = -(soft_labels * np.log(soft_labels + eps)).sum(axis=1)

cluster_report = {}

for c in range(best_k):
    members = np.where(hard_assignments == c)[0]
    if len(members) == 0:
        continue

    # Top categories in this cluster
    cat_counts = {}
    for idx in members:
        cname = label_names[labels[idx]]
        cat_counts[cname] = cat_counts.get(cname, 0) + 1
    top_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:3]

    # Representative doc (lowest entropy = most confident)
    member_entropy = entropy[members]
    rep_idx        = members[np.argmin(member_entropy)]
    bnd_idx        = members[np.argmax(member_entropy)]

    cluster_report[c] = {
        "size"       : len(members),
        "top_cats"   : top_cats,
        "avg_entropy": float(member_entropy.mean()),
    }

    print(f"\n── Cluster {c}  ({len(members):,} docs)")
    print(f"   Top categories : {top_cats}")
    print(f"   Avg entropy    : {member_entropy.mean():.3f}")
    print(f"   Example doc    : {texts[rep_idx][:150].strip()}")

joblib.dump(cluster_report, os.path.join(CLUSTER_DIR, "cluster_report.joblib"))
print(f"\n    Saved cluster_report.joblib")


# ══════════════════════════════════════════════════════════════════════════
# 6. UPDATE CHROMADB WITH CLUSTER IDs
# ══════════════════════════════════════════════════════════════════════════
print(f"\n🔄  Writing cluster IDs to ChromaDB ...")
client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION)

kept_indices = meta["kept_indices"]
n_docs       = len(kept_indices)
n_batches    = (n_docs + BATCH_SIZE - 1) // BATCH_SIZE

for b in range(n_batches):
    start = b * BATCH_SIZE
    end   = min(start + BATCH_SIZE, n_docs)
    if b % 10 == 0:
        print(f"    Batch {b+1}/{n_batches}...")

    ids      = [str(kept_indices[j]) for j in range(start, end)]
    new_meta = [
        {
            "cluster_id"     : int(hard_assignments[j]),
            "cluster_entropy": float(entropy[j]),
            "category_id"    : int(labels[j]),
            "category_name"  : label_names[labels[j]],
            "doc_index"      : int(kept_indices[j]),
        }
        for j in range(start, end)
    ]
    collection.update(ids=ids, metadatas=new_meta)

print(f"    Updated {n_docs:,} documents with cluster IDs ✅")


# ══════════════════════════════════════════════════════════════════════════
# 7. SAVE ALL ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════
print(f"\n💾  Saving clustering artifacts ...")
joblib.dump(gmm, os.path.join(CLUSTER_DIR, "gmm_model.joblib"))
joblib.dump(pca, os.path.join(CLUSTER_DIR, "pca_model.joblib"))
np.save(os.path.join(CLUSTER_DIR, "soft_labels.npy"), soft_labels)
np.save(os.path.join(CLUSTER_DIR, "hard_assignments.npy"), hard_assignments)
joblib.dump(
    {"k": best_k, "label_names": label_names},
    os.path.join(CLUSTER_DIR, "cluster_meta.joblib"),
)
print("    Saved: gmm_model.joblib, pca_model.joblib")
print("    Saved: soft_labels.npy, hard_assignments.npy")
print("    Saved: cluster_meta.joblib")

print("\n🎉  Part 2 COMPLETE!")
print("    Check clustering/bic_aic_plot.png to see your K selection graph")
print("    Next step: uvicorn api.main:app --reload --port 8000")
