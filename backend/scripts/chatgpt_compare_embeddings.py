# scripts/visualize_embeddings_pca_tsne.py

import chromadb
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt


# --- Config ---
CHROMA_PATH = "data/chromadb/test1"
TEXT_COLLECTION_NAME = "artwork_text_embeddings"   # BGE-M3 text embeddings
IMAGE_COLLECTION_NAME = "artwork_image_embeddings" # SigLIP image embeddings

MAX_POINTS = 500        # limit to keep t-SNE reasonable
OUT_PATH = "embeddings_pca_tsne.png"


def load_embeddings(collection) -> tuple[list[str], np.ndarray]:
    """
    Load all ids + embeddings from a Chroma collection.
    Returns (ids, embeddings_array).
    """
    data = collection.get(include=["embeddings"])
    ids = data["ids"]
    embs = np.array(data["embeddings"], dtype=np.float32)

    return ids, embs


def maybe_subsample(ids, embeddings, max_points: int):
    """
    Optionally subsample to max_points to avoid t-SNE pain.
    """
    n = embeddings.shape[0]
    if n <= max_points:
        return ids, embeddings

    idx = np.random.choice(n, size=max_points, replace=False)
    ids_sub = [ids[i] for i in idx]
    embs_sub = embeddings[idx]
    return ids_sub, embs_sub


def compute_pca_2d(embeddings: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(embeddings)


def compute_tsne_2d(embeddings: np.ndarray) -> np.ndarray:
    # Perplexity must be < n_samples; clamp to something reasonable
    n = embeddings.shape[0]
    perplexity = min(30, max(5, n // 5))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    return tsne.fit_transform(embeddings)


def main() -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    text_coll = client.get_collection(TEXT_COLLECTION_NAME)
    image_coll = client.get_collection(IMAGE_COLLECTION_NAME)

    # --- Load embeddings ---
    text_ids, text_embs = load_embeddings(text_coll)
    image_ids, image_embs = load_embeddings(image_coll)

    if text_embs.size == 0 or image_embs.size == 0:
        print("No embeddings in one or both collections.")
        return

    # Optional subsampling
    text_ids, text_embs = maybe_subsample(text_ids, text_embs, MAX_POINTS)
    image_ids, image_embs = maybe_subsample(image_ids, image_embs, MAX_POINTS)

    print(f"Using {len(text_ids)} text embeddings, {len(image_ids)} image embeddings")

    # --- PCA + t-SNE ---
    text_pca = compute_pca_2d(text_embs)
    text_tsne = compute_tsne_2d(text_embs)

    image_pca = compute_pca_2d(image_embs)
    image_tsne = compute_tsne_2d(image_embs)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax_text_pca = axes[0, 0]
    ax_text_tsne = axes[0, 1]
    ax_img_pca = axes[1, 0]
    ax_img_tsne = axes[1, 1]

    # Text PCA
    ax_text_pca.scatter(text_pca[:, 0], text_pca[:, 1], s=10, alpha=0.7)
    ax_text_pca.set_title("Text embeddings – PCA")
    ax_text_pca.set_xticks([])
    ax_text_pca.set_yticks([])

    # Text t-SNE
    ax_text_tsne.scatter(text_tsne[:, 0], text_tsne[:, 1], s=10, alpha=0.7)
    ax_text_tsne.set_title("Text embeddings – t-SNE")
    ax_text_tsne.set_xticks([])
    ax_text_tsne.set_yticks([])

    # Image PCA
    ax_img_pca.scatter(image_pca[:, 0], image_pca[:, 1], s=10, alpha=0.7)
    ax_img_pca.set_title("Image embeddings – PCA")
    ax_img_pca.set_xticks([])
    ax_img_pca.set_yticks([])

    # Image t-SNE
    ax_img_tsne.scatter(image_tsne[:, 0], image_tsne[:, 1], s=10, alpha=0.7)
    ax_img_tsne.set_title("Image embeddings – t-SNE")
    ax_img_tsne.set_xticks([])
    ax_img_tsne.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    plt.close()

    print(f"Saved PCA/t-SNE figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
