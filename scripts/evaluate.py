import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import chromadb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backend.query import query_via_text, query_via_images


def load_artworks_sample(db_path: str, limit: int = 100) -> List[dict]:
    """Load a sample of artworks (ids + metadata) from the text collection."""
    client = chromadb.PersistentClient(path=db_path)
    text_collection = client.get_collection("artwork_text_embeddings")

    results = text_collection.get(limit=limit, include=["metadatas"])
    artworks = []
    for art_id, metadata in zip(results["ids"], results["metadatas"]):
        m = dict(metadata) if metadata else {}
        m["id"] = art_id
        artworks.append(m)
    return artworks


def print_artist_counts(artworks: List[dict]) -> None:
    artists = []
    for a in artworks:
        artist = a.get("artist_title") or "Unknown"
        artists.append(artist)

    counts = Counter(artists)
    print("\nWorks per artist (in this sample):")
    for artist, c in counts.most_common():
        print(f"  {artist}: {c}")
    print()


def _extract_top_ids_and_metas(results: dict) -> Tuple[List[str], List[dict]]:
    # Chroma returns list-of-lists for ids/metadatas when using query
    ids = results.get("ids", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []
    return ids, metas


def positions_to_recall_curve(positions: List[Optional[int]], max_k: int) -> Dict[int, float]:
    """positions: list of 1..max_k if found, else None."""
    valid = [p for p in positions if p is not None]
    total = len(positions)
    recalls = {}
    for k in range(1, max_k + 1):
        hits = sum(1 for p in positions if p is not None and p <= k)
        recalls[k] = hits / total if total > 0 else 0.0
    return recalls


def evaluate_title_retrieval(
    artworks: List[dict],
    query_fn,
    db_path: str,
    max_k: int = 10,
) -> Tuple[Dict[int, float], List[Optional[int]], int]:
    """
    Query by title. Target is the *same artwork id*.
    Returns:
      recall@k curve, list of positions (1..max_k or None), error_count
    """
    positions: List[Optional[int]] = []
    error_count = 0

    # only evaluate artworks that have a non-empty title
    eval_items = [a for a in artworks if (a.get("title") or "").strip()]
    for a in tqdm(eval_items, desc="Title queries"):
        art_id = a["id"]
        title = a["title"].strip()

        try:
            results = query_fn(title, n_results=max_k, db_path=db_path)
            ids, _ = _extract_top_ids_and_metas(results)

            if art_id in ids:
                positions.append(ids.index(art_id) + 1)  # 1-indexed rank
            else:
                positions.append(None)
        except Exception:
            error_count += 1
            positions.append(None)

    recall_curve = positions_to_recall_curve(positions, max_k=max_k)
    return recall_curve, positions, error_count


def evaluate_artist_retrieval(
    artworks: List[dict],
    query_fn,
    db_path: str,
    max_k: int = 10,
) -> Tuple[Dict[int, float], float, int]:
    """
    Query by artist name.
    Artist-Recall@k: whether at least one of the top-k results has the same artist.
    Artist-Purity@10: out of the top-10, fraction matching the artist (averaged).
    """
    error_count = 0
    purity_sum = 0.0
    artist_hits_at_k = defaultdict(int)

    eval_items = [
        a for a in artworks
        if (a.get("artist_title") or "").strip() and a.get("artist_title") != "Unknown"
    ]
    total = len(eval_items)

    for a in tqdm(eval_items, desc="Artist queries"):
        artist = a["artist_title"].strip()

        try:
            results = query_fn(artist, n_results=max_k, db_path=db_path)
            ids, metas = _extract_top_ids_and_metas(results)

            # Purity@max_k (top max_k only)
            same_artist = 0
            for m in metas[:max_k]:
                if (m or {}).get("artist_title") == artist:
                    same_artist += 1
            purity_sum += (same_artist / max_k) if max_k > 0 else 0.0

            # Artist-Recall@k: is there at least one same-artist item in top-k?
            # Compute prefix hits efficiently
            prefix_hit = [False] * (max_k + 1)
            found = False
            for i in range(1, max_k + 1):
                if i - 1 < len(metas) and (metas[i - 1] or {}).get("artist_title") == artist:
                    found = True
                prefix_hit[i] = found

            for k in range(1, max_k + 1):
                if prefix_hit[k]:
                    artist_hits_at_k[k] += 1

        except Exception:
            error_count += 1
            # Treat as miss for recall; purity adds 0
            continue

    artist_recall = {k: (artist_hits_at_k[k] / total if total > 0 else 0.0) for k in range(1, max_k + 1)}
    purity_at_10 = purity_sum / total if total > 0 else 0.0
    return artist_recall, purity_at_10, error_count


def print_recall_curve(name: str, curve: Dict[int, float]) -> None:
    print(name)
    for k in range(1, max(curve.keys()) + 1):
        print(f"  Recall@{k}: {curve[k]:.2%}")
    print()


def main():
    db_path = "data/chromadb/test_full1"
    sample_limit = 50
    max_k = 10

    print("Loading artworks sample...")
    artworks = load_artworks_sample(db_path=db_path, limit=sample_limit)
    print(f"Loaded {len(artworks)} artworks from {db_path}")

    print_artist_counts(artworks)

    modes = [
        ("text", query_via_text),
        ("images", query_via_images),
    ]

    for mode_name, qfn in modes:
        print("=" * 70)
        print(f"MODE: {mode_name} (top-{max_k})")
        print("=" * 70)

        title_curve, title_positions, title_errors = evaluate_title_retrieval(
            artworks, qfn, db_path=db_path, max_k=max_k
        )
        print_recall_curve("Title self-retrieval (query = title, target = same artwork id)", title_curve)
        if title_errors:
            print(f"  (title query errors: {title_errors})\n")

        artist_curve, artist_purity10, artist_errors = evaluate_artist_retrieval(
            artworks, qfn, db_path=db_path, max_k=max_k
        )
        print_recall_curve("Artist-Recall@K (query = artist, hit if any top-k has same artist)", artist_curve)
        print(f"Artist-Purity@{max_k} (avg fraction of top-{max_k} with same artist): {artist_purity10:.2%}\n")
        if artist_errors:
            print(f"  (artist query errors: {artist_errors})\n")


if __name__ == "__main__":
    main()
