from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"


@dataclass
class TextEmbeddingConfig:
    """
    Configuration for the text embedding model (BGE-M3).
    """
    model_name: str = DEFAULT_MODEL_NAME
    # Not used yet, but you might want batching later
    batch_size: int = 16


def load_text_embedding_model(
    config: TextEmbeddingConfig,
) -> SentenceTransformer:
    """
    Load and return the BGE-M3 text embedding model.

    Returns
    -------
    SentenceTransformer
        The model used to compute text embeddings.
    """
    # SentenceTransformer will handle device selection internally
    model = SentenceTransformer(config.model_name)
    return model


def build_embedding_text(artwork: Dict[str, Any]) -> str:
    """
    Build a caption-like text representation of an artwork to embed with BGE-M3.
    """
    title = artwork.get("title") or ""
    artist = artwork.get("artist_title") or ""
    date = artwork.get("date_display") or ""
    medium = artwork.get("medium_display") or ""
    subject_titles = artwork.get("subject_titles") or []
    classification_titles = artwork.get("classification_titles") or []
    term_titles = artwork.get("term_titles") or []
    material_titles = artwork.get("material_titles") or []

    bits: List[str] = []

    if title:
        bits.append(title)

    desc_parts: List[str] = []

    if medium:
        desc_parts.append(medium)

    if artist:
        desc_parts.append(f"by {artist}")

    if date:
        desc_parts.append(f"from {date}")

    if desc_parts:
        bits.append(" ".join(desc_parts))

    # Tags line – optional
    tags = set()
    for lst in (subject_titles, classification_titles, term_titles, material_titles):
        for t in lst:
            if t:
                tags.add(t)

    if tags:
        bits.append("Tags: " + ", ".join(sorted(tags)))

    text = "\n".join(bits).strip()
    if not text:
        text = f"Artwork {artwork.get('id', '')}"

    # Debug (optional – remove if too spammy)
    # print(text, "\n")

    return text


def embed_text(
    text: str,
    model: SentenceTransformer,
) -> List[float]:
    """
    Compute a BGE-M3 embedding vector for a single piece of text.

    NOTE: For BGE-M3, they recommend different prompts for
    queries vs documents. Here we treat artwork descriptions as
    'documents', so we prepend the document prompt.

    For queries, you should use a separate function with the query prompt.
    """
    text = text.strip()
    if not text:
        return []

    # BGE-M3 doc prompt (recommended pattern)
    doc_text = f"Represent this document for retrieval: {text}"

    emb = model.encode(
        doc_text,
        normalize_embeddings=True,
    )
    return emb.tolist()


def embed_artwork_text(
    model: SentenceTransformer,
    artwork: Dict[str, Any],
) -> Tuple[int, List[float], str]:
    """
    Build embedding text for an artwork and embed it with BGE-M3.

    Returns
    -------
    (artwork_id, embedding, embedding_text)
    """
    artwork_id = artwork.get("id")
    embedding_text = build_embedding_text(artwork)
    embedding = embed_text(embedding_text, model)
    return artwork_id, embedding, embedding_text
