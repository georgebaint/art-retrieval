from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from transformers import AutoModel, AutoProcessor
import torch
import torch.nn.functional as F

# Default checkpoint – adjust if needed
DEFAULT_CKPT = "google/siglip2-base-patch16-naflex"


@dataclass
class TextEmbeddingConfig:
    """
    Configuration for the text embedding model.
    """
    model_ckpt: str = DEFAULT_CKPT
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    iiif_base_url: str = "https://www.artic.edu/iiif/2"
    # Not used yet, but you might want batching later
    batch_size: int = 16


def load_text_embedding_model(
    config: TextEmbeddingConfig,
) -> tuple[AutoModel, AutoProcessor, torch.device]:
    """
    Load and return the text embedding model, processor and device.

    Returns
    -------
    (model, processor, device)
        model : AutoModel
            The vision (SigLIP) model.
        processor : AutoProcessor
            Preprocessor for images.
        device : torch.device
            Device the model is on.
    """
    device = torch.device(config.device)

    model = AutoModel.from_pretrained(config.model_ckpt).to(device).eval()
    processor = AutoProcessor.from_pretrained(config.model_ckpt)

    return model, processor, device


def build_embedding_text(artwork: Dict[str, Any]) -> str:
    title = artwork.get("title") or ""
    artist = artwork.get("artist_title") or ""
    date = artwork.get("date_display") or ""
    place = artwork.get("place_of_origin") or ""
    medium = artwork.get("medium_display") or ""
    subject_titles = artwork.get("subject_titles") or []
    classification_titles = artwork.get("classification_titles") or []
    term_titles = artwork.get("term_titles") or []
    material_titles = artwork.get("material_titles") or []

    # Basic caption-like sentence
    bits = []

    if title:
        bits.append(title)

    # Simple one-line description
    desc_parts = []

    # subject / type
    # Try to pick one or two meaningful subject or classification tags
    subjects = [t for t in subject_titles if t] or [t for t in classification_titles if t]
    if subjects:
        desc_parts.append(subjects[0])

    if medium:
        desc_parts.append(medium)

    if artist:
        desc_parts.append(f"by {artist}")

    if date:
        desc_parts.append(f"from {date}")

    if place:
        desc_parts.append(f"({place})")

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

    print(text , "\n")
    return text


def embed_text(
    text: str,
    model: AutoModel,
    processor: AutoProcessor,
    device: torch.device,
) -> List[float]:
    if not text.strip():
        return []

    inputs = processor(
        text=[text],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        t = model.get_text_features(**inputs)  # [1, D]

    t = F.normalize(t, dim=-1)
    return t[0].detach().cpu().tolist()


def embed_artwork_text(
    model_bundle: Tuple[AutoModel, AutoProcessor, torch.device],
    artwork: Dict[str, Any],
) -> Tuple[int, List[float], str]:
    model, processor, device = model_bundle
    artwork_id = artwork.get("id")
    embedding_text = build_embedding_text(artwork)
    embedding = embed_text(embedding_text, model, processor, device)
    return artwork_id, embedding, embedding_text
