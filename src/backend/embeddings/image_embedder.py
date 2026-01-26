from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import io

import requests
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from src.backend.artic import build_iiif_url, download_iiif_image, create_artic_session

DEFAULT_CKPT = "google/siglip2-base-patch16-naflex"

@dataclass
class ImageEmbeddingConfig:
    """
    Configuration for the image embedding model.
    """
    model_ckpt: str = DEFAULT_CKPT
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    iiif_base_url: str = "https://www.artic.edu/iiif/2"
    # maybe add more config options later


def load_image_embedding_model(
    config: ImageEmbeddingConfig,
) -> tuple[AutoModel, AutoProcessor, torch.device]:
    """
    Load and return the image embedding model, processor and device.

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


def build_image_url(artwork: Dict[str, Any], iiif_base_url: str) -> Optional[str]:
    image_id = artwork.get("image_id")
    if not image_id:
        return None
    if not artwork.get("is_public_domain", False):
        return None
    return build_iiif_url(image_id, iiif_base_url)


def embed_image(
    img: Image.Image,
    model: AutoModel,
    processor: AutoProcessor,
    device: torch.device,
) -> List[float]:
    """
    Compute an embedding vector for a single image using SigLIP.

    Parameters
    ----------
    img : PIL.Image.Image
        Image to embed.
    model : AutoModel
        Image embedding model (SigLIP here).
    processor : AutoProcessor
        Corresponding processor.
    device : torch.device
        Device on which the model is loaded.

    Returns
    -------
    List[float]
        The normalized image embedding as a 1D list of floats.
    """
    inputs = processor(
        images=[img],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        v = model.get_image_features(**inputs)  # shape: [1, D]

    v = F.normalize(v, dim=-1)  # still [1, D]
    return v[0].detach().cpu().tolist()


def embed_artwork_image(
    model_bundle: Any,
    artwork: Dict[str, Any],
    iiif_base_url: str,
    session: Optional[requests.Session] = None,
) -> Optional[List[float]]:
    model, processor, device = model_bundle

    url = build_image_url(artwork, iiif_base_url)
    if not url:
        return None

    local_session = False
    if session is None:
        session = create_artic_session()
        local_session = True

    img = download_iiif_image(url, session)
    if img is None:
        if local_session:
            session.close()
        return None

    try:
        embedding = embed_image(img, model, processor, device)
    finally:
        img.close()
        if local_session:
            session.close()

    return embedding
