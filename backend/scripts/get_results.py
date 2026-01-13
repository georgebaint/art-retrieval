import sys
from pathlib import Path

# Add the backend directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from src.embeddings.text_embedder import TextEmbeddingConfig, load_text_embedding_model
from src.embeddings.image_embedder import ImageEmbeddingConfig, load_image_embedding_model
import torch
import torch.nn.functional as F

client = chromadb.PersistentClient(path="data/chromadb/test_full1")
artworks_collection_text = client.get_collection("artwork_text_embeddings")
artworks_collection_image = client.get_collection("artwork_image_embeddings")


def get_results(query_text, query_image=None, mode="text", n_results=6):
    if mode == "text":
        text_model = load_text_embedding_model(TextEmbeddingConfig())
        query_embedding = text_model.encode(query_text, normalize_embeddings=True)
        collection = artworks_collection_text

    elif mode == "vision":
        if query_image is None:
            raise ValueError("query_image must be provided for vision mode")

        image_model, processor, device = load_image_embedding_model(ImageEmbeddingConfig())
        inputs = processor(
            images=[query_image],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            v = image_model.get_image_features(**inputs)  # <-- image features
        v = F.normalize(v, dim=-1)
        query_embedding = v[0].detach().cpu().tolist()
        collection = artworks_collection_image

    elif mode == "hybrid":
        print("hybrid mode")
        image_model, processor, device = load_image_embedding_model(ImageEmbeddingConfig())
        inputs = processor(
            text=[query_text],
            # images=[query_image],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            v = image_model.get_text_features(**inputs)  # <-- text features
        v = F.normalize(v, dim=-1)
        query_embedding = v[0].detach().cpu().tolist()
        collection = artworks_collection_image
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    # Debug: see what Chroma returns
    print("Chroma ids:", results.get("ids"))
    print("Chroma metadatas:", results.get("metadatas"))

    return results
