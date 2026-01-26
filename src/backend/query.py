import chromadb
from src.backend.embeddings.text_embedder import TextEmbeddingConfig, load_text_embedding_model
from src.backend.embeddings.image_embedder import ImageEmbeddingConfig, load_image_embedding_model
import torch
import torch.nn.functional as F


def query_via_text(query_text: str, n_results: int = 6, db_path: str = "data/chromadb/test_full1"):
    """
    Query artwork embeddings using text-based search.
    
    Args:
        query_text: The text query to search for
        n_results: Number of results to return
        db_path: Path to the ChromaDB database
        
    Returns:
        Results from the text collection query
    """
    client = chromadb.PersistentClient(path=db_path)
    artworks_collection = client.get_collection("artwork_text_embeddings")
    
    text_model = load_text_embedding_model(TextEmbeddingConfig())
    query_embedding = text_model.encode(query_text, normalize_embeddings=True)

    try:
        results = artworks_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        # print("Chroma ids:", results.get("ids"))
        # print("Chroma metadatas:", results.get("metadatas"))

        return results
    except Exception as e:
        print(f"Error querying text collection: {e}")
        raise


def query_via_images(query_text: str, n_results: int = 6, db_path: str = "data/chromadb/test_full1"):
    """
    Query artwork embeddings using image-text search.
    
    Args:
        query_text: The text query to search with vision model
        n_results: Number of results to return
        db_path: Path to the ChromaDB database
        
    Returns:
        Results from the image collection query
    """
    client = chromadb.PersistentClient(path=db_path)
    artworks_collection = client.get_collection("artwork_image_embeddings")
    
    print("images mode")
    try:
        image_model, processor, device = load_image_embedding_model(ImageEmbeddingConfig())
        inputs = processor(
            text=[query_text],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            v = image_model.get_text_features(**inputs)
        v = F.normalize(v, dim=-1)
        query_embedding = v[0].detach().cpu().tolist()
    except Exception as e:
        print(f"Error processing images mode: {e}")
        raise

    try:
        results = artworks_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        # print("Chroma ids:", results.get("ids"))
        # print("Chroma metadatas:", results.get("metadatas"))

        return results
    except Exception as e:
        print(f"Error querying image collection: {e}")
        raise


def get_results(query_text, mode, n_results, db_path="data/chromadb/test_full1"):
    """
    Query artwork embeddings by text or image mode.
    
    Args:
        query_text: The text query to search for
        mode: "text" for text-based search or "images" for image-text search
        n_results: Number of results to return
        db_path: Path to the ChromaDB database
        
    Returns:
        Results from the collection query
    """
    if mode == "text":
        return query_via_text(query_text, n_results, db_path)
    elif mode == "images":
        return query_via_images(query_text, n_results, db_path)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'text' or 'images'.")

