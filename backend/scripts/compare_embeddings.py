import sys
from pathlib import Path

# Add the backend directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import chromadb
from src.embeddings.text_embedder import TextEmbeddingConfig, load_text_embedding_model, embed_artwork_text
from src.embeddings.image_embedder import ImageEmbeddingConfig, load_image_embedding_model, embed_artwork_image
import torch
import torch.nn.functional as F

client = chromadb.PersistentClient(path="data/chromadb/test2") 
artworks_text = client.get_collection("artwork_text_embeddings")


query_text = f"Portrait of Edouard Molé"

text_model = load_text_embedding_model(TextEmbeddingConfig())
query_embedding = text_model.encode(query_text, normalize_embeddings=True)



artworks_image = client.get_collection("artwork_image_embeddings")
image_model, processor, device = load_image_embedding_model(ImageEmbeddingConfig())

# inputs = processor(
    
#     return_tensors="pt",
# ).to(device)

# with torch.no_grad():
#     t = image_model.get_text_features(**inputs)  # [1, D]

# t = F.normalize(t, dim=-1)

# query_embedding  = t[0].detach().cpu().tolist()

results = artworks_text.query(
    query_embeddings=[query_embedding],
    n_results=3,
)

print(results)