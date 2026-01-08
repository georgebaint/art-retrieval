import sys
from pathlib import Path

# Add the backend directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
import json

from src.embeddings.text_embedder import TextEmbeddingConfig, load_text_embedding_model
from src.embeddings.image_embedder import ImageEmbeddingConfig, load_image_embedding_model
from src.embeddings.pipeline import (
    generate_text_embeddings_for_artworks,
    generate_image_embeddings_for_artworks,
)

# 1. Load artworks from your normalized file
def load_artworks(artworks_dir_path: str):
    artworks_dir = Path(artworks_dir_path)
    for json_path in artworks_dir.glob("*.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            print(f"Loading artworks from {json_path}...")
            try:
                # Try loading as JSON array
                data = json.load(f)
                if isinstance(data, list):
                    for artwork in data:
                        yield artwork
                else:
                    # If single object, yield it
                    yield data
            except json.JSONDecodeError:
                # If not valid JSON, try as JSONL
                f.seek(0)
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            artwork = obj.get("data", obj)
                            yield artwork
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num} in {json_path}: {e}")
                            continue

artworks_iter = list(load_artworks(r"C:\Users\30698\Documents\art_project\api-data\json\artworks"))  # Directory with 10 JSON files
print(f"Loaded {len(artworks_iter)} artworks for embedding.")
# 2. Load models
cfg_text = TextEmbeddingConfig()
model_bundle_text = load_text_embedding_model(cfg_text)

cfg_image = ImageEmbeddingConfig()
model_bundle_image = load_image_embedding_model(cfg_image)

# 3. Create DB collection for embeddings
import chromadb

# client = chromadb.Client()
client = chromadb.PersistentClient(path="data/chromadb/test2")

# client = chromadb.PersistentClient(path="data/chromadb/test1") 
# artworks_text = client.get_collection("artworks_text_embeddings")

empty_text_collection = client.get_or_create_collection(
    name="artwork_text_embeddings"
)

empty_image_collection = client.get_or_create_collection(
    name="artwork_image_embeddings"
)

# 4. Generate text embeddings
text_collection = generate_text_embeddings_for_artworks(
    artworks=artworks_iter,
    model_bundle=model_bundle_text,
    collection=empty_text_collection,
)

# 5. Generate image embeddings
image_collection = generate_image_embeddings_for_artworks(
    artworks=artworks_iter,
    model_bundle=model_bundle_image,
    iiif_base_url=cfg_image.iiif_base_url,
    collection=empty_image_collection,
)

print("Embedding process completed.")


# print(image_collection.peek())
# results = text_collection.query(
#     query_texts=["which paintings show the interior of a church?"],
#     n_results=2
# )

# print(results)