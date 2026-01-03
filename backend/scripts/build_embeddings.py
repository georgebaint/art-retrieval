import sys
from pathlib import Path

# Add the backend directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
import json

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
# 2. Load SigLIP model once
cfg = ImageEmbeddingConfig()
model_bundle = load_image_embedding_model(cfg)

# 3. Define how to save embeddings
def save_text_embedding(artwork_id, embedding, embedding_text):
    # TODO: replace with DB insert / write to file
    pass

def save_image_embedding(artwork_id, embedding):
    # TODO: replace with DB insert / write to file
    pass

# 4. Generate text embeddings
generate_text_embeddings_for_artworks(
    artworks=artworks_iter,
    model_bundle=model_bundle,
    save_fn=save_text_embedding,
)

# 5. Generate image embeddings
generate_image_embeddings_for_artworks(
    artworks=artworks_iter,
    model_bundle=model_bundle,
    iiif_base_url=cfg.iiif_base_url,
    save_fn=save_image_embedding,
)
