import sys
from pathlib import Path
import json

from tqdm import tqdm
import chromadb

# Add the backend/src directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.text_embedder import (
    TextEmbeddingConfig,
    load_text_embedding_model,
    embed_artwork_text,
)
from src.embeddings.image_embedder import (
    ImageEmbeddingConfig,
    load_image_embedding_model,
    embed_artwork_image,
)
from src.artic import create_artic_session

CLIENT_SAVE_PATH = "data/chromadb/test_full1"
ARTWORKS_DIR_PATH = r"C:\Users\30698\Documents\art_project\full-api\artworks"


def load_artworks(artworks_dir_path: str):
    """
    Stream artworks from a directory of JSON / JSONL files.
    """
    artworks_dir = Path(artworks_dir_path)
    for json_path in artworks_dir.glob("*.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for artwork in data:
                        yield artwork
                else:
                    yield data
            except json.JSONDecodeError:
                f.seek(0)
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        artwork = obj.get("data", obj)
                        yield artwork
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {json_path}: {e}")
                        continue


def main():
    # 1. Load models once
    cfg_text = TextEmbeddingConfig()
    text_model = load_text_embedding_model(cfg_text)

    cfg_image = ImageEmbeddingConfig()
    image_model_bundle = load_image_embedding_model(cfg_image)

    # 2. Create / get Chroma collections
    client = chromadb.PersistentClient(path=CLIENT_SAVE_PATH)

    text_collection = client.get_or_create_collection(
        name="artwork_text_embeddings"
    )
    image_collection = client.get_or_create_collection(
        name="artwork_image_embeddings"
    )

    # 3. Shared HTTP session for image downloads
    session = create_artic_session()

    # 4. Loop over artworks once and do both text + image embeddings
    for artwork in tqdm(load_artworks(ARTWORKS_DIR_PATH),
                        desc="Embedding artworks",
                        unit="artwork"):
        art_id = artwork.get("id")

        # --- HOTFIX: require image_id and sanitize title/artist ---
        image_id = artwork.get("image_id")
        if image_id is None:
            # No image -> skip both text & image embeddings for this artwork
            continue

        title = artwork.get("title") or "Unknown title"
        artist_title = artwork.get("artist_title") or "Unknown artist"

        meta = {
            "image_id": image_id,         # guaranteed non-None
            "title": title,               # guaranteed non-None str
            "artist_title": artist_title, # guaranteed non-None str
        }
        # -----------------------------------------------------------

        # Text embedding (BGE or SigLIP-text)
        try:
            embedding_text_vec, embedding_text = embed_artwork_text(
                text_model,
                artwork,
            )
        except Exception as e:
            print(f"[text] Error embedding artwork {art_id}: {e}")
            embedding_text_vec = None

        if embedding_text_vec:
            try:
                text_collection.add(
                    ids=[str(art_id)],
                    documents=[embedding_text],
                    embeddings=[embedding_text_vec],
                    metadatas=[meta],   # uses sanitized meta
                )
            except Exception as e:
                print(f"[text] Error saving embedding for artwork {art_id}: {e}")

        # Image embedding (SigLIP image tower)
        try:
            embedding_img = embed_artwork_image(
                image_model_bundle,
                artwork,
                iiif_base_url=cfg_image.iiif_base_url,
                session=session,
            )
        except Exception as e:
            print(f"[image] Error embedding artwork {art_id}: {e}")
            embedding_img = None

        if embedding_img:
            try:
                image_collection.add(
                    ids=[str(art_id)],
                    embeddings=[embedding_img],
                    metadatas=[meta],   # same sanitized meta
                )
            except Exception as e:
                print(f"[image] Error saving embedding for artwork {art_id}: {e}")
                continue

    session.close()
    print("Embedding process completed.")


if __name__ == "__main__":
    main()
