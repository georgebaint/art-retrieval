import sys
from pathlib import Path

# Add the src directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests

from embeddings.image_embedder import embed_artwork_image
from embeddings.text_embedder import embed_artwork_text
from artic import create_artic_session


def generate_text_embeddings_for_artworks(
    artworks: Iterable[Dict[str, Any]],
    model_bundle: Any,
    # save_fn: Callable[[int, List[float], str, Any], None],
    collection: Any,
    *,
    progress_every: int = 100,
) -> Any:
    """
    Iterate over artworks and generate text embeddings (SigLIP text tower).

    Parameters
    ----------
    artworks : iterable of dict
        Iterable yielding artwork records (artworks.json 'data' objects).
    model_bundle : Any
        Tuple (model, processor, device) as returned by your SigLIP loader.
    save_fn : callable
        Callback to persist the result: (artwork_id, embedding, embedding_text).
        You plug in your DB or file-writing logic here.
    progress_every : int, optional
        How often to print a simple progress message (0 to disable).
    """
    count = 0
    saved = 0

    for artwork in artworks:
        count += 1
        try:
            artwork_id, embedding, embedding_text = embed_artwork_text(
                model_bundle,
                artwork,
            )
        except Exception as e:
            # Don't kill the whole run for one bad record
            print(f"[text] Error embedding artwork {artwork.get('id')}: {e}")
            continue

        if not embedding:
            # Empty embedding – probably empty text; skip
            continue

        try:
            collection.add(
                ids=[str(artwork_id)],
                documents=[embedding_text],
                embeddings=[embedding],
                metadatas=[{"type": "text"}],
            )
            saved += 1
        except Exception as e:
            print(f"[text] Error saving embedding for artwork {artwork_id}: {e}")
            continue

        if progress_every and count % progress_every == 0:
            print(f"[text] processed={count}, saved={saved}")

    print(f"[text] DONE: processed={count}, saved={saved}")

    return collection

def generate_image_embeddings_for_artworks(
    artworks: Iterable[Dict[str, Any]],
    model_bundle: Any,
    iiif_base_url: str,
    collection: Any,
    # save_fn: Callable[[int, List[float], Any], None],
    *,
    progress_every: int = 50,
    session: Optional[requests.Session] = None,
) -> Any:
    """
    Iterate over artworks and generate image embeddings (SigLIP image tower).

    Parameters
    ----------
    artworks : iterable of dict
        Iterable yielding artwork records (artworks.json 'data' objects).
    model_bundle : Any
        Tuple (model, processor, device) as returned by your SigLIP loader.
    iiif_base_url : str
        Base IIIF URL, e.g. 'https://www.artic.edu/iiif/2'.
    save_fn : callable
        Callback that persists the result: (artwork_id, image_embedding).
    progress_every : int, optional
        How often to print a simple progress message (0 to disable).
    session : requests.Session, optional
        Reusable HTTP session. If None, this function will create and close
        its own session; for large runs, pass a shared session in.
    """
    local_session = False
    if session is None:
        session = create_artic_session()
        local_session = True

    count = 0
    saved = 0

    try:
        for artwork in artworks:
            count += 1
            artwork_id = artwork.get("id")

            try:
                embedding = embed_artwork_image(
                    model_bundle,
                    artwork,
                    iiif_base_url=iiif_base_url,
                    session=session,
                )
            except Exception as e:
                print(f"[image] Error embedding artwork {artwork_id}: {e}")
                continue

            if embedding is None:
                # No image, not public domain, or download failed
                print(f"[image] Skipping artwork {artwork_id} (no embedding)")
                continue

            try:
                collection.add(
                    ids=[str(artwork_id)],
                    embeddings=[embedding],
                    metadatas=[{"type": "image"}],
                )

                saved += 1
            except Exception as e:
                print(f"[image] Error saving embedding for artwork {artwork_id}: {e}")
                continue

            if progress_every and count % progress_every == 0:
                print(f"[image] processed={count}, saved={saved}")

    finally:
        if local_session:
            session.close()

    print(f"[image] DONE: processed={count}, saved={saved}")

    return collection
