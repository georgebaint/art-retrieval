"""Utility functions for the frontend module.

Includes image conversion and backend query functions that bridge
the UI and the embedding-based search backend.
"""
import sys
from pathlib import Path

import pygame

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.backend.query import get_results


def pil_to_surface(pil_img) -> pygame.Surface:
    """Convert a PIL.Image to a Pygame Surface.
    
    Handles the conversion between PIL's image format and pygame's Surface
    format for display on screen.
    """
    mode = pil_img.mode
    size = pil_img.size
    data = pil_img.tobytes()
    return pygame.image.fromstring(data, size, mode)


def search_backend(
    query: str,
    mode: str,
    n_results: int = 6,
):
    """Query the embedding database and format results for display.
    
    Calls the backend embedding system via get_results() and transforms
    the results into a flat list of artwork dictionaries with display information.
    
    Args:
        query: Text query from the user
        mode: Search mode ("text" for metadata search, "hybrid" for vision+text)
        n_results: Number of results to return
        
    Returns:
        List of artwork dictionaries with keys: id, title, artist, image_id
    """
    if not query.strip():
        return []

    # Query the embedding database (pass mode directly as backend supports it)
    results = get_results(query_text=query, mode=mode, n_results=n_results)

    # Extract IDs and metadata from Chroma results
    # Chroma returns lists per query; we only do one query at a time
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # Transform into display-friendly format
    artworks = []
    for art_id, meta in zip(ids, metadatas):
        title = meta.get("title") or f"Artwork {art_id}"
        artist = (
            meta.get("artist_title")
            or meta.get("artist_display")
            or "Unknown artist"
        )
        image_id = meta.get("image_id")

        artworks.append(
            {
                "id": art_id,
                "title": title,
                "artist": artist,
                "image_id": image_id,
            }
        )
    print(ids)
    return artworks

