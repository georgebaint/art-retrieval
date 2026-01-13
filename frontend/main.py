import sys
from pathlib import Path
from typing import Dict, Optional

import pygame

# Add project root to sys.path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from frontend.widgets import TextInput, ToggleGroup, ToggleOption, PaintingCard
from backend.scripts.get_results import get_results
from backend.src.artic import (
    ArticConfig,
    create_artic_session,
    download_iiif_image,
    build_iiif_url,
)

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
FPS = 60

N_RESULTS = 15


def pil_to_surface(pil_img) -> pygame.Surface:
    """Convert a PIL.Image to a Pygame Surface."""
    mode = pil_img.mode
    size = pil_img.size
    data = pil_img.tobytes()
    return pygame.image.fromstring(data, size, mode)


def search_backend(
    query: str,
    mode: str,
    n_results: int = 6,
):
    """
    Call the embedding DB via get_results and convert into a flat list
    of artwork dicts:
        { "id", "title", "artist", "image_id" }
    """
    if not query.strip():
        return []

    # Map UI mode -> backend mode
    if mode == "text":
        backend_mode = "text"
    elif mode == "vision":
        backend_mode = "vision"
        # For now, vision mode requires an image query; you haven't wired that yet.
        # So you can either disable vision in the UI or treat it as text.
        # For now we just fallback:
        backend_mode = "text"
    elif mode == "hybrid":
        # Hybrid not implemented yet; fallback to text retrieval
        backend_mode = "hybrid"
    else:
        backend_mode = "text"

    results = get_results(query_text=query, mode=backend_mode, n_results=n_results)

    # Chroma returns lists per query -> we only do one query at a time
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    artworks = []
    for art_id, meta in zip(ids, metadatas):
        title = meta.get("title") or f"Artwork {art_id}"
        artist = (
            meta.get("artist_title")
            or meta.get("artist_display")
            or "Unknown artist"
        )
        image_id = meta.get("image_id")  # must exist in your embedding pipeline

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


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Art Search (Frontend Prototype)")

    clock = pygame.time.Clock()

    font_ui = pygame.font.SysFont("arial", 18)
    font_title = pygame.font.SysFont("arial", 16, bold=True)
    font_sub = pygame.font.SysFont("arial", 14)

    # ArtIC config + session (shared)
    artic_cfg = ArticConfig()
    artic_session = create_artic_session()

    # In-memory image cache: image_id -> pygame.Surface
    image_cache: Dict[str, pygame.Surface] = {}

    # Search bar area
    search_rect = pygame.Rect(20, 20, WINDOW_WIDTH - 40, 40)
    search_input = TextInput(
        rect=search_rect,
        font=font_ui,
        placeholder="Search artworks (press Enter to search)...",
    )

    # Toggle group (mode selection)
    toggle_rect = pygame.Rect(20, 70, 400, 30)
    toggles = ToggleGroup(
        options=[
            ToggleOption("Text only", "text"),
            ToggleOption("Text + Vision", "hybrid"),
            ToggleOption("Vision only", "vision"),
        ],
        rect=toggle_rect,
        font=font_ui,
        initial_value="text",
    )

    cards = []

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # not used yet

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            search_input.handle_event(event)
            toggles.handle_event(event)

            # Trigger search on Enter
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                query = search_input.value.strip()
                mode = toggles.selected_value

                artworks = search_backend(query, mode, n_results=N_RESULTS)

                # Build cards from artworks
                cards = []
                margin_x = 20
                margin_y = 120
                card_width = 220
                card_height = 220
                gap_x = 20
                gap_y = 20
                per_row = max(
                    1,
                    (WINDOW_WIDTH - 2 * margin_x + gap_x) // (card_width + gap_x),
                )

                for idx, art in enumerate(artworks):
                    row = idx // per_row
                    col = idx % per_row

                    x = margin_x + col * (card_width + gap_x)
                    y = margin_y + row * (card_height + gap_y)

                    rect = pygame.Rect(x, y, card_width, card_height)

                    image_surface: Optional[pygame.Surface] = None

                    image_id = art.get("image_id")
                    if image_id:
                        if image_id in image_cache:
                            image_surface = image_cache[image_id]
                        else:
                            url = build_iiif_url(
                                image_id,
                                iiif_base_url=artic_cfg.iiif_base_url,
                            )
                            print(f"Downloading image {image_id} from {url}...")
                            pil_img = download_iiif_image(url, artic_session)
                            if pil_img is not None:
                                image_surface = pil_to_surface(pil_img)
                                pil_img.close()
                                image_cache[image_id] = image_surface

                    mode_label = {
                        "text": "Text-only",
                        "hybrid": "Text+Vision (TODO)",
                        "vision": "Vision-only (TODO)",
                    }.get(mode, "")

                    card = PaintingCard(
                        rect=rect,
                        title=art["title"],
                        subtitle=art["artist"],
                        mode_label=mode_label,
                        image_surface=image_surface,
                    )
                    cards.append(card)

        # Draw
        screen.fill((30, 30, 40))

        pygame.draw.rect(screen, (45, 45, 60), (0, 0, WINDOW_WIDTH, 110))

        search_input.draw(screen)
        toggles.draw(screen)

        for card in cards:
            card.draw(screen, font_title, font_sub)

        footer_text = "Press Enter to search. Vision/Hybrid modes are partially wired; backend still reuses text for now."
        footer_surface = font_sub.render(footer_text, True, (160, 160, 160))
        footer_rect = footer_surface.get_rect()
        footer_rect.bottomleft = (20, WINDOW_HEIGHT - 10)
        screen.blit(footer_surface, footer_rect)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
