# frontend/main.py

import sys
from pathlib import Path
import pygame

# If needed, adjust sys.path so "frontend" can be imported when running from project root
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from widgets import TextInput, ToggleGroup, ToggleOption, PaintingCard


WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
FPS = 60


def fake_search(query: str, mode: str):
    """
    Placeholder function simulating backend search.

    Parameters
    ----------
    query : str
        User query text.
    mode : str
        One of: "text", "hybrid", "vision".

    Returns
    -------
    List[dict]
        List of fake artworks with title/artist.
    """
    if not query:
        query = "Untitled"

    mode_label = {
        "text": "Text-only",
        "hybrid": "Text+Vision",
        "vision": "Vision-only",
    }.get(mode, "")

    results = []
    for i in range(6):  # 6 fake results
        results.append(
            {
                "title": f"{query} #{i+1}",
                "artist": f"Artist {chr(65 + i)}",
                "mode_label": mode_label,
            }
        )
    return results


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Art Search (Frontend Prototype)")

    clock = pygame.time.Clock()

    font_ui = pygame.font.SysFont("arial", 18)
    font_title = pygame.font.SysFont("arial", 16, bold=True)
    font_sub = pygame.font.SysFont("arial", 14)

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

    # Painting cards container
    cards = []

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # not really used yet, but fine to keep

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Basic ESC to quit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            # Delegate events
            search_input.handle_event(event)
            toggles.handle_event(event)

            # Trigger search on Enter
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                # The search_input already got the KEYPRESS; we just read its value.
                query = search_input.value.strip()
                mode = toggles.selected_value
                results = fake_search(query, mode)

                # Build cards from fake results
                cards = []
                margin_x = 20
                margin_y = 120
                card_width = 220
                card_height = 220
                gap_x = 20
                gap_y = 20
                per_row = max(1, (WINDOW_WIDTH - 2 * margin_x + gap_x) // (card_width + gap_x))

                for idx, r in enumerate(results):
                    row = idx // per_row
                    col = idx % per_row

                    x = margin_x + col * (card_width + gap_x)
                    y = margin_y + row * (card_height + gap_y)

                    rect = pygame.Rect(x, y, card_width, card_height)
                    card = PaintingCard(
                        rect=rect,
                        title=r["title"],
                        subtitle=r["artist"],
                        mode_label=r["mode_label"],
                    )
                    cards.append(card)

        # Draw
        screen.fill((30, 30, 40))

        # Panel background for search and toggles
        pygame.draw.rect(screen, (45, 45, 60), (0, 0, WINDOW_WIDTH, 110))

        search_input.draw(screen)
        toggles.draw(screen)

        # Draw cards
        for card in cards:
            card.draw(screen, font_title, font_sub)

        # Basic footer text
        footer_text = "Prototype frontend – backend wiring still to be implemented."
        footer_surface = font_sub.render(footer_text, True, (160, 160, 160))
        footer_rect = footer_surface.get_rect()
        footer_rect.bottomleft = (20, WINDOW_HEIGHT - 10)
        screen.blit(footer_surface, footer_rect)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
