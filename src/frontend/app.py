"""Main application for the art search frontend.

This module contains the ArtSearchApp class which orchestrates the entire
user interface, event handling, image downloading, and integration with
the backend query system.
"""
import pygame
from dataclasses import dataclass

from src.frontend.widgets import TextInput, ToggleGroup, ToggleOption, PaintingCard
from src.frontend.constants import FontConfig, GridLayout, get_mode_label
from src.frontend.image_manager import ImageManager
from src.frontend.utils import search_backend


@dataclass
class AppConfig:
    """Configuration for the art search application."""
    window_width: int = 1000
    window_height: int = 700
    fps: int = 60
    n_results: int = 8


class ArtSearchApp:
    """Main application for searching and displaying artworks.
    
    Manages the pygame window, UI components, event handling, and
    communication with the backend embedding system to search for artworks.
    
    The app allows users to:
    - Enter text queries via the search input field
    - Choose between text-based and hybrid search modes
    - View results as artwork cards with thumbnails and metadata
    """

    def __init__(self, config: AppConfig = None):
        """Initialize the application with pygame and UI components."""
        self.config = config or AppConfig()
        
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.config.window_width, self.config.window_height)
        )
        pygame.display.set_caption("Art Search (Frontend Prototype)")

        self.clock = pygame.time.Clock()

        # Create fonts from configuration
        font_cfg = FontConfig()
        self.fonts = font_cfg.create_fonts()

        # Setup grid layout for cards
        self.grid_layout = GridLayout()

        # Initialize image manager (handles ARTIC config and sessions internally)
        self.image_manager = ImageManager()

        # Initialize UI components
        self._init_ui()

        self.cards = []
        self.running = True

    def _init_ui(self):
        """Initialize all UI components (search input and toggle buttons)."""
        # Text input field for search queries
        search_rect = pygame.Rect(20, 20, self.config.window_width - 40, 40)
        self.search_input = TextInput(
            rect=search_rect,
            font=self.fonts["ui"],
            placeholder="Search artworks (press Enter to search)...",
        )

        # Toggle group to switch between search modes
        toggle_rect = pygame.Rect(20, 70, 500, 30)
        self.toggles = ToggleGroup(
            options=[
                ToggleOption("Search by metadata", "text"),
                ToggleOption("Describe an artpiece", "images"),
            ],
            rect=toggle_rect,
            font=self.fonts["ui"],
            initial_value="text",
        )

    def _build_cards_from_artworks(self, artworks: list, mode: str):
        """Build painting cards from artwork data and download images.
        
        For each artwork, creates a card widget and downloads its thumbnail
        image from the ARTIC API, caching results to avoid re-downloading.
        """
        self.cards = []
        per_row = self.grid_layout.get_cards_per_row(self.config.window_width)

        for idx, art in enumerate(artworks):
            # Get card position from grid layout
            x, y = self.grid_layout.get_card_position(idx, self.config.window_width)
            rect = pygame.Rect(x, y, self.grid_layout.card_width, self.grid_layout.card_height)

            # Get image from manager (handles caching internally)
            image_id = art.get("image_id")
            image_surface = self.image_manager.get_image(image_id) if image_id else None

            # Create card widget with label from constants
            mode_label = get_mode_label(mode)

            card = PaintingCard(
                rect=rect,
                title=art["title"],
                subtitle=art["artist"],
                mode_label=mode_label,
                image_surface=image_surface,
            )
            self.cards.append(card)

    def _draw(self):
        """Render the entire application to the screen.
        
        Draws the background, header area with controls, artwork cards,
        and footer text.
        """
        # Fill screen with dark background
        self.screen.fill((30, 30, 40))

        # Draw header area with darker background
        pygame.draw.rect(self.screen, (45, 45, 60), (0, 0, self.config.window_width, 110))

        # Draw UI controls
        self.search_input.draw(self.screen)
        self.toggles.draw(self.screen)

        # Draw all artwork cards
        for card in self.cards:
            card.draw(self.screen, self.fonts["title"], self.fonts["sub"])

        # Draw footer text
        footer_text = "Press Enter to search."
        footer_surface = self.fonts["sub"].render(footer_text, True, (160, 160, 160))
        footer_rect = footer_surface.get_rect()
        footer_rect.bottomleft = (20, self.config.window_height - 10)
        self.screen.blit(footer_surface, footer_rect)

        pygame.display.flip()

    def handle_events(self):
        """Process pygame events (user input and window events).
        
        Handles:
        - Window close and escape key (exit app)
        - Text input to the search field
        - Toggle button clicks to change search mode
        - Enter key to perform search
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False

            # Delegate to UI components
            self.search_input.handle_event(event)
            self.toggles.handle_event(event)

            # Trigger search when user presses Enter
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                query = self.search_input.value.strip()
                mode = self.toggles.selected_value

                # Query backend and build cards from results
                artworks = search_backend(query, mode, n_results=self.config.n_results)
                self._build_cards_from_artworks(artworks, mode)

    def run(self):
        """Main application loop.
        
        Continuously:
        1. Tick the clock to maintain FPS
        2. Handle user input events
        3. Render the UI to the screen
        
        Exits when self.running is set to False.
        """
        while self.running:
            self.clock.tick(self.config.fps)
            self.handle_events()
            self._draw()

        pygame.quit()
