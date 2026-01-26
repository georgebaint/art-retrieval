"""Helper classes and constants for the frontend."""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pygame


Color = Tuple[int, int, int]


@dataclass
class FontConfig:
    """Font configuration for the application."""
    ui_size: int = 18
    title_size: int = 16
    subtitle_size: int = 14
    font_name: str = "arial"
    
    def create_fonts(self):
        """Create pygame font objects from this configuration."""
        return {
            "ui": pygame.font.SysFont(self.font_name, self.ui_size),
            "title": pygame.font.SysFont(self.font_name, self.title_size, bold=True),
            "sub": pygame.font.SysFont(self.font_name, self.subtitle_size),
        }


@dataclass
class GridLayout:
    """Configuration for the card grid layout."""
    margin_x: int = 20
    margin_y: int = 120
    card_width: int = 220
    card_height: int = 220
    gap_x: int = 20
    gap_y: int = 20
    
    def get_cards_per_row(self, window_width: int) -> int:
        """Calculate how many cards fit per row given window width."""
        return max(
            1,
            (window_width - 2 * self.margin_x + self.gap_x) // (self.card_width + self.gap_x),
        )
    
    def get_card_position(self, idx: int, window_width: int) -> tuple:
        """Get (x, y) position for a card at index idx.
        
        Returns:
            Tuple of (x, y) position coordinates
        """
        per_row = self.get_cards_per_row(window_width)
        row = idx // per_row
        col = idx % per_row
        
        x = self.margin_x + col * (self.card_width + self.gap_x)
        y = self.margin_y + row * (self.card_height + self.gap_y)
        
        return (x, y)


# Mapping between UI modes and display labels
MODE_LABELS = {
    "text": "Text-only",
    "images": "Image Search",
}


def get_mode_label(mode: str) -> str:
    """Get the display label for a search mode."""
    return MODE_LABELS.get(mode, "")
