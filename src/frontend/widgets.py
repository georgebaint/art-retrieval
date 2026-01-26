"""UI widgets for the art search application.

Provides reusable pygame-based UI components for text input,
toggle buttons, and artwork display cards.
"""
from __future__ import annotations
import pygame
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional

pygame.init()

Color = Tuple[int, int, int]


@dataclass
class ToggleOption:
    """A single option in a toggle group.
    
    Associates a display label with an internal value.
    """
    label: str  # Text shown to user
    value: str  # Internal identifier


class TextInput:
    """A simple text input field widget for pygame.
    
    Handles user text input with a visual placeholder when empty.
    Supports backspace and printable character input.
    """
    def __init__(
        self,
        rect: pygame.Rect,
        font: pygame.font.Font,
        text_color: Color = (0, 0, 0),
        bg_color: Color = (255, 255, 255),
        border_color: Color = (180, 180, 180),
        placeholder: str = "Type your search and press Enter...",
    ):
        self.rect = rect
        self.font = font
        self.text_color = text_color
        self.bg_color = bg_color
        self.border_color = border_color
        self.placeholder = placeholder

        self.text = ""
        self.active = False

    @property
    def value(self) -> str:
        return self.text

    def handle_event(self, event: pygame.event.Event) -> None:
        """Process pygame events for text input.
        
        Handles mouse clicks to activate/deactivate the input field,
        and keyboard input for typing and deletion.
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Activate input when user clicks inside the field
            self.active = self.rect.collidepoint(event.pos)

        if not self.active:
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Enter key triggers search; handled by the app
                pass
            elif event.key == pygame.K_BACKSPACE:
                # Delete last character
                self.text = self.text[:-1]
            else:
                # Add any printable character to the input
                if event.unicode.isprintable():
                    self.text += event.unicode

    def draw(self, surface: pygame.Surface) -> None:
        """Render the text input field to the screen.
        
        Draws the background, border, and either the user's text or
        a grayed-out placeholder when empty.
        """
        # Draw background and border
        pygame.draw.rect(surface, self.bg_color, self.rect)
        pygame.draw.rect(surface, self.border_color, self.rect, 2)

        # Show either user input or placeholder
        if self.text:
            render_text = self.text
            color = self.text_color
        else:
            render_text = self.placeholder
            color = (150, 150, 150)  # Grayed out for placeholder

        # Render text and position it with padding
        txt_surface = self.font.render(render_text, True, color)
        text_rect = txt_surface.get_rect()
        text_rect.left = self.rect.left + 8
        text_rect.centery = self.rect.centery

        surface.blit(txt_surface, text_rect)




class ToggleGroup:
    """A horizontal button group where only one option can be selected.
    
    Divides the available space equally among all options and highlights
    the currently selected one.
    """
    def __init__(
        self,
        options: List[ToggleOption],
        rect: pygame.Rect,
        font: pygame.font.Font,
        initial_value: Optional[str] = None,
    ):
        self.options = options
        self.rect = rect
        self.font = font

        # Calculate button positions by dividing space equally
        self.button_rects: List[pygame.Rect] = []
        total = len(options)
        if total == 0:
            return

        btn_width = rect.width // total
        x = rect.x
        for _ in options:
            self.button_rects.append(pygame.Rect(x, rect.y, btn_width, rect.height))
            x += btn_width

        # Initialize selection to provided value or first option
        if initial_value is not None:
            self.selected_value = initial_value
        else:
            self.selected_value = options[0].value

    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle mouse clicks to change the selected option."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check which button was clicked
            for opt, r in zip(self.options, self.button_rects):
                if r.collidepoint(event.pos):
                    self.selected_value = opt.value
                    break

    def draw(self, surface: pygame.Surface) -> None:
        """Render all toggle buttons with appropriate styling.
        
        Selected button is highlighted in blue; unselected in gray.
        """
        for opt, r in zip(self.options, self.button_rects):
            is_selected = (opt.value == self.selected_value)

            # Blue for selected, light gray for unselected
            bg = (60, 120, 200) if is_selected else (230, 230, 230)
            fg = (255, 255, 255) if is_selected else (50, 50, 50)
            border = (50, 50, 50)

            # Draw background, border, and label
            pygame.draw.rect(surface, bg, r)
            pygame.draw.rect(surface, border, r, 2)

            txt_surface = self.font.render(opt.label, True, fg)
            text_rect = txt_surface.get_rect(center=r.center)
            surface.blit(txt_surface, text_rect)

class PaintingCard:
    """A card widget displaying an artwork with thumbnail, title, and metadata.
    
    Shows the artwork image (if available), title, artist name, and
    the search mode that found it (text-only or hybrid).
    """
    def __init__(
        self,
        rect: pygame.Rect,
        title: str,
        subtitle: str,
        mode_label: str = "",
        image_surface: Optional[pygame.Surface] = None,
    ):
        self.rect = rect
        self.title = title
        self.subtitle = subtitle
        self.mode_label = mode_label
        self.image_surface = image_surface

    def draw(self, surface: pygame.Surface, font_title, font_sub) -> None:
        """Render the artwork card to the screen.
        
        Draws the card background, thumbnail image (or placeholder),
        title, artist name, and search mode label.
        """
        # Card background and border
        pygame.draw.rect(surface, (245, 245, 245), self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 1)

        # Thumbnail area takes up the top 55% of the card
        thumb_rect = pygame.Rect(
            self.rect.x + 8,
            self.rect.y + 8,
            self.rect.width - 16,
            int(self.rect.height * 0.55),
        )

        # Draw image or gray placeholder
        if self.image_surface is not None:
            thumb_img = pygame.transform.smoothscale(self.image_surface, thumb_rect.size)
            surface.blit(thumb_img, thumb_rect)
        else:
            pygame.draw.rect(surface, (210, 210, 210), thumb_rect)  # Gray placeholder

        # Artwork title below thumbnail
        title_surface = font_title.render(self.title, True, (20, 20, 20))
        title_rect = title_surface.get_rect()
        title_rect.topleft = (self.rect.x + 8, thumb_rect.bottom + 4)
        surface.blit(title_surface, title_rect)

        # Artist name below title
        subtitle_surface = font_sub.render(self.subtitle, True, (80, 80, 80))
        subtitle_rect = subtitle_surface.get_rect()
        subtitle_rect.topleft = (self.rect.x + 8, title_rect.bottom + 2)
        surface.blit(subtitle_surface, subtitle_rect)

        # Search mode label in bottom right corner
        if self.mode_label:
            ml_surface = font_sub.render(self.mode_label, True, (100, 100, 160))
            ml_rect = ml_surface.get_rect()
            ml_rect.bottomright = (self.rect.right - 6, self.rect.bottom - 4)
            surface.blit(ml_surface, ml_rect)
