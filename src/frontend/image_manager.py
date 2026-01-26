"""Image management for the frontend."""
from typing import Dict, Optional
import pygame

from src.backend.artic import (
    ArticConfig,
    create_artic_session,
    download_iiif_image,
    build_iiif_url,
)
from src.frontend.utils import pil_to_surface


class ImageManager:
    """Handles image downloading, caching, and conversion for display.
    
    Manages the lifecycle of artwork images: downloads from ARTIC,
    converts to pygame format, and caches to avoid re-downloading.
    """
    
    def __init__(self, artic_cfg: ArticConfig = None, artic_session=None):
        """Initialize the image manager.
        
        Args:
            artic_cfg: ARTIC API configuration (optional, uses default if not provided)
            artic_session: Requests session for HTTP calls (optional, creates new if not provided)
        """
        self.artic_cfg = artic_cfg or ArticConfig()
        self.artic_session = artic_session or create_artic_session()
        self.cache: Dict[str, pygame.Surface] = {}
    
    def get_image(self, image_id: str) -> Optional[pygame.Surface]:
        """Get an image by ID, downloading and caching if necessary.
        
        Args:
            image_id: The ARTIC image ID
            
        Returns:
            Pygame surface of the image, or None if download failed
        """
        # Return cached image if available
        if image_id in self.cache:
            return self.cache[image_id]
        
        # Download and cache
        url = build_iiif_url(
            image_id,
            iiif_base_url=self.artic_cfg.iiif_base_url,
        )
        print(f"Downloading image {image_id} from {url}...")
        
        pil_img = download_iiif_image(url, self.artic_session)
        if pil_img is None:
            return None
        
        try:
            image_surface = pil_to_surface(pil_img)
            self.cache[image_id] = image_surface
            return image_surface
        finally:
            pil_img.close()
    
    def clear_cache(self):
        """Clear all cached images."""
        self.cache.clear()
