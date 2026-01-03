from dataclasses import dataclass
from typing import Optional
import requests
from PIL import Image
import io

@dataclass
class ArticConfig:
    iiif_base_url: str = "https://www.artic.edu/iiif/2"
    api_base_url: str = "https://api.artic.edu/api/v1"

def create_artic_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.artic.edu/",
    })
    try:
        session.get("https://www.artic.edu/", timeout=20)
    except Exception:
        pass
    return session

def build_iiif_url(image_id: str, iiif_base_url: str, size: str = "843,") -> str:
    return f"{iiif_base_url}/{image_id}/full/{size}/0/default.jpg"

def download_iiif_image(url: str, session: requests.Session) -> Optional[Image.Image]:
    try:
        resp = session.get(url, stream=True, timeout=20)
        if resp.status_code == 403:
            return None
        resp.raise_for_status()
    except Exception:
        return None

    try:
        img = Image.open(io.BytesIO(resp.content))
        return img.convert("RGB")
    except Exception:
        return None
