# Art Retrieval

A multi-modal search system for discovering artworks from the Art Institute of Chicago collection using text and image-based queries.

## Overview

This project combines advanced machine learning models with vector databases to enable intelligent art discovery. Users can search for artworks using:
- **Text queries** - Describe what you're looking for
- **Image queries** - Upload an image to find similar artworks
- **Hybrid search** - Combine both text and image inputs

## Features

- 🎨 **Multi-modal Search** - Search by text, image, or both
- 🔄 **Vector Embeddings** - Uses transformer models for semantic understanding
- 🗄️ **ChromaDB Integration** - Fast similarity search with persistent storage
- 🖼️ **IIIF Support** - Direct integration with Art Institute of Chicago's image server
- 🎮 **Interactive GUI** - Pygame-based frontend for easy exploration

## Project Structure

```
art-retrieval/
├── backend/
│   ├── scripts/
│   │   ├── build_embeddings.py    # Build embeddings from artwork data
│   │   └── get_results.py         # Query the vector database
│   └── src/
│       ├── artic.py              # Art Institute API integration
│       ├── api_calls.py           # API utilities
│       ├── utils.py               # Helper functions
│       └── embeddings/
│           ├── chroma_db.py       # Database management
│           ├── text_embedder.py   # Text embedding model
│           └── image_embedder.py  # Image embedding model
├── frontend/
│   ├── main.py                    # Main GUI application
│   └── widgets.py                 # UI components
└── data/
    └── chromadb/                  # Vector database storage
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd art-retrieval
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

   Required packages:
   - `torch` - Deep learning framework
   - `transformers` - Pre-trained ML models
   - `chromadb` - Vector database
   - `Pillow` - Image processing
   - `pygame` - GUI framework
   - `requests` - HTTP client

## Usage

### Building Embeddings

Generate embeddings for artwork data:

```bash
python backend/scripts/build_embeddings.py
```

This script:
- Loads artwork data from JSON files
- Creates text and image embeddings using transformer models
- Stores embeddings in ChromaDB for fast retrieval

### Running the GUI

Launch the interactive search interface:

```bash
python frontend/main.py
```

**Controls:**
- Type text queries to search by description
- Upload or paint images to search by visual similarity
- Toggle between search modes (text, vision, hybrid)
- Browse results with artwork images and metadata

### Querying Programmatically

```python
from backend.scripts.get_results import get_results

# Text-based search
results = get_results("landscape painting", mode="text", n_results=10)

# Image-based search
from PIL import Image
img = Image.open("path/to/image.jpg")
results = get_results("", query_image=img, mode="vision", n_results=10)
```

## Configuration

Key settings are defined in the embedding models:

- **TextEmbeddingConfig** - Text model hyperparameters
- **ImageEmbeddingConfig** - Vision model hyperparameters
- **CLIENT_SAVE_PATH** - Location for ChromaDB storage (default: `data/chromadb/test_full1`)

## Testing

Run tests to verify functionality:

```bash
pytest backend/tests/
```

## Data Source

Artwork data is sourced from the [Art Institute of Chicago API](https://api.artic.edu/). Images are served via their IIIF image server for efficient delivery.

## Requirements

- Python 3.11+
- CUDA-capable GPU (recommended for faster embedding generation)
- 4+ GB RAM
- Internet connection (for downloading models and images)

## License

See [LICENSE](LICENSE) file for details.
