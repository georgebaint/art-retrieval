# Art Retrieval

A semantic search system for the Art Institute of Chicago's artwork collection. Find artworks by describing them or searching visually using AI-powered embeddings.

## What It Does

Search artworks by text or image using transformer models. The system converts queries and artworks into embeddings, then finds the most similar matches using ChromaDB.

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Build Database

```bash
python scripts/build_embeddings.py
```

Creates text and image embeddings from artwork data and stores them in ChromaDB.

### 3. Run GUI

```bash
python scripts/run_app.py
```

Or search programmatically:

```python
from src.backend.query import query_via_text, query_via_images

# Text search
results = query_via_text("impressionist landscape", n_results=10)

# Image search
results = query_via_images("abstract art", n_results=10)
```

## Project Structure

```
src/
├── backend/
│   ├── artic.py              - Art Institute API & IIIF downloader
│   ├── query.py              - Search functions
│   ├── utils.py              - Utilities
│   └── embeddings/
│       ├── text_embedder.py   - BGE model
│       └── image_embedder.py  - SigLIP2 model
├── frontend/
│   ├── app.py                - Pygame GUI
│   ├── widgets.py            - UI components
│   ├── image_manager.py      - Image handling
│   ├── constants.py          - UI config
│   └── utils.py              - Utilities

scripts/
├── build_embeddings.py       - Create embeddings database
├── evaluate.py               - Evaluation metrics
├── run_app.py                - Launch GUI

data/chromadb/               - Vector databases
notebooks/evaluation.ipynb   - Analysis notebook
```

## Models

- **Text**: `BAAI/bge-base-en-v1.5` (sentence-transformers)
- **Image**: `google/siglip2-base-patch16-naflex` (transformers)

## Troubleshooting

**Models download fails** - Check internet connection and disk space (500MB-2GB).

**Database not found** - Run `build_embeddings.py` first.

**Pygame won't start** - Check display access. Use programmatic API on headless systems.

## License

See [LICENSE](LICENSE) file for details.
