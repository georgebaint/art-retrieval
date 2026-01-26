"""Simple script to display a few examples from the image embedding database."""
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))

CLIENT_SAVE_PATH = "data/chromadb/test_full1"


def main():
    # Connect to the database
    client = chromadb.PersistentClient(path=CLIENT_SAVE_PATH)
    image_collection = client.get_or_create_collection(name="artwork_image_embeddings")
    
    # Get total count
    count = image_collection.count()
    print(f"Total artworks in image database: {count}\n")
    
    # Get a few examples
    print("=" * 80)
    print("SAMPLE IMAGE EMBEDDINGS")
    print("=" * 80)
    
    # Retrieve first 3 items (include_embeddings=True to get the vectors)
    results = image_collection.get(limit=3, include=["embeddings", "metadatas"])
    
    for i, (artwork_id, metadata) in enumerate(
        zip(results["ids"], results["metadatas"]), 1
    ):
        embedding = results["embeddings"][i-1] if results["embeddings"] is not None else None
        print(f"\n[Example {i}]")
        print(f"Artwork ID: {artwork_id}")
        print(f"Title: {metadata.get('title')}")
        print(f"Artist: {metadata.get('artist_title')}")
        print(f"Image ID: {metadata.get('image_id')}")
        print(f"\nImage Embedding Vector (first 5 dims):")
        print("-" * 40)
        if embedding is not None:
            print(f"Vector length: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
        else:
            print("(No embedding data)")
        print("-" * 40)


if __name__ == "__main__":
    main()
