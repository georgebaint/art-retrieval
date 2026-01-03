# for each row in the json file
# 1) Get the meta data
# 2) Download the image
# 3) Compute the embedding
# 4) Store everything in the DB

import json
import sqlite3
import os
import pickle
from PIL import Image
from api_calls import download_image
from sig_lip2 import embed_image

DB_PATH = "artworks.db"
IMAGES_DIR = "images"

def create_tables(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS artworks (
            id INTEGER PRIMARY KEY,
            title TEXT,
            artist TEXT,
            date TEXT,
            image_path TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            artwork_id INTEGER,
            embedding BLOB,
            FOREIGN KEY(artwork_id) REFERENCES artworks(id)
        )
    ''')
    conn.commit()

def process_json_file(json_file_path):
    with open(json_file_path, 'r') as f:
        artworks = json.load(f)

    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    for art in artworks:
        # 1) Get metadata
        art_id = art.get('id')
        title = art.get('title', '')
        artist = art.get('artist_title', '')
        department = art.get('department_title', '')
        reference_number = art.get('main_reference_number', '')
        # date = art.get('date_display', '')
        # image_id = art.get('image_id')

        if not art_id:
            print(f"Skipping artwork {art_id}: no image_id")
            continue

        # 2) Download the image
        image_path = os.path.join(IMAGES_DIR, f"{art_id}.jpg")
        result = download_image(image_id, image_path)
        if not result:
            print(f"Failed to download image for {art_id}")
            continue

        # 3) Compute the embedding
        try:
            img = Image.open(image_path).convert("RGB")
            embedding = embed_image(img)
            embedding_bytes = pickle.dumps(embedding.cpu().numpy())
        except Exception as e:
            print(f"Failed to compute embedding for {art_id}: {e}")
            continue

        # 4) Store everything in the DB
        conn.execute('''
            INSERT OR REPLACE INTO artworks (id, title, artist, date, image_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (art_id, title, artist, date, image_path))

        conn.execute('''
            INSERT INTO embeddings (artwork_id, embedding)
            VALUES (?, ?)
        ''', (art_id, embedding_bytes))

        print(f"Processed artwork {art_id}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Example usage
    process_json_file("artworks.json")

