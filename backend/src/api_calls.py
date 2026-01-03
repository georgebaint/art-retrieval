import requests
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # avoids matplotlib backend issues in test runs
import matplotlib.pyplot as plt

SEARCH_URL = "https://api.artic.edu/api/v1/artworks/search"
IIIF = "https://www.artic.edu/iiif/2/{image_id}/full/843,/0/default.jpg"  # recommended pattern

def fetch_public_domain_artworks(limit=2):
	# Fetch more than we need, then filter down, because some hits can still have null image_id
	params = {
		"query[term][is_public_domain]": "true",
		"limit": max(10, limit * 5),
		"fields": "id,title,image_id,artist_display,date_display,is_public_domain",
	}
	resp = requests.get(SEARCH_URL, params=params, timeout=20)
	resp.raise_for_status()
	data = resp.json().get("data", [])
	data = [d for d in data if d.get("image_id")]
	return data[:limit]

import time
import requests

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.artic.edu/",
})

def download_image(image_id: str, out_path: str):
    url = IIIF.format(image_id=image_id)
    print(f"url = {url}")

    # optional warm-up to get whatever cookies the edge wants
    SESSION.get("https://www.artic.edu/", timeout=20)

    # time.sleep(1.0)  # be polite
    resp = SESSION.get(url, timeout=20)

    if resp.status_code == 403:
        print("403 body (first 300 chars):", resp.text[:300])  # now you'll actually see it
    resp.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path

# def download_image(image_id: str, out_path: str):
#     url = IIIF.format(image_id=image_id)
#     resp = requests.get(url, headers={"AIC-User-Agent": AIC_UA}, timeout=20)
#     resp.raise_for_status()
#     with open(out_path, "wb") as f:
#         f.write(resp.content)
#     return out_path

def plot_image(path: str, out_png: str):
	img = Image.open(path)
	plt.figure(figsize=(6, 6))
	plt.imshow(img)
	plt.axis("off")
	plt.savefig(out_png, bbox_inches="tight")
	plt.close()

def test_fetch_and_save(tmp_path):
	artworks = fetch_public_domain_artworks(limit=10)

	saved = []
	for art in artworks:
		image_id = art.get("image_id")
		out_jpg = tmp_path + "/" + f"{art['id']}.jpg"
		out_png = tmp_path + "/" + f"{art['id']}.png"
		try:
			download_image(image_id, str(out_jpg))
			# plot_image(str(out_jpg), str(out_png))
			saved.append(str(out_jpg))
		except requests.HTTPError as e:
			r = e.response
			print(f"IIIF download failed: {r.status_code} {r.reason} url={r.url}")
			print(f"Content-Type: {r.headers.get('content-type')}")

			continue
		
	print("artworks returned:", len(artworks))
	print("downloaded:", len(saved))

	assert saved, "No images were downloaded (API returned no usable image_ids or IIIF fetch failed)."
