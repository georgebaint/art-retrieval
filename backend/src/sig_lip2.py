import os
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

# ckpt = "google/siglip2-large-patch16-512"
ckpt = "google/siglip2-base-patch16-naflex"
# ckpt = "google/siglip2-so400m-patch16-naflex"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


model = AutoModel.from_pretrained(ckpt).to(device).eval()
processor = AutoProcessor.from_pretrained(ckpt)

def embed_text(prompt: str):
    prompt = prompt.strip().lower()
    # prompt = f"This is a photo of {prompt}."  # optional prompt engineering
    inputs = processor(
        text=[prompt],
        # padding="max_length",
        # truncation=True,
        # max_length=64, 
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        t = model.get_text_features(**inputs)
    return F.normalize(t, dim=-1)

def embed_image(img: Image.Image):
    inputs = processor(
        images=[img],
        # max_num_patches=256,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        v = model.get_image_features(**inputs)
    return F.normalize(v, dim=-1)

# prompt = "a room with blue walls, a bed, a window and a table"
# prompt = "a japanese painting of waves and a boat"
prompt = "people in a park with trees and grass"
t = embed_text(prompt)

dir_path = r"C:\Users\30698\Documents\art_project\art-retrieval\temp_test_output"
for name in os.listdir(dir_path):
    if name.endswith(".jpg"):
        img = Image.open(os.path.join(dir_path, name)).convert("RGB")
        v = embed_image(img)
        score = (t * v).sum(dim=-1).item()  
        print(name, score)