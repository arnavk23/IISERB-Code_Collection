import os
import requests
from ddgs import DDGS
from tqdm import tqdm

CATEGORIES = [
    "cat", "dog", "car", "airplane", "flower",
    "tree", "building", "person", "food", "bicycle"
]

DATASET_DIR = "dataset"
IMAGES_PER_CATEGORY = 2000
MIN_IMAGE_SIZE = 50 * 1024

os.makedirs(DATASET_DIR, exist_ok=True)

def download_images(category, num_images):
    import time
    category_dir = os.path.join(DATASET_DIR, category)
    os.makedirs(category_dir, exist_ok=True)
    downloaded = 0
    with DDGS() as ddgs:
        try:
            results = ddgs.images(category, max_results=num_images*2)
        except Exception as e:
            print(f"Error fetching images for {category}: {e}")
            return
        for result in tqdm(results, desc=f"{category}"):
            url = result.get("image")
            if not url:
                continue
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200 and len(resp.content) > MIN_IMAGE_SIZE:
                    ext = url.split('.')[-1].split('?')[0]
                    fname = f"{downloaded:05d}.{ext}"
                    fpath = os.path.join(category_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(resp.content)
                    downloaded += 1
                if downloaded >= num_images:
                    break
            except Exception:
                time.sleep(1)
                continue

if __name__ == "__main__":
    for cat in CATEGORIES:
        download_images(cat, IMAGES_PER_CATEGORY)
    print("Download complete. Check the 'dataset' folder.")
