"""Download test fixture images.

Run once: python tests/generate_fixtures.py

Mix of real photographs (picsum.photos/Unsplash) and iconic test images
from various fields (Wikimedia Commons).
"""

import urllib.request
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURES.mkdir(exist_ok=True)

UA = "grape-test-fixtures/1.0 (testing)"

# --- Real photographs from picsum.photos ---
# Picsum IDs chosen for strong CLIP classification scores.
PICSUM = {
    "dog.jpg": 237,
    "cat.jpg": 593,
    "beach.jpg": 1041,
    "food.jpg": 429,
    "city.jpg": 1029,
    "forest.jpg": 1039,
}

PICSUM_SUBDIR = {
    "sunset.jpg": 399,
}


def download_picsum(name: str, picsum_id: int, dest: Path) -> None:
    path = dest / name
    if path.exists():
        print(f"  skip {path} (exists)")
        return
    url = f"https://picsum.photos/id/{picsum_id}/400/400.jpg"
    print(f"  {url} → {path}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as resp:
        path.write_bytes(resp.read())


# --- Iconic test images from Wikimedia Commons ---
# "Hello world" images from different fields, downloaded at native resolution.
WIKIMEDIA = {
    # Image processing — the OG test image (1973)
    "lenna.png": "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
    # Computer graphics — the Utah teapot (1975)
    "teapot.png": "https://upload.wikimedia.org/wikipedia/commons/5/5f/Utah_teapot_simple_2.png",
    # Machine learning — MNIST handwritten digits
    "mnist.png": "https://upload.wikimedia.org/wikipedia/commons/b/b1/MNIST_dataset_example.png",
    # Audio compression — Tom's Diner (the building whose song made MP3)
    "toms_diner.jpg": "https://upload.wikimedia.org/wikipedia/commons/d/df/Tom%27s_Restaurant_on_2880_Broadway%2C_New_York.JPG",
    # Cooking — the French omelette (chef's hello world)
    "omelette.jpg": "https://upload.wikimedia.org/wikipedia/commons/6/6e/Blond_unbrowned_omelet_with_mushrooms_and_herbs.jpg",
    # Woodworking — dovetail joint
    "dovetail.jpg": "https://upload.wikimedia.org/wikipedia/commons/2/2f/Finished_dovetail.jpg",
    # Blacksmithing — blacksmith at work
    "blacksmith.jpg": "https://upload.wikimedia.org/wikipedia/commons/c/c5/A_blacksmith_at_work.jpg",
}


def download_wikimedia(name: str, url: str, dest: Path) -> None:
    path = dest / name
    if path.exists():
        print(f"  skip {path} (exists)")
        return
    print(f"  {name} → {path}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as resp:
        path.write_bytes(resp.read())


# Download everything
print("Picsum photos:")
for name, pid in PICSUM.items():
    download_picsum(name, pid, FIXTURES)

print("\nWikimedia Commons:")
for name, url in WIKIMEDIA.items():
    download_wikimedia(name, url, FIXTURES)

# A non-image file (to test filtering)
(FIXTURES / "readme.txt").write_text("not an image")

# Subdirectory with an image (for recursive search tests)
sub = FIXTURES / "subdir"
sub.mkdir(exist_ok=True)
print("\nSubdir:")
for name, pid in PICSUM_SUBDIR.items():
    download_picsum(name, pid, sub)

print(f"\nFixtures written to {FIXTURES}")
