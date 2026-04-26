from roboflow import Roboflow
import shutil
from pathlib import Path

# Your API key — get it from https://app.roboflow.com → Settings → API
API_KEY = "BxrXCt3fqnz0bQyglM7W"

rf      = Roboflow(api_key=BxrXCt3fqnz0bQyglM7W) # type: ignore
project = rf.workspace("weopon-detection").project("weapon-detection-using-yolov8")
version = project.version(1)
dataset = version.download("yolov8")

# Auto-copy into project folders
src      = Path(dataset.location)
dst_imgs = Path("data/raw/images")
dst_lbls = Path("data/raw/labels")
dst_imgs.mkdir(parents=True, exist_ok=True)
dst_lbls.mkdir(parents=True, exist_ok=True)

count_i = count_l = 0
for split in ["train", "valid", "test"]:
    img_dir = src / split / "images"
    lbl_dir = src / split / "labels"
    if img_dir.exists():
        for f in img_dir.glob("*"):
            shutil.copy(f, dst_imgs / f.name)
            count_i += 1
    if lbl_dir.exists():
        for f in lbl_dir.glob("*.txt"):
            shutil.copy(f, dst_lbls / f.name)
            count_l += 1

print(f"✅ {count_i} images  → data/raw/images/")
print(f"✅ {count_l} labels  → data/raw/labels/")
print("▶ Now run: python main.py preprocess")