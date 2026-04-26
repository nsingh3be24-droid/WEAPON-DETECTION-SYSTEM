# preprocessing.py — Dataset preparation for the Weapon Detection System
#
# Responsibilities:
#   1. Validate raw images + YOLO labels
#   2. Resize / normalise images
#   3. Train / val / test split
#   4. Optional offline augmentation (blur, brightness, etc.)
#   5. Generate data.yaml for Ultralytics training

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from config import CLASS_NAMES, PREPROCESS_CFG, PROCESSED_DIR, RAW_DIR
from logger import get_logger

log = get_logger("preprocessing")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def resize_and_pad(
    image: np.ndarray,
    target: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Letterbox-resize image to `target` (w, h).
    Returns (padded_image, scale, (pad_left, pad_top)).
    """
    tw, th = target
    h, w   = image.shape[:2]
    scale  = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_left = (tw - nw) // 2
    pad_top  = (th - nh) // 2
    padded   = np.full((th, tw, 3), 114, dtype=np.uint8)
    padded[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
    return padded, scale, (pad_left, pad_top)


def normalise(image: np.ndarray) -> np.ndarray:
    """Normalise uint8 BGR image to float32 in [0,1]."""
    return image.astype(np.float32) / 255.0


def unnormalise(image: np.ndarray) -> np.ndarray:
    """Convert float32 [0,1] back to uint8."""
    return (image * 255).clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def augment_image(image: np.ndarray) -> list[np.ndarray]:
    """
    Return a list of augmented copies of `image` for offline dataset expansion.
    YOLO handles most augmentation online; these are for static dataset prep.
    """
    augmented = []

    # Horizontal flip
    augmented.append(cv2.flip(image, 1))

    # Random brightness / contrast
    alpha = random.uniform(0.75, 1.25)   # contrast
    beta  = random.randint(-30, 30)      # brightness
    augmented.append(cv2.convertScaleAbs(image, alpha=alpha, beta=beta))

    # Gaussian blur (simulate camera shake / low-res sensor)
    k = random.choice([3, 5])
    augmented.append(cv2.GaussianBlur(image, (k, k), 0))

    # Salt-and-pepper noise
    noisy = image.copy()
    n_pixels = random.randint(200, 800)
    coords_y = np.random.randint(0, image.shape[0], n_pixels)
    coords_x = np.random.randint(0, image.shape[1], n_pixels)
    noisy[coords_y, coords_x] = [255, 255, 255]
    augmented.append(noisy)

    return augmented


def flip_yolo_labels_h(labels: list[str]) -> list[str]:
    """Horizontally flip YOLO-format label lines (cx becomes 1-cx)."""
    flipped = []
    for line in labels:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = parts[0], parts[1], parts[2], parts[3], parts[4]
        flipped_cx = str(round(1.0 - float(cx), 6))
        flipped.append(f"{cls} {flipped_cx} {cy} {w} {h}")
    return flipped


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_sample(img_path: Path, label_path: Path) -> bool:
    """
    Return True if both the image and label are readable and consistent.
    Logs a warning for each issue found.
    """
    ok = True

    # Image check
    if not img_path.exists():
        log.warning("Missing image: %s", img_path)
        return False
    img = cv2.imread(str(img_path))
    if img is None:
        log.warning("Corrupt / unreadable image: %s", img_path)
        return False

    # Label check
    if not label_path.exists():
        log.warning("Missing label for: %s", img_path.name)
        return False

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            log.warning("%s line %d: expected 5 fields, got %d", label_path.name, i, len(parts))
            ok = False
            continue
        cls_id = int(parts[0])
        vals   = [float(p) for p in parts[1:]]
        if cls_id < 0 or cls_id >= len(CLASS_NAMES):
            log.warning("%s line %d: unknown class_id %d", label_path.name, i, cls_id)
            ok = False
        if any(not (0.0 <= v <= 1.0) for v in vals):
            log.warning("%s line %d: bbox value out of [0,1] range", label_path.name, i)
            ok = False

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT + COPY
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    raw_images_dir: Optional[Path] = None,
    raw_labels_dir: Optional[Path] = None,
    out_dir:        Optional[Path] = None,
    augment:        bool           = True,
    target_size:    tuple[int,int] = (640, 640),
) -> Path:
    """
    Build a YOLO-ready dataset from raw images + labels.

    Directory layout produced:
        out_dir/
            images/train/   images/val/   images/test/
            labels/train/   labels/val/   labels/test/
            data.yaml

    Returns the path to data.yaml.
    """
    raw_images_dir = raw_images_dir or RAW_DIR / "images"
    raw_labels_dir = raw_labels_dir or RAW_DIR / "labels"
    out_dir        = out_dir        or PROCESSED_DIR

    log.info("Building dataset from %s", raw_images_dir)

    # Collect valid samples
    img_paths   = sorted(p for p in raw_images_dir.iterdir()
                         if p.suffix.lower() in VALID_IMG_EXTS)
    valid_pairs: list[tuple[Path, Path]] = []
    for img_path in img_paths:
        lbl_path = raw_labels_dir / (img_path.stem + ".txt")
        if validate_sample(img_path, lbl_path):
            valid_pairs.append((img_path, lbl_path))

    log.info("Valid samples: %d / %d", len(valid_pairs), len(img_paths))
    if not valid_pairs:
        raise RuntimeError("No valid image/label pairs found — check RAW_DIR.")

    # Shuffle + split
    random.seed(PREPROCESS_CFG["seed"])
    random.shuffle(valid_pairs)
    n       = len(valid_pairs)
    n_val   = max(1, int(n * PREPROCESS_CFG["val_split"]))
    n_test  = max(1, int(n * PREPROCESS_CFG["test_split"]))
    n_train = n - n_val - n_test

    splits = {
        "train": valid_pairs[:n_train],
        "val":   valid_pairs[n_train:n_train + n_val],
        "test":  valid_pairs[n_train + n_val:],
    }
    log.info("Split → train:%d  val:%d  test:%d", n_train, n_val, n - n_train - n_val)

    # Copy (and optionally augment train)
    for split, pairs in splits.items():
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in pairs:
            img = cv2.imread(str(img_path))
            img_resized, _, _ = resize_and_pad(img, target_size)
            dst_img = img_out / img_path.name
            cv2.imwrite(str(dst_img), img_resized)
            shutil.copy(lbl_path, lbl_out / lbl_path.name)

            # Offline augmentation — train split only
            if augment and split == "train":
                aug_imgs = augment_image(img_resized)
                with open(lbl_path) as f:
                    orig_lines = f.readlines()

                for idx, aug in enumerate(aug_imgs):
                    stem = img_path.stem
                    aug_name = f"{stem}_aug{idx}{img_path.suffix}"
                    cv2.imwrite(str(img_out / aug_name), aug)

                    # Labels: flip horizontally for aug0, keep rest the same
                    if idx == 0:
                        aug_lines = flip_yolo_labels_h(orig_lines)
                    else:
                        aug_lines = orig_lines

                    with open(lbl_out / f"{stem}_aug{idx}.txt", "w") as f:
                        f.writelines(aug_lines)

        log.info("  [%s] %d images written to %s", split, len(pairs), img_out)

    # Write data.yaml
    yaml_path = out_dir / "data.yaml"
    data_yaml = {
        "path":  str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASS_NAMES),
        "names": [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    log.info("data.yaml written → %s", yaml_path)
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weapon Detection — Dataset Builder")
    parser.add_argument("--raw-images", type=Path, default=None)
    parser.add_argument("--raw-labels", type=Path, default=None)
    parser.add_argument("--out-dir",    type=Path, default=None)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    yaml_file = build_dataset(
        raw_images_dir=args.raw_images,
        raw_labels_dir=args.raw_labels,
        out_dir=args.out_dir,
        augment=not args.no_augment,
    )
    print(f"Dataset ready. data.yaml → {yaml_file}")
