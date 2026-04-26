"""Quick script to count class IDs in label files and compare mappings."""
from pathlib import Path
from collections import Counter

# 1. Count class IDs in the original Roboflow dataset
roboflow_label_dir = Path(r"Weapon-Detection-using-YOLOv8-1/train/labels")
counter = Counter()
for f in roboflow_label_dir.glob("*.txt"):
    for line in f.read_text().strip().splitlines():
        parts = line.strip().split()
        if parts:
            counter[int(parts[0])] += 1

print("=== Roboflow dataset class distribution (train) ===")
for cls_id in sorted(counter):
    print(f"  Class {cls_id}: {counter[cls_id]} annotations")

# 2. Show the two different class mappings
print("\n=== Roboflow data.yaml class order ===")
roboflow_names = ["Handgun", "Knife", "Missile", "Rifle", "Shotgun", "Sword", "Tank"]
for i, name in enumerate(roboflow_names):
    print(f"  {i}: {name}")

print("\n=== config.py CLASS_NAMES ===")
config_names = {0: "Tank", 1: "Knife", 2: "Handgun", 3: "Rifle", 4: "Missile", 5: "Shotgun", 6: "Sword"}
for i in sorted(config_names):
    print(f"  {i}: {config_names[i]}")

print("\n=== processed/data.yaml class order ===")
processed_names = ["Tank", "Knife", "Handgun", "Rifle", "Missile", "Shotgun", "Sword"]
for i, name in enumerate(processed_names):
    print(f"  {i}: {name}")

# 3. Show the mismatch
print("\n=== MISMATCH ANALYSIS ===")
print("The label files use class IDs from the ROBOFLOW mapping.")
print("But config.py / processed data.yaml use a DIFFERENT mapping.")
print()
print("What the model LEARNED vs what inference DISPLAYS:")
for i in range(7):
    robo = roboflow_names[i]
    conf = config_names[i]
    match = "✓" if robo == conf else "✗ WRONG"
    print(f"  class_id={i}:  trained on '{robo}'  →  displayed as '{conf}'  {match}")
