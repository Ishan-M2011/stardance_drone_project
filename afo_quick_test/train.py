
from pathlib import Path
import subprocess
import random
import shutil

print("Quick AFO test started!", flush=True)

# STEP 1: Install YOLO
subprocess.run(
    ["python", "-m", "pip", "install", "-q", "ultralytics"],
    check=True
)

from ultralytics import YOLO

# STEP 2: Find AFO dataset
folder = Path("/kaggle/input")
class_file = next(folder.rglob("6categories.names"), None)

if class_file is None:
    raise FileNotFoundError("AFO not attached")

part1 = class_file.parent
afo = part1.parent.parent

assert class_file.read_text().splitlines()[0].strip() == "human"

print("AFO dataset found!", flush=True)

# STEP 3: Find images from all 3 parts
files = [
    p for p in afo.rglob("*.jpg")
    if p.parent.name == "images"
]

images = {p.name: p for p in files}

if len(images) != len(files):
    raise ValueError("Duplicate image names")

# Read the official training and test lists
train_names = set((part1 / "train.txt").read_text().splitlines())
test_names = set((part1 / "test.txt").read_text().splitlines())

train_names -= test_names  # Never train on test images

# STEP 4: Split by video to avoid related frames mixing
videos = sorted({n.rsplit("_", 1)[0] for n in train_names})

if len(videos) < 2:
    raise ValueError("Not enough video groups")

random.Random(42).shuffle(videos)

val_videos = set(videos[:max(1, len(videos) // 5)])

train = sorted(
    n for n in train_names
    if n.rsplit("_", 1)[0] not in val_videos
)

val = sorted(
    n for n in train_names
    if n.rsplit("_", 1)[0] in val_videos
)

# Choose a small sample
random.Random(42).shuffle(train)
random.Random(42).shuffle(val)

train = train[:100]
val = val[:25]

if not train or not val:
    raise ValueError("Empty training or validation split")

print("Training images:", len(train), flush=True)
print("Validation images:", len(val), flush=True)

# STEP 5: Prepare human-only annotations
data = Path("/kaggle/temp/afo_quick")

for split, names in [("train", train), ("val", val)]:
    for name in names:

        if name not in images:
            raise FileNotFoundError(name)

        image = images[name]
        label = part1 / "6categories" / Path(name).with_suffix(".txt")

        if not label.exists():
            raise FileNotFoundError(label)

        # Create image link
        new_image = data / "images" / split / name
        new_image.parent.mkdir(parents=True, exist_ok=True)
        new_image.symlink_to(image.resolve())

        # Keep only human class 0
        boxes = [
            line for line in label.read_text().splitlines()
            if line.strip() and line.split()[0] == "0"
        ]

        new_label = data / "labels" / split / Path(name).with_suffix(".txt")
        new_label.parent.mkdir(parents=True, exist_ok=True)
        new_label.write_text("\n".join(boxes))

print("Human labels ready!", flush=True)

# STEP 6: Create YOLO dataset configuration
yaml = data / "data.yaml"

yaml.write_text(
    f"path: {data}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names: [person]\n"
)

# STEP 7: Train one epoch
print("Starting training!", flush=True)

model = YOLO("yolo11n.pt")

model.train(
    data=str(yaml),
    epochs=1,         # One training cycle
    imgsz=512,        # Smaller image size
    batch=4,          # Four images per batch
    device=0,         # Kaggle T4 GPU
    workers=0,        # Simpler data loading
    plots=False,      # Skip extra graphs
    project="/kaggle/working",
    name="afo_quick",
    seed=42
)

# STEP 8: Save the trained model
shutil.copy2(
    "/kaggle/working/afo_quick/weights/best.pt",
    "/kaggle/working/afo_quick_best.pt"
)

print("AFO QUICK TEST COMPLETE!", flush=True)
