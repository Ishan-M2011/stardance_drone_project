
from pathlib import Path
import random
import shutil
import subprocess

print("Starting optimized AFO training!", flush=True) # Lets us know if it actually started

# STEP 1: Install YOLO and Pillow
subprocess.run(
    ["python", "-m", "pip", "install", "-q", "ultralytics", "pillow"],
    check=True
)

from ultralytics import YOLO
from PIL import Image

# STEP 2: Find AFO dataset
folder = Path("/kaggle/input")
class_file = next(folder.rglob("6categories.names"), None)

if class_file is None:
    raise FileNotFoundError("AFO dataset not attached") #In case dataset not properly attached

part1 = class_file.parent
afo = part1.parent.parent

assert class_file.read_text().splitlines()[0].strip() == "human"

print("AFO found!", flush=True)

# STEP 3: Find all images
image_files = [
    p for p in afo.rglob("*.jpg")
    if p.parent.name == "images"
]

images = {p.name: p for p in image_files}

if len(images) != len(image_files): # Like erro says, finding duplicate images, need this to prevent retraining or testing on trained images
    raise ValueError("Duplicate image filenames")

print("Available images:", len(images), flush=True)

# STEP 4: Read original train and test lists
def read_list(name):
    return {
        Path(line.strip()).name
        for line in (part1 / name).read_text().splitlines()
        if line.strip()
    }

train_names = read_list("train.txt")
test_names = read_list("test.txt")

if train_names & test_names:
    raise ValueError("Train and test images overlap")

# Use all available training images
missing = train_names - set(images)

if missing:
    print("Missing training images:", len(missing), flush=True)

train_names = train_names & set(images)

# STEP 5: Split by video for validation
def video_name(name):
    return name.rsplit("_", 1)[0]

# Keep test videos separate
test_videos = {video_name(n) for n in test_names}

train_names = {
    n for n in train_names
    if video_name(n) not in test_videos
}

videos = sorted({video_name(n) for n in train_names})

random.Random(42).shuffle(videos)

if len(videos) < 2:
    raise ValueError("Not enough video groups")

val_videos = set(videos[:max(1, len(videos) // 5)])

train = sorted(
    n for n in train_names if video_name(n) not in val_videos
)

val = sorted(
    n for n in train_names if video_name(n) in val_videos
)

print("Training images:", len(train), flush=True) 
print("Validation images:", len(val), flush=True) #prints out number of respective images

if not train or not val:
    raise ValueError("Empty training or validation split")

# STEP 6: Prepare local images and human labels
data = Path("/kaggle/temp/afo_full_local")

for split, names in [("train", train), ("val", val)]:

    for i, name in enumerate(names):

        image = images[name]
        label = part1 / "6categories" / Path(name).with_suffix(".txt")

        if not label.exists(): # if labels don't exist, skip the image
            raise FileNotFoundError(label)

        # Save a smaller working copy of the image
        new_image = data / "images" / split / name
        new_image.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(image) as img:
            img.thumbnail((1920, 1920))
            img.convert("RGB").save(new_image, quality=85)

        # Keep only human bounding boxes (class 0)
        boxes = []

        for line in label.read_text().splitlines():
            parts = line.split()

            if not parts:
                continue

            if len(parts) != 5:
                raise ValueError(f"Invalid label: {label}")

            if parts[0] == "0":
                boxes.append(line)

        new_label = data / "labels" / split / Path(name).with_suffix(".txt")
        new_label.parent.mkdir(parents=True, exist_ok=True)
        new_label.write_text("\n".join(boxes))

        if (i + 1) % 500 == 0:
            print(f"Prepared {split}: {i + 1}/{len(names)}", flush=True)

print("Dataset prepared!", flush=True)

# STEP 7: Create YOLO configuration
yaml = data / "data.yaml"

yaml.write_text(
    f"path: {data}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names: [person]\n"
)

# STEP 8: Train on the FULL dataset
print("Starting full YOLO training!", flush=True)

model = YOLO("yolo11n.pt")

model.train(
    data=str(yaml),       # Prepared dataset
    epochs=5,             # Five passes through training data
    imgsz=640,            # Training resolution
    batch=16,             # More images per batch
    workers=2,            # Two data-loading workers
    device=0,             # Kaggle T4 GPU
    amp=True,             # Faster mixed-precision training
    cache=False,          # Images already copied locally
    plots=False,          # Skip extra graphs
    save_period=1,        # Save checkpoint after each epoch
    project="/kaggle/working",
    name="afo_full_optimized",
    exist_ok=True,
    seed=42
)

# STEP 9: Save the best model
shutil.copy2(
    "/kaggle/working/afo_full_optimized/weights/best.pt",
    "/kaggle/working/afo_best.pt"
)

print("AFO FULL TRAINING COMPLETE!", flush=True) 
