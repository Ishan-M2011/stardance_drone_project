
from pathlib import Path
from zipfile import ZipFile
import random
import shutil
import subprocess

print("Starting training!", flush=True)

# Install YOLO
subprocess.run(
    ["python", "-m", "pip", "install", "-q", "ultralytics"],
    check=True
)

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.split import split_classify_dataset

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")

# STEP 1: Find SARD and AFO
sard = next(
    (p.parent for p in INPUT.rglob("data.yaml")
     if p.parent.name == "search-and-rescue-2"), None
)

afo = INPUT / "afo-aerial-dataset-of-floating-objects"
part1 = afo / "PART_1/PART_1"

if sard is None or not (part1 / "6categories.names").exists():
    raise FileNotFoundError("SARD or AFO not found")

# Confirm AFO class 0 is human
assert (part1 / "6categories.names").read_text().splitlines()[0] == "human"

# STEP 2: Find and prepare AIDER
print("Checking AIDER...", flush=True)

classes = {
    "fire", "flooded_areas", "collapsed_building",
    "traffic_incident", "normal"
}

aider_folder = WORK / "aider_unpacked"

aider_zip = next(
    (p for p in INPUT.rglob("*.zip")
     if p.name.lower() == "aider.zip"), None
)

if aider_zip:
    with ZipFile(aider_zip) as z:
        z.extractall(aider_folder)
else:
    # If Kaggle already extracted AIDER, copy its class folders
    original = next(
        (p.parent for p in INPUT.rglob("normal")
         if p.is_dir()
         and all((p.parent / c).is_dir() for c in classes)),
        None
    )

    if original is None:
        raise FileNotFoundError("AIDER images or ZIP not found")

    shutil.copytree(original, aider_folder / "AIDER")

aider = next(
    (p.parent for p in aider_folder.rglob("normal")
     if p.is_dir()
     and all((p.parent / c).is_dir() for c in classes)),
    None
)

if aider is None:
    raise ValueError("AIDER class folders need inspection")

print("AIDER found!", flush=True)

# STEP 3: Download and prepare VisDrone
print("Preparing VisDrone...", flush=True)
vis = check_det_dataset("VisDrone.yaml")

# STEP 4: Prepare one combined human dataset
people = WORK / "people_data"

def save_person(image, label, split, source, human_ids):

    if not label.exists():
        raise FileNotFoundError(label)

    # Save image link to our new combined dataset
    new_image = people / "images" / split / f"{source}_{image.name}"
    new_label = people / "labels" / split / f"{source}_{image.stem}.txt"

    new_image.parent.mkdir(parents=True, exist_ok=True)
    new_label.parent.mkdir(parents=True, exist_ok=True)

    new_image.symlink_to(image.resolve())

    # Keep only human bounding boxes
    boxes = []

    for line in label.read_text().splitlines():
        parts = line.split()

        if not parts:
            continue

        if len(parts) != 5:
            raise ValueError(f"Invalid annotation: {label}")

        if int(parts[0]) in human_ids:
            boxes.append("0 " + " ".join(parts[1:]))

    new_label.write_text("\n".join(boxes))


# STEP 5: Add standard YOLO datasets
def add_dataset(name, folder, split, human_ids):

    count = 0

    for image in sorted(folder.iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        # Match each image with its annotation
        label = Path(
            str(image).replace("/images/", "/labels/")
        ).with_suffix(".txt")

        save_person(image, label, split, name, human_ids)
        count += 1

    print(name, split, count, "images", flush=True)


# SARD: All six classes describe people
add_dataset("sard", sard / "train/images", "train", set(range(6)))
add_dataset("sard", sard / "valid/images", "val", set(range(6)))

# VisDrone: Class 0 = pedestrian, class 1 = people
add_dataset("vis", Path(vis["train"]), "train", {0, 1})
add_dataset("vis", Path(vis["val"]), "val", {0, 1})

# STEP 6: Add AFO's three image parts
print("Preparing AFO...", flush=True)

# Find images across PART_1, PART_2, and PART_3
afo_images = {
    p.name: p for p in afo.rglob("*")
    if p.parent.name == "images"
    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
}

# Read original dataset split files
def read_names(path):
    return {
        Path(line.strip()).name
        for line in path.read_text().splitlines()
        if line.strip()
    }

train_names = read_names(part1 / "train.txt")
test_names = read_names(part1 / "test.txt")

# Use official validation split if present
val_file = next(
    (part1 / n for n in ["val.txt", "valid.txt", "validation.txt"]
     if (part1 / n).exists()), None
)

if val_file:
    val_names = read_names(val_file)
    train_names -= val_names
else:
    # Keep frames from the same video together
    videos = sorted({n.rsplit("_", 1)[0] for n in train_names})
    random.Random(42).shuffle(videos)

    val_videos = set(videos[:max(1, len(videos) // 5)])
    val_names = {
        n for n in train_names
        if n.rsplit("_", 1)[0] in val_videos
    }
    train_names -= val_names

# Never train on original test images
if (train_names & test_names) or (val_names & test_names):
    raise ValueError("AFO splits overlap")

# Add AFO humans: class 0
for split, names in [("train", train_names), ("val", val_names)]:

    for name in sorted(names):
        if name not in afo_images:
            raise FileNotFoundError(f"Missing AFO image: {name}")

        image = afo_images[name]
        label = part1 / "6categories" / Path(name).with_suffix(".txt")

        save_person(image, label, split, "afo", {0})

    print("AFO", split, len(names), "images", flush=True)

print("AFO test images reserved:", len(test_names), flush=True)

# STEP 7: Create combined dataset configuration
data_file = people / "data.yaml"

data_file.write_text(
    f"path: {people}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names: [person]\n"
)

# STEP 8: Train the combined human detector
print("Training human detector...", flush=True)

human = YOLO("yolo11n.pt")

human.train(
    data=str(data_file),
    epochs=2,          # Short test before full training
    imgsz=640,
    batch=8,
    device=0,
    project=str(WORK),
    name="human_training",
    exist_ok=True,
    seed=42
)

shutil.copy2(
    WORK / "human_training/weights/best.pt",
    WORK / "human_best.pt"
)

print("Human model saved!", flush=True)

# STEP 9: Train AIDER emergency classifier
print("Training emergency classifier...", flush=True)

aider_split = split_classify_dataset(aider, train_ratio=0.8)

emergency = YOLO("yolo11n-cls.pt")

emergency.train(
    data=str(aider_split),
    epochs=2,
    imgsz=224,
    batch=16,
    device=0,
    project=str(WORK),
    name="emergency_training",
    exist_ok=True,
    seed=42
)

shutil.copy2(
    WORK / "emergency_training/weights/best.pt",
    WORK / "emergency_best.pt"
)

# STEP 10: Remove temporary datasets, keep trained models
shutil.rmtree(people)
shutil.rmtree(aider_folder)
shutil.rmtree(aider_split)

print("BOTH MODELS SAVED!", flush=True)
