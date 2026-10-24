
from pathlib import Path
import random
import shutil
import subprocess

print("Starting AFO training!", flush=True)

# Install YOLO
subprocess.run(["python", "-m", "pip", "install", "-q", "ultralytics"], check=True)
from ultralytics import YOLO

folder = Path("/kaggle/input")

part1 = next(
    (p.parent for p in folder.rglob("6categories.names")),
    None
)

if part1 is None:
    print("Attached folders:", list(folder.iterdir()))
    raise FileNotFoundError("AFO dataset not attached")

afo = part1.parent.parent  # Main AFO folder

print("AFO found:", afo, flush=True)

# Find images across all three parts
files = list(afo.rglob("*.jpg"))
images = {p.name: p for p in files}

if len(images) != len(files):
    raise ValueError("Duplicate AFO image names")

# STEP 2: Read training and test image lists
def read_list(name):
    return {
        Path(line.strip()).name
        for line in (part1 / name).read_text().splitlines()
        if line.strip()
    }

train = read_list("train.txt")
test = read_list("test.txt")

# Use original validation list if available
val_file = next(
    (p for p in ["val.txt", "valid.txt", "validation.txt"]
     if (part1 / p).exists()), None
)

if val_file:
    val = read_list(val_file)
    train -= val
else:
    # Keep images from the same video in one split
    videos = sorted({n.rsplit("_", 1)[0] for n in train})
    random.Random(42).shuffle(videos)

    if len(videos) < 2:
        raise ValueError("Not enough video groups")

    val_videos = set(videos[:max(1, len(videos) // 5)])
    val = {n for n in train if n.rsplit("_", 1)[0] in val_videos}
    train -= val

# Prevent training on test images
if (train & test) or (val & test) or (train & val):
    raise ValueError("Dataset splits overlap")

# STEP 3: Prepare person-only YOLO labels
data = Path("/kaggle/temp/afo_person")

for split, names in [("train", train), ("val", val)]:
    for name in sorted(names):

        if name not in images:
            raise FileNotFoundError(f"Missing image: {name}")

        image = images[name]
        label = part1 / "6categories" / Path(name).with_suffix(".txt")

        if not label.exists():
            raise FileNotFoundError(label)

        # Save an image link
        new_image = data / "images" / split / name
        new_image.parent.mkdir(parents=True, exist_ok=True)
        new_image.symlink_to(image)

        # Keep only human class 0
        boxes = [
            line for line in label.read_text().splitlines()
            if line.strip() and line.split()[0] == "0"
        ]

        new_label = data / "labels" / split / Path(name).with_suffix(".txt")
        new_label.parent.mkdir(parents=True, exist_ok=True)
        new_label.write_text("\n".join(boxes))

    print(split, "images:", len(names), flush=True)

print("Reserved test images:", len(test), flush=True)

# STEP 4: Create dataset configuration
yaml = data / "data.yaml"
yaml.write_text(
    f"path: {data}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names: [person]\n"
)

# STEP 5: Train YOLO
print("Training AFO model...", flush=True)

model = YOLO("yolo11n.pt")

model.train(
    data=str(yaml),
    epochs=2,              # Short test
    imgsz=640,             # Image resolution
    batch=8,               # Images per batch
    device=0,              # Kaggle GPU
    project="/kaggle/working",
    name="afo_training",
    seed=42
)

# STEP 6: Save trained model
shutil.copy2(
    "/kaggle/working/afo_training/weights/best.pt",
    "/kaggle/working/afo_best.pt"
)

print("AFO MODEL SAVED!", flush=True)
