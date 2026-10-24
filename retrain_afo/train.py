
from pathlib import Path
import subprocess
import sys
import shutil
import random

print("Preparing AFO...", flush=True)

# Install AI libraries
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pillow"],
    check=True
)

from ultralytics import YOLO
from PIL import Image

INPUT = Path("/kaggle/input")
TEMP = Path("/kaggle/temp")
WORK = Path("/kaggle/working")

# STEP 1: Find model and dataset
old_model = next(INPUT.rglob("afo_best.pt"), None)
class_file = next(INPUT.rglob("6categories.names"), None)

if old_model is None or class_file is None:
    raise FileNotFoundError("AFO model or dataset missing")

part1 = class_file.parent
afo = part1.parent.parent

assert class_file.read_text().splitlines()[0].strip() == "human"

# STEP 2: Find images across three parts
images = {
    p.name: p for p in afo.rglob("*.jpg")
    if p.parent.name == "images"
}

# Read AFO image lists
def read_list(filename):
    return {
        Path(line.strip()).name
        for line in (part1 / filename).read_text().splitlines()
        if line.strip()
    }

train_names = read_list("train.txt")
test_names = read_list("test.txt")

if train_names & test_names:
    raise ValueError("Train and test lists overlap")

# STEP 3: Keep test videos separate
def video(name):
    return Path(name).stem.rsplit("_", 1)[0]

test_videos = {video(name) for name in test_names}

train_names = {
    name for name in train_names
    if name in images and video(name) not in test_videos
}

# Divide training videos into train and validation
videos = sorted({video(name) for name in train_names})
random.Random(42).shuffle(videos)

if len(videos) < 2:
    raise ValueError("Not enough video groups")

val_videos = set(videos[:max(1, len(videos) // 5)])

train = sorted(name for name in train_names if video(name) not in val_videos)
val = sorted(name for name in train_names if video(name) in val_videos)

# STEP 4: Prepare images and human labels locally
data = TEMP / "afo_people"

for split, names in [("train", train), ("val", val)]:

    if not names:
        raise ValueError("Empty " + split + " split")

    for i, name in enumerate(names):

        image = images[name]
        label = part1 / "6categories" / (Path(name).stem + ".txt")

        if not label.exists():
            raise FileNotFoundError(label)

        # Resize and save image locally
        new_image = data / "images" / split / name
        new_image.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(image) as img:
            img.thumbnail((1920, 1920))
            img.convert("RGB").save(new_image, quality=85)

        # Only keep human class 0
        boxes = [
            line for line in label.read_text().splitlines()
            if line.strip() and line.split()[0] == "0"
        ]

        new_label = data / "labels" / split / (Path(name).stem + ".txt")
        new_label.parent.mkdir(parents=True, exist_ok=True)
        new_label.write_text("\n".join(boxes))

        if (i + 1) % 500 == 0:
            print(split, i + 1, "images prepared", flush=True)

    print(split, "total images:", len(names), flush=True)

# STEP 5: Create YOLO configuration
data_yaml = data / "data.yaml"

data_yaml.write_text(
    f"path: {data}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names: [person]\n"
)

# STEP 6: Retrain existing AFO model
print("Retraining AFO...", flush=True)

model = YOLO(str(old_model))

model.train(
    data=str(data_yaml),
    epochs=5,
    imgsz=960,
    batch=8,
    workers=2,
    device=0,
    amp=True,
    plots=False,
    cls_remap=False,
    project=str(WORK),
    name="afo_retrained",
    exist_ok=True,
    seed=42
)

# STEP 7: Save trained model
new_model = WORK / "afo_retrained/weights/best.pt"

shutil.copy2(
    new_model,
    WORK / "afo_retrained_best.pt"
)

# STEP 8: Compare previous and retrained models
for name, weights in [("BEFORE", old_model), ("AFTER", new_model)]:

    metrics = YOLO(str(weights)).val(
        data=str(data_yaml),
        split="val",
        imgsz=960,
        batch=8,
        device=0,
        plots=False
    )

    print(
        name,
        "Precision:", round(metrics.box.mp * 100, 2),
        "Recall:", round(metrics.box.mr * 100, 2),
        "mAP50:", round(metrics.box.map50 * 100, 2),
        flush=True
    )

print("AFO RETRAINING COMPLETE!", flush=True)
