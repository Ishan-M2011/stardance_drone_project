
from pathlib import Path
import subprocess
import sys
import shutil

print("Preparing VisDrone...", flush=True)

# Install YOLO
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ultralytics"],
    check=True
)

import yaml
from ultralytics import YOLO
from ultralytics.utils import ROOT
from ultralytics.data.utils import check_det_dataset

INPUT = Path("/kaggle/input")
TEMP = Path("/kaggle/temp")
WORK = Path("/kaggle/working")
TEMP.mkdir(parents=True, exist_ok=True)  # Create temp folder if missing

# STEP 1: Find our previously trained model
old_model = next(INPUT.rglob("visdrone_best.pt"), None)

if old_model is None:
    raise FileNotFoundError("Previous VisDrone model not found")

# STEP 2: Download VisDrone into temporary storage
config = yaml.safe_load(
    (ROOT / "cfg/datasets/VisDrone.yaml").read_text()
)

config["path"] = str(TEMP / "VisDrone")

source_yaml = TEMP / "visdrone_source.yaml"
source_yaml.write_text(yaml.safe_dump(config))

vis = check_det_dataset(str(source_yaml))

print("VisDrone downloaded!", flush=True)

# STEP 3: Prepare human-only annotations
data = TEMP / "visdrone_people"

for split in ["train", "val"]:
    image_folder = Path(vis[split])
    count = 0

    for image in sorted(image_folder.glob("*.jpg")):

        # Find image's original annotation
        label = image_folder.parent.parent / "labels" / split / (image.stem + ".txt")

        lines = label.read_text().splitlines() if label.exists() else []

        # Keep only pedestrian and people classes
        boxes = []

        for line in lines:
            parts = line.split()

            if len(parts) == 5 and parts[0] in ["0", "1"]:
                boxes.append("0 " + " ".join(parts[1:]))

        # Link original image
        new_image = data / "images" / split / image.name
        new_label = data / "labels" / split / (image.stem + ".txt")

        new_image.parent.mkdir(parents=True, exist_ok=True)
        new_label.parent.mkdir(parents=True, exist_ok=True)

        new_image.symlink_to(image.resolve())
        new_label.write_text("\n".join(boxes))

        count += 1

    print(split, "images prepared:", count, flush=True)

    if count == 0:
        raise ValueError("No images found for " + split)

# STEP 4: Create YOLO configuration
data_yaml = data / "data.yaml"

data_yaml.write_text(
    f"path: {data}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names: [person]\n"
)

# STEP 5: Retrain from our previous checkpoint
print("Retraining VisDrone...", flush=True)

model = YOLO(str(old_model))

model.train(
    data=str(data_yaml),
    epochs=5,              # Five additional epochs
    imgsz=960,             # Higher resolution
    batch=8,               # Images per batch
    device=0,              # Kaggle GPU
    workers=2,             # Faster data loading
    amp=True,              # Mixed precision
    plots=False,
    cls_remap=False,       # Preserve existing one-class head weights
    project=str(WORK),
    name="visdrone_retrained",
    exist_ok=True,
    seed=42
)

# STEP 6: Save updated model
new_model = WORK / "visdrone_retrained/weights/best.pt"

shutil.copy2(
    new_model,
    WORK / "visdrone_retrained_best.pt"
)

# STEP 7: Compare both versions on same validation images
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

print("VISDRONE RETRAINING COMPLETE!", flush=True)
