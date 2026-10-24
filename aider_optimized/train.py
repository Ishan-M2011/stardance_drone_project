
from pathlib import Path
from zipfile import ZipFile
import subprocess
import random
import shutil

print("AIDER started!", flush=True)

# STEP 1: Install AI libraries
subprocess.run(
    ["python", "-m", "pip", "install", "-q", "ultralytics"],
    check=True
)

from ultralytics import YOLO
from PIL import Image

# STEP 2: Find AIDER
folder = Path("/kaggle/input")
classes = ["fire", "flooded_areas", "collapsed_building",
           "traffic_incident", "normal"]

# Look for already-extracted class folders
aider = next(
    (p.parent for p in folder.rglob("normal")
     if p.is_dir() and all((p.parent / c).is_dir() for c in classes)),
    None
)

# Extract ZIP if needed
if aider is None:
    archive = next(
        (p for p in folder.rglob("*.zip")
         if "aider" in p.name.lower()), None
    )

    if archive is None:
        raise FileNotFoundError("AIDER not found")

    with ZipFile(archive) as z:
        z.extractall("/kaggle/temp/aider_source")

    aider = next(
        (p.parent for p in Path("/kaggle/temp/aider_source").rglob("normal")
         if p.is_dir() and all((p.parent / c).is_dir() for c in classes)),
        None
    )

if aider is None:
    raise FileNotFoundError("AIDER class folders missing")

print("AIDER found!", flush=True)

# STEP 3: Prepare images locally
data = Path("/kaggle/temp/aider_ready")

for name in classes:
    images = sorted(
        p for p in (aider / name).rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    random.Random(42).shuffle(images)

    if len(images) < 2:
        raise ValueError(f"Not enough images for {name}")

    val_count = max(1, int(len(images) * 0.2))

    for i, image in enumerate(images):
        split = "val" if i < val_count else "train"

        output = data / split / name / f"{i}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)

        # Resize once and save locally for faster training
        with Image.open(image) as img:
            img.thumbnail((640, 640))
            img.convert("RGB").save(output, quality=85)

    print("Prepared:", name, len(images), flush=True)

# STEP 4: Train emergency classifier
print("Training AIDER...", flush=True)

model = YOLO("yolo11n-cls.pt")

model.train(
    data=str(data),
    epochs=3,         # Initial full-dataset experiment
    imgsz=224,        # Classification input size
    batch=16,         # Same as optimized AFO
    workers=2,        # Same as optimized AFO
    amp=True,         # GPU mixed precision
    cache=False,      # Images already stored locally
    plots=False,
    device=0,
    project="/kaggle/working",
    name="aider_training",
    seed=42
)

# STEP 5: Save only our trained model
shutil.copy2(
    "/kaggle/working/aider_training/weights/best.pt",
    "/kaggle/working/emergency_best.pt"
)

print("AIDER TRAINING COMPLETE!", flush=True)
