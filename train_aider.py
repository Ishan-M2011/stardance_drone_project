
from pathlib import Path
from zipfile import ZipFile
import random
import shutil
import subprocess

print("Starting AIDER training!", flush=True)

# Install YOLO
subprocess.run(["python", "-m", "pip", "install", "-q", "ultralytics"], check=True)

from ultralytics import YOLO

# STEP 1: Find AIDER dataset
folder = Path("/kaggle/input")

classes = [
    "fire",
    "flooded_areas",
    "collapsed_building",
    "traffic_incident",
    "normal"
]

# Find class folders if already extracted
aider = next(
    (p.parent for p in folder.rglob("normal")
     if p.is_dir()
     and all((p.parent / c).is_dir() for c in classes)),
    None
)

# Extract AIDER.zip if necessary
if aider is None:
    zip_file = next(
        (p for p in folder.rglob("*.zip")
         if "aider" in p.name.lower()),
        None
    )

    if zip_file is None:
        raise FileNotFoundError("AIDER dataset not found")

    print("Extracting AIDER...", flush=True)

    with ZipFile(zip_file) as z:
        z.extractall("/kaggle/temp/aider_raw")

    aider = next(
        (p.parent for p in Path("/kaggle/temp/aider_raw").rglob("normal")
         if p.is_dir()
         and all((p.parent / c).is_dir() for c in classes)),
        None
    )

if aider is None:
    raise FileNotFoundError("AIDER class folders not found")

# STEP 2: Prepare training and validation images
data = Path("/kaggle/temp/aider_prepared")

for name in classes:
    images = sorted(
        p for p in (aider / name).rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if len(images) < 2:
        raise ValueError(f"Not enough images: {name}")

    random.Random(42).shuffle(images)
    val_count = max(1, int(len(images) * 0.2))

    for i, image in enumerate(images):
        split = "val" if i < val_count else "train"

        new_image = data / split / name / image.name
        new_image.parent.mkdir(parents=True, exist_ok=True)
        new_image.symlink_to(image.resolve())

    print(name, "images:", len(images), flush=True)

# STEP 3: Train emergency classifier
print("Training emergency model...", flush=True)

model = YOLO("yolo11n-cls.pt")

model.train(
    data=str(data),        # Prepared images
    epochs=2,              # Short test
    imgsz=224,
    batch=16,
    device=0,
    project="/kaggle/working",
    name="emergency_training",
    seed=42
)

# STEP 4: Save model
shutil.copy2(
    "/kaggle/working/emergency_training/weights/best.pt",
    "/kaggle/working/emergency_best.pt"
)

print("EMERGENCY MODEL SAVED!", flush=True)
