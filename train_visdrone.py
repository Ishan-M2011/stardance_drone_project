
import subprocess
import shutil

print("Starting VisDrone training!", flush=True)

# Install YOLO
subprocess.run(["python", "-m", "pip", "install", "-q", "ultralytics"], check=True)

from ultralytics import YOLO

# STEP 1: Load pretrained model
model = YOLO("yolo11n.pt")

# STEP 2: Download and train on VisDrone
print("Preparing VisDrone and starting training...", flush=True)

model.train(
    data="VisDrone.yaml",  # Automatically downloads/converts data
    classes=[0, 1],        # Pedestrian and people
    single_cls=True,       # Combine into human class
    epochs=2,              # Short test
    imgsz=640,
    batch=8,
    device=0,
    project="/kaggle/working",
    name="visdrone_training",
    seed=42
)

# STEP 3: Save the best model
shutil.copy2(
    "/kaggle/working/visdrone_training/weights/best.pt",
    "/kaggle/working/visdrone_best.pt"
)

print("VISDRONE MODEL SAVED!", flush=True)
