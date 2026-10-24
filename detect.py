print("Python started", flush=True)
from pathlib import Path
import subprocess

import kagglehub
subprocess.run(["pip", "install", "-q", "ultralytics"], check=True) # Install Yolo library on kaggle

from ultralytics import YOLO 

folder = Path("/kaggle/input")#Path to kaggle dataset
images = sorted(
    p for p in folder.rglob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)[:100] # find the 100 images in all formats
print("Dataset files:", [p.name for p in list(folder.rglob("*"))[:10]])
print("Images found:", len(images)) #Check if images found
if not images:
    raise FileNotFoundError("No images found in the specified folder.")
model = YOLO("yolo11n.pt")
correct = 0
wrong = 0 
missed = 0
false_alarm = 0
actual_positive = 0

for i, image in enumerate(images):
    print(f"Processing image {i+1}", flush=True)
    results = model.predict(
        source=str(image), #Image to analyze
        classes=[0], # What class to detect (0 = person)
        conf=0.10,
        imgsz=1280, #image size to be used in YOLO
        device=0,

    )
    results[0].save(
        filename=f"/kaggle/working/image_{i}.jpg"
    )
    # Finding dataset correct answer
    label = image.parent.parent / "labels" / (image.stem + ".txt")

    actual = label.exists() and bool(label.read_text().strip())# True if labeled exists
    predicted = len(results[0].boxes) > 0 # TRUE if YOLO detected the person

    actual_positive += int(actual)

    if actual == predicted:
        correct += 1
    else:
        wrong += 1
        if actual:
            missed += 1

        else:
            false_alarm += 1
accuracy = correct / len(images) * 100 #calc final accuracy
print("\nFINAL RESULTS")
print("Total Images:", len(images))
print("Images With Humans:", actual_positive)
print("Correct:", correct)
print("Wrong:", wrong)
print("Missed Human Images:", missed)
print("False Alarm Images:", false_alarm)
print(f"Accuracy: {accuracy:.2f}%")