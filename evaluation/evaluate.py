from pathlib import Path
import subprocess
subprocess.run(["pip", "install", "-q", "ultralytics"], check=True) #install yolo on kaggle

from ultralytics import YOLO

# find dataset and trained model
folder = Path("/kaggle/input") 

model_file = next( 
    folder.rglob("sard_best.pt"), None
) #Model file we had trained is trying to be found


dataset = next(
    (p.parent for p in folder.rglob("data.yaml")
     if (p.parent / "test/images").exists()),
    None
)  # Find the SARD 2 dataset folder

if model_file is None or dataset is None: # If files not found, let us know
    raise FileNotFoundError("Could not find the model file or dataset folder.")

images = sorted((dataset / "test/images").glob("*.jpg"))[:100] # Select 100 images

model = YOLO(str(model_file)) # Load the trained model

correct = 0

for i, image in enumerate(images): #test each image
    label = dataset / "test/labels" / (image.stem + ".txt") # Find correct answers from dataset
    actual = bool(label.read_text().strip()) if label.exists () else False
    # True means human is labeled
    # False means human is not labeled
    # Ask our trained AI to detect humans

    # Ask our trained AI to detect humans
    result = model.predict(
        source=str(image),  # Image being tested
        conf=0.10,          # Minimum confidence of 10%
        imgsz=1280,         # Image resolution
        device=0,           # Use Kaggle GPU
        verbose=False       # Hide extra detection logs
    )[0]

    predicted = len(result.boxes) > 0 #check whether YOLO detected at least 1 human

    # Compare prediction to correct answer
    if actual == predicted:
        correct += 1  # Increase score if prediction is correct

    # Save the first 12 images with bounding boxes
    if i < 12:
        result.save(
            filename=f"/kaggle/working/trained_{i}.jpg"
        )

# STEP 5: Calculate final detection rate
accuracy = correct / len(images) * 100  # Convert score to percentage

# STEP 6: Show final results in Kaggle logs
print("FINAL RESULTS")
print("Total Images:", len(images))
print("Correct:", correct)
print("Wrong:", len(images) - correct)
print(f"Detection Accuracy: {accuracy:.2f}%")


# STEP 7: Evaluate the actual bounding boxes

from pathlib import Path

# Create a dataset configuration with correct paths
config = Path("/kaggle/working/eval.yaml")

config.write_text(f"""
path: {dataset}
train: train/images
val: valid/images
test: test/images
names: [Running, Walking, laying_down, not_defined, seated, stands]
""")

# Compare YOLO predictions with actual human boxes
metrics = model.val(
    data=str(config),  # Dataset and correct labels
    split="test",       # Use test images
    single_cls=True,   # Treat all human poses as person
    imgsz=1280,        # Same image size
    device=0,          # Kaggle GPU
    plots=True         # Save evaluation graphs
)

# Print the real detection metrics
print("BOUNDING BOX RESULTS")
print("Precision:", round(metrics.box.mp * 100, 2), "%")
print("Recall:", round(metrics.box.mr * 100, 2), "%")
print("mAP50:", round(metrics.box.map50 * 100, 2), "%")
print("mAP50-95:", round(metrics.box.map * 100, 2), "%")
