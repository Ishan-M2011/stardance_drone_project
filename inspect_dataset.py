# Inspecting dataset to find structure bc we want all annotation and everything
print("Inspection started", flush=True)
from pathlib import Path
from collections import Counter
import yaml

folder = Path("/kaggle/input")
yaml_file = next(
    p for p in folder.rglob("*.yaml")
    if (p.parent / "train/images").exists()
)
dataset = yaml_file.parent
print("Dataset location:", dataset)
print("Dataset configuration:")
print(yaml_file.read_text())
# Count images and labels 
for split in ["train", "valid", "val", "test"]: #loop through dataset folders
    images = list((dataset / split / "images").glob("*.jpg"))
    labels = list((dataset / split / "labels").glob("*.txt"))

    if images:
        print(split, "Images:", len(images), "Labels:", len(labels))

#count classes
counts = Counter()
for label in (dataset / "train/labels").glob("*.txt"): #open every label file
    for line in label.read_text().splitlines(): #read each file
        if line.strip():
            counts[int(line.split()[0])] += 1

print("Training class counts:", dict(counts))            