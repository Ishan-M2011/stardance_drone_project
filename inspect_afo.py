
from pathlib import Path

# Location of AFO dataset
afo = Path("/kaggle/input/afo-aerial-dataset-of-floating-objects")

# STEP 1: Check each dataset part
for part in sorted(afo.glob("PART_*")):
    folder = part / part.name

    print("\nPART:", part.name)
    print("Folders:", [p.name for p in folder.iterdir() if p.is_dir()])

    # Read instructions provided by the dataset
    readme = folder / "readme.txt"
    if readme.exists():
        print("README:", readme.read_text()[:1000])

# STEP 2: Find class names
for pattern in ["*.names", "classes.txt"]:
    for file in list(afo.rglob(pattern))[:3]:
        print("\nClass file:", file)
        print(file.read_text()[:500])

# STEP 3: Check training and testing lists
for name in ["train.txt", "valid.txt", "val.txt", "test.txt"]:
    for file in list(afo.rglob(name))[:1]:
        print("\nSplit file:", file)
        print(file.read_text().splitlines()[:3])

# STEP 4: Find annotations for one image
image = next(afo.rglob("*.jpg"))

print("\nExample image:", image)

for label in list(afo.rglob(image.stem + ".txt"))[:6]:
    print("Label:", label)
    print(label.read_text().splitlines()[:2])

print("\nINSPECTION COMPLETE!")
