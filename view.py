#made to see the images after YOLO processing
from pathlib import Path
from PIL import Image #open images
import matplotlib.pyplot as plt #displays images

images = list(Path("results").rglob("*.jpg"))[:25] 

for i, image in enumerate(images):
    plt.subplot(5, 5, i+1) # 2x3 grid
    plt.imshow(Image.open(image)) #display image
    plt.axis("off") # Hides axises

plt.tight_layout() #adjust spacing
plt.show() #Display final images!!
# Helps with viewing the images, run it usually after training, view bounding boxes and all that, opens in seperate white tab