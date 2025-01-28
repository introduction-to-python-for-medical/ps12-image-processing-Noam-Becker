
import matplotlib.pyplot as plt
from image_utils import load_image, edge_detection

# טעינת התמונה המקורית
image = load_image("path_to_your_image.jpg")

# ביצוע זיהוי קצוות
edges = edge_detection(image)

# הצגת התמונה המקורית והתמונה עם הקצוות
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Edge Detected Image")
plt.imshow(edges, cmap='gray')

plt.show()
