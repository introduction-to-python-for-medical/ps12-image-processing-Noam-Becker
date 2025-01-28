from image_utils import image_load, detection_edge, denoise_image, binarize_image, save_and_show_image

# שלב 1: טעינת התמונה
image = image_load("path_to_your_image.jpg")

# שלב 2: הסרת רעשים
clean_image = denoise_image(image)

# שלב 3: זיהוי קצוות
edge_mag = detection_edge(clean_image)

# שלב 4: הפיכת התמונה לבינארית
binary_image = binarize_image(edge_mag, threshold=100)

# שלב 5: שמירה והצגה
save_and_show_image(binary_image, "edge_detected_image.png")
