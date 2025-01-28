from PIL import Image
import numpy as np

def load_image(path):
    image = Image.open(path).convert("L")  # המרה לגריסקייל
    return np.array(image)
from scipy.signal import convolve2d

def edge_detection(image):
    # הגדרת מסנן Sobel אופקי ואנכי
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # מסנן אופקי
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # מסנן אנכי

    # ביצוע קונבולוציה עם המסננים
    grad_x = convolve2d(image, sobel_x, mode='same', boundary='symm')
    grad_y = convolve2d(image, sobel_y, mode='same', boundary='symm')

    # חישוב התוצאה הסופית (שורש סכום ריבועים)
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # חזרה על התמונה עם ערכים בין 0 ל-255
    return np.uint8(np.clip(edge_magnitude, 0, 255))
