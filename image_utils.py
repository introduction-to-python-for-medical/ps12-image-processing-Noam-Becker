from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from skimage.filters import median
from skimage.morphology import ball

def image_load(path):
    """
    פונקציה לטעינת תמונה ממסלול והמרת הצבעים לגריסקייל.
    מחזירה את התמונה כ-matrix (מערך) של NumPy.
    """
    image = Image.open(path).convert('L')  # 'L' עבור גריסקייל
    return np.array(image)

def detection_edge(image):
    """
    פונקציה לביצוע זיהוי קצוות בתמונה.
    מקבלת תמונה (מערך NumPy) ומחזירה את המגבלה של הקצוות.
    """
    # יצירת פילטרים עבור זיהוי קצוות
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # פילטר אופקי
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # פילטר אנכי
    
    # ביצוע קונבולוציה על התמונה
    edge_x = convolve2d(image, sobel_x, mode='same', boundary='symm')
    edge_y = convolve2d(image, sobel_y, mode='same', boundary='symm')
    
    # חישוב המגבלה (שני הכיוונים)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    
    return edge_mag

def denoise_image(image):
    """
    פונקציה להסרת רעשים בתמונה באמצעות פילטר חציוני.
    מחזירה תמונה ללא רעשים.
    """
    return median(image, ball(3))  # הערך 3 הוא רדיוס הסוג של הפילטר

def binarize_image(edge_mag, threshold=100):
    """
    פונקציה להמיר את התמונה לבינארית על בסיס סף שנבחר.
    מחזירה תמונה בינארית (True/False).
    """
    return edge_mag > threshold

def save_and_show_image(image, filename):
    """
    פונקציה לשמירת התמונה כקובץ PNG והצגתה.
    """
    from skimage.io import imsave
    import matplotlib.pyplot as plt

    # שמירת התמונה כקובץ PNG
    imsave(filename, image.astype(np.uint8) * 255)  # המרת True/False ל-0 ו-255
    # הצגת התמונה
    plt.imshow(image, cmap='gray')
    plt.show()
