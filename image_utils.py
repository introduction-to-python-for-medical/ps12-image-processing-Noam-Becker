from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    """
    Load an image from the given file path and convert it into a NumPy array.
    """
    image = Image.open(path)  # Open image
    image = np.array(image)  # Convert to NumPy array
    return image

def edge_detection(image):
    """
    Perform edge detection on an image array using vertical and horizontal filters.
    """
    # Convert to grayscale by averaging RGB channels
    if len(image.shape) == 3:
        gray_image = np.mean(image, axis=2)
    else:
        gray_image = image  # If already grayscale, keep as is
    
    # Define vertical and horizontal edge detection filters
    vertical_filter = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
    
    horizontal_filter = np.array([[-1, -2, -1],
                                   [0,  0,  0],
                                   [1,  2,  1]])
    
    # Apply convolution
    edge_x = convolve2d(gray_image, vertical_filter, mode='same', boundary='fill', fillvalue=0)
    edge_y = convolve2d(gray_image, horizontal_filter, mode='same', boundary='fill', fillvalue=0)
    
    # Compute magnitude of edges
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    
    # Normalize to range [0, 255]
    edge_mag = (edge_mag / edge_mag.max()) * 255
    
    return edge_mag.astype(np.uint8)
