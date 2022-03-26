import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as img

# Sobel edge operator
kernel_Sobel_x = np.array([
    [-0.5, 0, 0.5],
    [-1, 0, 1],
    [-0.5, 0, 0.5]
])
kernel_Sobel_y = np.array([
    [0.5, 1, 0.5],
    [0, 0, 0],
    [-0.5, -1, -0.5]
])

# Prewitt edge operator
kernel_Prewitt_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])
kernel_Prewitt_y = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])

# Plot few images in a row for comparison
def plot_images(images, labels) :
    if(len(images) != len(labels)):
        raise RuntimeError(f'Cannot assign {len(labels)} labels to {len(images)} images!')

    fig, axes = plt.subplots(1, len(images))
    for idx, ax in enumerate(axes) :
        ax.imshow(images[idx], cmap=plt.get_cmap('gray'))
        ax.set_title(labels[idx])
        ax.axis('off')
    plt.show()

# Convert RGB image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Warning appliable only for 3x3 filters
def apply_filter(image: np.ndarray, kernel_x: np.ndarray, kernel_y: np.ndarray) :
    if(len(image.shape) != 2 or len(kernel_x.shape) != 2 or len(kernel_y.shape) != 2):
        raise RuntimeError('Image and kernels should have 2 dimentions')
    result = image.copy()

    # Add padding for image to avoid index out of bounds
    image = np.vstack((image[-1], image, image[0]))
    image = np.hstack((image[:,-1][:, np.newaxis], image, image[:,0][:, np.newaxis]))

    # Iterate through image
    for i in range(1, image.shape[0] - 2) :
        for j in range(1, image.shape[1] - 2) :
            result[i, j] = np.sqrt(
                np.sum(kernel_x * image[i-1:i+2, j-1:j+2])**2 +
                np.sum(kernel_y * image[i-1:i+2, j-1:j+2])**2
            )
    return result

# Read images
images = [
    rgb2gray(img.imread('hand_compressed.jpg')),
    rgb2gray(img.imread('shepherd_compressed.jpg')),
    rgb2gray(img.imread('high_contrast.jpg')),
    rgb2gray(img.imread('low_contrast.jpg'))
]

labels = [
    'Original image (grayscale)',
    'After Sobel filtration',
    'After Prewitt filtration'
]

for img in images:
    imgs = [
        img,
        apply_filter(img, kernel_Sobel_x, kernel_Sobel_y),
        apply_filter(img, kernel_Prewitt_x, kernel_Prewitt_y),
    ]
    plot_images(imgs, labels)