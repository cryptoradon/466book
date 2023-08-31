from scipy.ndimage import gaussian_laplace
import matplotlib.pyplot as plt
import cv2
import numpy as np

def zero_crossings(laplacian):
    output = np.zeros_like(laplacian, dtype=np.uint8)
    rows, cols = laplacian.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            window = laplacian[i-1:i+2, j-1:j+2]
            max_val = window.max()
            min_val = window.min()
            if max_val > 0 and min_val < 0:
                output[i, j] = 1
    return output

def enhance_laplacian(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    enhanced_image = cv2.dilate(image, kernel, iterations=1)
    return enhanced_image

image = cv2.imread('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure17\\IMG_0883\\IMG_0883.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 2)

laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)

zero_crossings_image = zero_crossings(laplacian)

gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
magnitude_x = np.abs(gradient_x)
magnitude_y = np.abs(gradient_y)
gradient_diag1 = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_diag2 = np.sqrt(gradient_x**2 + gradient_y**2)
magnitude_diag1 = np.abs(gradient_diag1)
magnitude_diag2 = np.abs(gradient_diag2)
all_magnitudes = np.stack((magnitude_x, magnitude_y, magnitude_diag1, magnitude_diag2), axis=-1)
max_magnitude = np.max(all_magnitudes, axis=2)

norm_image = image / 255.0

cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure17\\IMG_0883\\Gray_Image.jpg', gray_image)
gradient_threshold = 0
edge_map = np.zeros_like(gray_image)
for y in range(gray_image.shape[0]):
    for x in range(gray_image.shape[1]):
        if zero_crossings_image[y, x] and max_magnitude[y, x] > gradient_threshold:
            edge_map[y, x] = 255
cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure17\\IMG_0883\\Edge_Map_0.jpg', edge_map)

gradient_threshold = np.max(max_magnitude) * 0.25
edge_map = np.zeros_like(gray_image)
for y in range(gray_image.shape[0]):
    for x in range(gray_image.shape[1]):
        if zero_crossings_image[y, x] and max_magnitude[y, x] > gradient_threshold:
            edge_map[y, x] = 255
cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure17\\IMG_0883\\Edge_Map_0.25.jpg', edge_map)

min_val = np.min(laplacian)
max_val = np.max(laplacian)
laplacian_normalized = 255 * (laplacian - min_val) / (max_val - min_val)
laplacian_normalized = laplacian_normalized.astype(np.uint8)
enhanced_laplacian = enhance_laplacian(laplacian_normalized)
cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure17\\IMG_0883\\Laplacian_of_Gaussian.jpg', enhanced_laplacian)