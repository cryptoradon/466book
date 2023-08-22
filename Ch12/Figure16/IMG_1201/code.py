from scipy.ndimage import gaussian_laplace
import matplotlib.pyplot as plt
import cv2
import numpy as np

def zero_crossings(image):
    threshold = 5
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(1, padded_image.shape[0] - 1):
        for j in range(1, padded_image.shape[1] - 1):
            window = padded_image[i-1:i+2, j-1:j+2]

            positive_values = window[window > 0]
            negative_values = window[window < 0]

            if positive_values.size > 0 and negative_values.size > 0:
                max_positive = np.max(positive_values)
                min_negative = np.min(negative_values)
                
                if max_positive - min_negative > threshold:
                    output[i-1, j-1] = 255

    return output

image = cv2.imread('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure16\\IMG_1201\\IMG_1201.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 2)

laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)

min_val = np.min(laplacian)
max_val = np.max(laplacian)
laplacian_normalized = 255 * (laplacian - min_val) / (max_val - min_val)
laplacian_normalized = laplacian_normalized.astype(np.uint8)

zero_crossings_image = zero_crossings(laplacian)

cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure16\\IMG_1201\\LoG_11.jpg', laplacian_normalized)
cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure16\\IMG_1201\\zerocross_11.jpg', zero_crossings_image)

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)

laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)

min_val = np.min(laplacian)
max_val = np.max(laplacian)
laplacian_normalized = 255 * (laplacian - min_val) / (max_val - min_val)
laplacian_normalized = laplacian_normalized.astype(np.uint8)

zero_crossings_image = zero_crossings(laplacian)

cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure16\\IMG_1201\\LoG_5.jpg', laplacian_normalized)
cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure16\\IMG_1201\\zerocross_5.jpg', zero_crossings_image)