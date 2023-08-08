import cv2
import numpy as np

image = cv2.imread('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure5\\IMG_0883\\gradient_magnitude.jpg', cv2.IMREAD_GRAYSCALE)

threshold_value = 160
_, pruned_edges = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure5\\IMG_0883\\pruned_edges.jpg', pruned_edges)

contours, _ = cv2.findContours(pruned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cleaned_image = np.zeros_like(pruned_edges)
for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    extent = area / (w * h)

    if area < 15:
        continue
    if aspect_ratio > 5 or aspect_ratio < 1 / 5:
        continue
    if 0.9 < aspect_ratio < 1.1 and 0.78 < extent < 0.81:
        continue
    cv2.drawContours(cleaned_image, [contour], -1, 255)

cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure5\\IMG_0883\\removed_shapes.jpg', cleaned_image)

kernel = np.ones((2,2), np.uint8)
closed_edges = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure5\\IMG_0883\\closed_edges.jpg', closed_edges)