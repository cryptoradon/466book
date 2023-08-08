import cv2
import numpy as np

image = cv2.imread('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure5\\f05d57c3-9faa-48c6-b615-0b5570619386\\gradient_magnitude.jpg', cv2.IMREAD_GRAYSCALE)

threshold_value = 200
_, pruned_edges = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure5\\f05d57c3-9faa-48c6-b615-0b5570619386\\pruned_edges.jpg', pruned_edges)

contours, _ = cv2.findContours(pruned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cleaned_image = np.zeros_like(pruned_edges)
for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    extent = area / (w * h)

    if area < 40:
        continue
    if aspect_ratio > 50 or aspect_ratio < 1 / 50:
        continue
    cv2.drawContours(cleaned_image, [contour], -1, 255)

cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure5\\f05d57c3-9faa-48c6-b615-0b5570619386\\removed_shapes.jpg', cleaned_image)

kernel = np.ones((3,3), np.uint8)
closed_edges = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure5\\f05d57c3-9faa-48c6-b615-0b5570619386\\closed_edges.jpg', closed_edges)