import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure5\\IMG_0255\\IMG_0255.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

smoothed_image = cv2.GaussianBlur(gray_image, (45, 45), 0)

sobel_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)

gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

cv2.imwrite('C:\\Users\\indus\\OneDrive\\Desktop\\ImageBook\\MyImplementations\\Ch12\\Figure5\\IMG_0255\\gradient_magnitude.jpg', gradient_magnitude)