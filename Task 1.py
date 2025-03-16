import cv2
import numpy as np

# Read the image in grayscale
image_path = 'download.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian smoothing
smoothed_image = cv2.GaussianBlur(image, (3, 3), 0)

# Sobel operator kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Perform convolution to get gradient in x and y directions for smoothed image
gradient_x_smoothed = cv2.filter2D(smoothed_image, cv2.CV_64F, sobel_x)
gradient_y_smoothed = cv2.filter2D(smoothed_image, cv2.CV_64F, sobel_y)

# Calculate the magnitude of the gradients for smoothed image
gradient_magnitude_smoothed = np.sqrt(gradient_x_smoothed ** 2 + gradient_y_smoothed ** 2)

# Normalize gradient magnitude to [0, 255]
gradient_magnitude_smoothed = cv2.normalize(gradient_magnitude_smoothed, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Perform convolution to get gradient in x and y directions for raw image
gradient_x_raw = cv2.filter2D(image, cv2.CV_64F, sobel_x)
gradient_y_raw = cv2.filter2D(image, cv2.CV_64F, sobel_y)

# Calculate the magnitude of the gradients for raw image
gradient_magnitude_raw = np.sqrt(gradient_x_raw ** 2 + gradient_y_raw ** 2)

# Normalize gradient magnitude to [0, 255]
gradient_magnitude_raw = cv2.normalize(gradient_magnitude_raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display the results
cv2.imshow('Smoothed Image', smoothed_image)
cv2.imshow('Edge Detection with Smoothing', gradient_magnitude_smoothed)
cv2.imshow('Edge Detection without Smoothing', gradient_magnitude_raw)
cv2.waitKey(0)
cv2.destroyAllWindows()
