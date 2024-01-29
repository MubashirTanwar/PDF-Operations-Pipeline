import cv2
import numpy as np

# Read the image
image = cv2.imread(r'C:\NIC AI Training\PDF Pipeline\temp_images\page_2.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to obtain a binary image
_, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)  # Use THRESH_BINARY_INV

# Define a kernel for dilation

Dkernel = np.ones((4,4), np.uint8)
Ekernel = np.ones((2,2), np.uint8)
img = cv2.dilate(binary_image, Dkernel, iterations=1)
img = cv2.erode(img, Ekernel, iterations=1) 

# Display the original and dilated images
cv2.imshow('Original Image', image)
cv2.imshow('Dilated Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
