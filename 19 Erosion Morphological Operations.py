import cv2
import numpy as np

# Read image in grayscale
image = cv2.imread("sample.jpg", 0)

# Convert to binary
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Kernel
kernel = np.ones((5,5), np.uint8)

# Apply erosion
eroded = cv2.erode(binary, kernel, iterations=1)

# Show
cv2.imshow("Original Binary", binary)
cv2.imshow("Eroded Image", eroded)

cv2.waitKey(0)
cv2.destroyAllWindows()