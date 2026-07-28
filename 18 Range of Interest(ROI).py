import cv2

# Read image
image = cv2.imread("sample.jpg")

# Select region (y1:y2, x1:x2)
roi = image[100:300, 150:350]

# Show
cv2.imshow("Original Image", image)
cv2.imshow("Cropped ROI", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()