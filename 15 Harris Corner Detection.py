import cv2
import numpy as np

image = cv2.imread("sample.jpg")

if image is None:
    print("Error")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
harris_corners = cv2.dilate(harris_corners, None)

image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

cv2.imshow("Original Image", image)
cv2.imshow("Harris Corner Detection", image)

cv2.imwrite("harris_corners.jpg", image)

cv2.waitKey(0)
cv2.destroyAllWindows()