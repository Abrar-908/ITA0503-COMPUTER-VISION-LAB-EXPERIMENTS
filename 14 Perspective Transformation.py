import cv2 
import numpy as np 
image = cv2.imread(r"C:\Users\lenovo\OneDrive\Documents\Computer Vision\sample.jpg")  # Replace with your image file path 
rows, cols, ch = image.shape 
pts1 = np.float32([[50, 50], [400, 50], [50, 400], [400, 400]]) 
pts2 = np.float32([[10, 100], [300, 50], [100, 300], [350, 350]]) 
matrix = cv2.getPerspectiveTransform(pts1, pts2) 
transformed_image = cv2.warpPerspective(image, matrix, (cols, rows)) 
cv2.imshow("Original Image", image) 
cv2.imshow("Perspective Transformed Image", transformed_image) 
cv2.imwrite(r"C:\Users\lenovo\OneDrive\Documents\Computer Vision\perspective_transformed.jpg", transformed_image) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 