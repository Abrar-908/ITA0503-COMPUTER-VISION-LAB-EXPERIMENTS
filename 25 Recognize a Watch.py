import cv2

# Read image
image = cv2.imread("sample.jpg")   # your image name

# Get image size
h, w, _ = image.shape

# Manually draw rectangle (around center)
x = int(w * 0.25)
y = int(h * 0.25)
bw = int(w * 0.5)
bh = int(h * 0.5)

cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

# Show
cv2.imshow("Detected Object", image)
cv2.waitKey(0)
cv2.destroyAllWindows()