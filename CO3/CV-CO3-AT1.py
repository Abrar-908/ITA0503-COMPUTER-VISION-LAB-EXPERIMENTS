import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("input.jpg")

if img is None:
    print("Error: input.jpg not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 1)

# Edge detection
edges = cv2.Canny(blur, 50, 150)

# Make copy for drawing
result = img.copy()

# -----------------------------
# Hough Line Transform
# -----------------------------
lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    threshold=80,
    minLineLength=50,
    maxLineGap=10
)

line_count = 0

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        line_count += 1

# -----------------------------
# Hough Circle Transform
# -----------------------------
circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=100
)

circle_count = 0

if circles is not None:
    circles = np.uint16(np.around(circles))

    for circle in circles[0]:
        x, y, r = circle

        # Draw circle
        cv2.circle(
            result,
            (x, y),
            r,
            (255, 0, 0),
            2
        )

        # Draw center
        cv2.circle(
            result,
            (x, y),
            3,
            (0, 0, 255),
            -1
        )

        circle_count += 1

# Print results
print("Number of detected lines:", line_count)
print("Number of detected circles:", circle_count)

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edges")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Hough Shape Detection")
plt.axis("off")

plt.tight_layout()
plt.show()