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

# -----------------------------
# Add Salt-and-Pepper Noise
# -----------------------------
noisy = gray.copy()

# Salt noise
salt = np.random.rand(*gray.shape) < 0.02
noisy[salt] = 255

# Pepper noise
pepper = np.random.rand(*gray.shape) < 0.02
noisy[pepper] = 0

# -----------------------------
# Apply Filters
# -----------------------------

# Mean filter
mean_filter = cv2.blur(noisy, (5, 5))

# Gaussian filter
gaussian_filter = cv2.GaussianBlur(
    noisy,
    (5, 5),
    0
)

# Median filter
median_filter = cv2.medianBlur(
    noisy,
    5
)

# Bilateral filter
bilateral_filter = cv2.bilateralFilter(
    noisy,
    9,
    75,
    75
)

# -----------------------------
# Display Results
# -----------------------------

plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(mean_filter, cmap="gray")
plt.title("Mean Filter")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(gaussian_filter, cmap="gray")
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(median_filter, cmap="gray")
plt.title("Median Filter")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(bilateral_filter, cmap="gray")
plt.title("Bilateral Filter")
plt.axis("off")

plt.tight_layout()
plt.show()