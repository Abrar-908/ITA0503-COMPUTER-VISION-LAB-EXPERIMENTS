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
# Add Gaussian Noise
# -----------------------------

noise = np.random.normal(
    0,
    25,
    gray.shape
)

noisy = gray.astype(np.float32) + noise

noisy = np.clip(
    noisy,
    0,
    255
).astype(np.uint8)

# -----------------------------
# ORB Feature Detector
# -----------------------------

orb = cv2.ORB_create(
    nfeatures=1000
)

# Detect features
kp_original, des_original = orb.detectAndCompute(
    gray,
    None
)

kp_noisy, des_noisy = orb.detectAndCompute(
    noisy,
    None
)

# Print number of features
print(
    "Original Image Keypoints:",
    len(kp_original)
)

print(
    "Noisy Image Keypoints:",
    len(kp_noisy)
)

# -----------------------------
# Draw Keypoints
# -----------------------------

original_features = cv2.drawKeypoints(
    gray,
    kp_original,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

noisy_features = cv2.drawKeypoints(
    noisy,
    kp_noisy,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(
    cv2.cvtColor(
        original_features,
        cv2.COLOR_BGR2RGB
    )
)
plt.title(
    "Features in Original Image"
)
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(
    cv2.cvtColor(
        noisy_features,
        cv2.COLOR_BGR2RGB
    )
)
plt.title(
    "Features in Noisy Image"
)
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# Bar Chart
# -----------------------------

labels = [
    "Original",
    "Noisy"
]

counts = [
    len(kp_original),
    len(kp_noisy)
]

plt.figure(figsize=(7, 5))
plt.bar(labels, counts)

plt.ylabel("Number of Keypoints")
plt.title("Effect of Noise on Feature Detection")

plt.show()