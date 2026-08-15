import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------
# Read Image
# -----------------------------

img = cv2.imread("input.jpg")

if img is None:
    print("Error: input.jpg not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

# Resize image
gray = cv2.resize(
    gray,
    (200, 200)
)

# -----------------------------
# Create Image Patches
# -----------------------------

patch_size = 10

features = []

for i in range(
    0,
    gray.shape[0] - patch_size + 1,
    patch_size
):
    for j in range(
        0,
        gray.shape[1] - patch_size + 1,
        patch_size
    ):

        patch = gray[
            i:i + patch_size,
            j:j + patch_size
        ]

        # Flatten patch
        features.append(
            patch.flatten()
        )

X = np.array(features)

print(
    "Original Feature Dimensions:",
    X.shape
)

# -----------------------------
# Standardize Features
# -----------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -----------------------------
# PCA
# -----------------------------

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

print(
    "Reduced Feature Dimensions:",
    X_pca.shape
)

# Explained variance
variance = (
    np.sum(
        pca.explained_variance_ratio_
    ) * 100
)

print(
    "Information Preserved: {:.2f}%"
    .format(variance)
)

# -----------------------------
# Visualization Before PCA
# -----------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    alpha=0.6
)

plt.xlabel("Original Feature 1")
plt.ylabel("Original Feature 2")

plt.title(
    "Feature Visualization Before PCA"
)

plt.grid(True)

plt.show()

# -----------------------------
# Visualization After PCA
# -----------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    alpha=0.6
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "Feature Visualization After PCA"
)

plt.grid(True)

plt.show()

# -----------------------------
# Explained Variance Plot
# -----------------------------

pca_full = PCA()

pca_full.fit(X_scaled)

cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)

plt.figure(figsize=(8, 5))

plt.plot(
    range(
        1,
        len(cumulative_variance) + 1
    ),
    cumulative_variance * 100,
    marker="o"
)

plt.xlabel(
    "Number of Principal Components"
)

plt.ylabel(
    "Cumulative Variance Preserved (%)"
)

plt.title(
    "PCA Explained Variance"
)

plt.grid(True)

plt.show()