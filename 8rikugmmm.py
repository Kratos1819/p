import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Load the handwritten digits dataset (8x8 pixel images of digits 0-9)
digits = load_digits()
X = digits.data  # Features (8x8 pixel values)
y = digits.target  # Labels (digit classes)

# Standardize the features (important for GMM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit a Gaussian Mixture Model with 10 components (since there are 10 digits)
gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

# Predict cluster assignments (which digit each sample is most likely to belong to)
predictions = gmm.predict(X_scaled)

# Visualizing the results
plt.figure(figsize=(10, 6))

# Scatter plot of the first two features (for visualization)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=predictions, cmap='viridis', marker='o')
plt.title('Clustering of Handwritten Digits using GMM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster ID')
plt.show()