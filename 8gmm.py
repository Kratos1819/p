import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import matplotlib.pyplot as plt

digits = load_digits()
X_pca_30 = PCA(n_components=30).fit_transform(digits.data)
gmm = GaussianMixture(n_components=10, random_state=42).fit(X_pca_30)
pred = gmm.predict(X_pca_30)

def map_labels(y_true, y_pred):
    mapped = np.zeros_like(y_pred)
    for i in range(10):
        mask = (y_pred == i)
        if np.any(mask): mapped[mask] = mode(y_true[mask], keepdims=False).mode
    return mapped

final = map_labels(digits.target, pred)
print("Accuracy:", accuracy_score(digits.target, final))

X_pca_2 = PCA(n_components=2).fit_transform(digits.data)
plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=final, cmap='tab10', s=10, alpha=0.7)
plt.title("GMM Cluster Assignments (after mapping)")
plt.colorbar(label="Predicted Label")
plt.tight_layout()
plt.show()