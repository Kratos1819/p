import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Load and preprocess MNIST
mnist = datasets.fetch_openml('mnist_784', version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualize sample predictions
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f'{y_pred[i]}|{y_test.iloc[i]}')
    plt.axis('off')
plt.show()

# PCA Visualization
X_pca = PCA(n_components=2).fit_transform(X_test)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='tab10', s=10, alpha=0.6)
plt.title("PCA of Predictions")
plt.show()