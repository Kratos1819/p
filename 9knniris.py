import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.draw import disk, rectangle, polygon
from skimage.util import random_noise

# Generate a simple noisy shape image
def make_image(shape, size=64):
    img = np.zeros((size, size))
    if shape == "circle":
        rr, cc = disk((32, 32), 16)
    elif shape == "square":
        rr, cc = rectangle((16, 16), end=(48, 48))
    elif shape == "triangle":
        pts = np.array([[32, 16], [16, 48], [48, 48]])
        rr, cc = polygon(pts[:, 0], pts[:, 1])
    img[rr, cc] = 1
    return random_noise(img, var=0.01)

# Create dataset
shapes = ["circle", "square", "triangle"]
X, y = [], []
for i, shape in enumerate(shapes):
    for _ in range(50):  # fewer samples for simplicity
        X.append(make_image(shape).flatten())
        y.append(i)

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Show predictions
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
    plt.title(f"Pred: {shapes[pred[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
