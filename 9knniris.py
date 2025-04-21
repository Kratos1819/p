import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from skimage.draw import disk, rectangle, polygon
from skimage.util import random_noise

# Generate one image of a specific shape
def generate_shape_image(shape, size=64):
    img = np.zeros((size, size), dtype=np.float32)
    if shape == 'circle':
        rr, cc = disk((size//2, size//2), size//4)
    elif shape == 'square':
        rr, cc = rectangle(start=(size//4, size//4), end=(3*size//4, 3*size//4))
    elif shape == 'triangle':
        coords = np.array([[size//2, size//4], [size//4, 3*size//4], [3*size//4, 3*size//4]])
        rr, cc = polygon(coords[:, 0], coords[:, 1])
    img[rr, cc] = 1.0
    return random_noise(img, mode='gaussian', var=0.01)

# Create dataset
shapes = ['circle', 'square', 'triangle']
images, labels = [], []

for idx, shape in enumerate(shapes):
    for _ in range(100):
        images.append(generate_shape_image(shape).flatten())
        labels.append(idx)

X, y = np.array(images), np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predictions and evaluation
y_pred = knn.predict(X_test)
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=shapes))

# Show predictions with actual labels
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
    actual = shapes[y_test[i]]
    predicted = shapes[y_pred[i]]
    plt.title(f"True: {actual}\nPred: {predicted}")
    plt.axis('off')
plt.tight_layout()
plt.show()
