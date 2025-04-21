from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X = X[:, 2:4]  # petal length & width
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

best_k, best_acc = 1, 0
for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    if acc > best_acc: best_k, best_acc = k, acc
    print(f"K={k}, Accuracy={acc:.2f}")

final_model = KNeighborsClassifier(n_neighbors=best_k).fit(Xtr, ytr)
final_preds = final_model.predict(Xte)
print(f"Best K: {best_k}, Accuracy: {best_acc:.2f}")