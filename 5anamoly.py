import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv("kddcup.data_10_percent_corrected", header=None)
df[41] = df[41].apply(lambda x: 1 if x.strip() != 'normal.' else 0)  # 1 = anomaly

# Preprocess
cat_cols = [1, 2, 3]
num_cols = list(set(df.columns) - set(cat_cols) - {41})

X_cat = OneHotEncoder(sparse=False).fit_transform(df[cat_cols])
X_num = StandardScaler().fit_transform(df[num_cols])
X = np.hstack((X_num, X_cat))
y = df[41].values

# PCA for visualization
X_pca = PCA(n_components=2).fit_transform(X)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette=["blue", "red"], alpha=0.5)
plt.title("PCA View: Normal vs Anomalous")
plt.show()

# Train Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
y_pred = model.fit_predict(X)
y_pred = np.where(y_pred == 1, 0, 1)  # 1=anomaly, 0=normal

# Evaluate
print(classification_report(y, y_pred))
sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()