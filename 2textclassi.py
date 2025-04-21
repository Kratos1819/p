from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
train = fetch_20newsgroups(subset='train', shuffle=True)
test = fetch_20newsgroups(subset='test', shuffle=True)

# Build pipeline
model = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# Train and predict
model.fit(train.data, train.target)
pred = model.predict(test.data)

# Evaluate
print("Accuracy:", metrics.accuracy_score(test.target, pred))
print(metrics.classification_report(test.target, pred, target_names=test.target_names))

# Confusion matrix
sns.heatmap(metrics.confusion_matrix(test.target, pred),
             fmt='d',
            xticklabels=test.target_names, 
            yticklabels=test.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Save model
joblib.dump(model, "text_classifier.pkl")