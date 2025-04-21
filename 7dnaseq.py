import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import mode
import matplotlib.pyplot as plt

seqs = ["ATGCG", "CGTGA", "GCGTT", "ATGCC", "TTGCA"]
labels = ["EEIII", "EIIII", "EEEII", "EEIII", "IIEEE"]
obs_map = {"A": 0, "T": 1, "G": 2, "C": 3}
state_map = {"E": 0, "I": 1}

X = np.concatenate([[obs_map[c] for c in seq] for seq in seqs]).reshape(-1,1)
y = np.concatenate([[state_map[c] for c in lab] for lab in labels])
lengths = [len(seq) for seq in seqs]

model = hmm.MultinomialHMM(n_components=2, n_iter=100).fit(X, lengths)
_, hidden = model.decode(X)
mapped = {s: mode(y[hidden==s], keepdims=True).mode[0] for s in np.unique(hidden)}
pred = np.array([mapped[s] for s in hidden])

print("Accuracy:", accuracy_score(y, pred))
print(classification_report(y, pred, target_names=["Exon", "Intron"]))

plt.plot(y, label="True", marker='o')
plt.plot(pred, label="Pred", linestyle='--', marker='x')
plt.legend(); plt.title("DNA State Prediction"); plt.xlabel("Position"); plt.ylabel("State")
plt.show()