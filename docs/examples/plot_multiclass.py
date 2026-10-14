"""
Multiclass: per-class thresholds and cost-sensitive decisions
=============================================================

Predicting the most probable class ignores rare classes and ignores that
some mistakes cost more than others. Two remedies: per-class thresholds
tuned for macro F1 on held-out data, and the Bayes decision rule for a cost
matrix, which picks the class with the lowest expected cost.
"""

# %%
# An imbalanced three-class problem
# ---------------------------------
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from optimal_cutoffs import optimize_decisions, optimize_thresholds

X, y = make_classification(
    n_samples=9000,
    n_classes=3,
    n_informative=4,
    weights=[0.7, 0.2, 0.1],
    class_sep=0.8,
    random_state=0,
)
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=0
)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=0
)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
P_val = model.predict_proba(X_val)
P_test = model.predict_proba(X_test)
argmax = P_test.argmax(axis=1)

# %%
# Per-class thresholds for macro F1
# ---------------------------------
tuned = optimize_thresholds(y_val, P_val, metric="f1", average="macro")
print("thresholds tuned on validation:", np.round(tuned.thresholds, 3))
print(
    f"test macro F1: argmax {f1_score(y_test, argmax, average='macro'):.3f}, ", end=""
)
print(f"tuned {f1_score(y_test, tuned.predict(P_test), average='macro'):.3f}")

# %%
# Decisions from a cost matrix
# ----------------------------
# Rows are the true class, columns the decision. Missing class 2 is costly.
costs = np.array([[0, 1, 1], [2, 0, 1], [20, 10, 0]])
decide = optimize_decisions(P_val, costs)
for label, pred in [("argmax", argmax), ("Bayes rule", decide.predict(P_test))]:
    print(f"{label:<11} mean cost {costs[y_test, pred].mean():.3f}, ", end="")
    print(f"class 2 recall {(pred[y_test == 2] == 2).mean():.2f}")
