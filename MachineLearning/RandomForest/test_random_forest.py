from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from random_forest import RandomForest

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

clf = RandomForest(n_trees=20, n_features=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc =  accuracy(y_test, predictions)
#print(f"Features: {clf.n_features}")
#print(f"Accuracy: {acc}")

#Effect of the number of features on the accuracy
#for nf in [1, 2, 5, 10, 20, 30]:
#    accs = []
#    for _ in range(3):
#        clf = RandomForest(n_trees=20, n_features=nf)
#        clf.fit(X_train, y_train)
#       accs.append(accuracy(y_test, clf.predict(X_test)))
#    print(f"Features: {nf}: Mean: {np.mean(accs):.4f} Range: {min(accs):.4f}-{max(accs):.4f}")

#Effect of max depth on the accuracy
#for md in [1, 2, 3, 5, 10, 20, 100]:
#    train_accs, test_accs = [], []
#    for _ in range(3):
#        clf = RandomForest(n_trees=20, n_features=5, max_depth=md)
#        clf.fit(X_train, y_train)
#        train_accs.append(accuracy(y_train, clf.predict(X_train)))
#        test_accs.append(accuracy(y_test, clf.predict(X_test)))
#    print(f"Max Depth: {md}: Train: {np.mean(train_accs):.4f} Test: {np.mean(test_accs):.4f} Range: {min(test_accs):.4f}-{max(test_accs):.4f}")

#Effect of the number of trees on the accuracy
for nt in [1, 5, 10, 20, 50, 100]:
    accs = []
    for _ in range(3):
        clf = RandomForest(n_trees=nt, n_features=5)
        clf.fit(X_train, y_train)
        accs.append(accuracy(y_test, clf.predict(X_test)))
    print(f"Trees: {nt}: Mean: {np.mean(accs):.4f} Range: {min(accs):.4f}-{max(accs):.4f}")