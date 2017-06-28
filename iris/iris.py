import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

print iris.feature_names
print iris.target_names

test_idx = [0, 50, 100]

traning_target = np.delete(iris.target, test_idx)
traning_data = np.delete(iris.data, test_idx, axis=0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(traning_data, traning_target)

print test_target
print clf.predict(test_data)