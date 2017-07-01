from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

from sklearn.neighbors import KNeighborsClassifier
my_classfier = KNeighborsClassifier()

my_classfier.fit(x_train, y_train);

predictions = my_classfier.predict(x_test)

print accuracy_score(y_test, predictions)