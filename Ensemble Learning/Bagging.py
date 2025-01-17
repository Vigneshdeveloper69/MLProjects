from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy Score : ", accuracy_score(y_test, y_pred))