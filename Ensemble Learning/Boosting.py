from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

boosting_model = AdaBoostClassifier(n_estimators=100, random_state=42)

boosting_model.fit(X_train, y_train)

y_pred = boosting_model.predict(X_test)

print(f"Accuracy Score : {accuracy_score(y_test, y_pred) * 100:.2f}")