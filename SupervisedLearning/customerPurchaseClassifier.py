import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('purchase_data.csv')

X = df.drop('purchase', axis=1)

y = df['purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

weak_classifier = DecisionTreeClassifier(max_depth=1)

adaBoost_classifier = AdaBoostClassifier(estimator=weak_classifier, n_estimators=50, random_state=42)

adaBoost_classifier.fit(X_train, y_train)

y_pred = adaBoost_classifier.predict(X_test)

print(f"Accuracy score : {accuracy_score(y_test, y_pred)*100:.2f}%")

print("Classification report : \n")
print(classification_report(y_test, y_pred))

new_customer_pred = adaBoost_classifier.predict(pd.DataFrame([[23, 30000, 2]], columns=['age','income','customer_loyalty_score']))

print("New Customer (age=23, income=30000, customer_loyalty_score=2) : ", "Purchase" if new_customer_pred[0] == 1 else "Not Purchase")