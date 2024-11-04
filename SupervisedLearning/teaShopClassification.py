import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('tea_shop.csv')

X = df[['age', 'income']]

y = df['prefers_tea']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = svm.SVC(kernel='linear')

clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

print(f"Accuracy score : {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\nClassification report :")
print(classification_report(y_test, y_pred, zero_division=1))

new_pred_scaler = scaler.fit_transform(pd.DataFrame([[37, 52000]], columns=X.columns))

new_pred = clf.predict(new_pred_scaler)

if new_pred[0] == 1:

    print("The new customer (age=37, salary=52000) will pay Tea.")

else:

    print("The new customer (age=37, salary=52000) will pay Coffee.")