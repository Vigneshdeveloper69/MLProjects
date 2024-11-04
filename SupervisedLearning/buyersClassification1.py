import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('buyers_status.csv')

X = df[['Age', 'Income']]

y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = LogisticRegression()

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy :",accuracy_score(y_test, y_pred))
print("Percision score :",precision_score(y_test, y_pred))
print("Recall score :", recall_score(y_test, y_pred))
print("F1 score :", f1_score(y_test, y_pred))
print("ROC_AUC score :", roc_auc_score(y_test,y_pred))
print("Confusion matrix :", confusion_matrix(y_test, y_pred))

new_customer = pd.DataFrame([[40, 40000]], columns=['Age', 'Income'])

new_customer_scale = scalar.transform(new_customer)

new_customer_prediction = model.predict(new_customer_scale)

print("Prediction for new customer (Age = 40, Income = 40000) : \n", 'Purchase' if new_customer_prediction[0] == 1 else 'Not Purchase')