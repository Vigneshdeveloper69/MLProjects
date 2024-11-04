import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('email_spams_data.csv')

X = df['Text']

y = df['Label']

cv = CountVectorizer()

X = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy score : {accuracy_score(y_test, y_pred)*100:.2f}%")

print("Confusion matrix :\n")
print(confusion_matrix(y_test, y_pred))

new_email = "Hi, here you have one free online course opportunity kindly claim it!"

new_email_transformed = cv.transform([new_email])

new_email_pred = model.predict(new_email_transformed)

print("New email (Hi, here you have one free online course opportunity kindly claim it!) : ",new_email_pred[0])