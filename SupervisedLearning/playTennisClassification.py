import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree

df = pd.read_csv('play_tennis.csv')

le = LabelEncoder()

df['Outlook'] = le.fit_transform(df['Outlook'])
df['Temperature'] = le.fit_transform(df['Temperature'])
df['Humidity'] = le.fit_transform(df['Humidity'])
df['Windy'] = df['Windy'].astype(int)
df['Play Tennis'] = df['Play Tennis'].map({'Yes':1, 'No':0})

X = df[['Outlook', 'Temperature', 'Humidity', 'Windy']]
y = df['Play Tennis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy : ", accuracy_score(y_test, y_pred))
print("Classification report :")
print(classification_report(y_test, y_pred))
print("Confusion matrix")
print(confusion_matrix(y_test, y_pred))

new_weather = pd.DataFrame([[2,1,0,1]], columns=['Outlook','Temperature','Humidity','Windy'])

new_weather_prediction = clf.predict(new_weather)

if new_weather_prediction[0] == 1:
    
    print("You will play the tennis")
else:
    
    print("You won\'t play the tennis")
    
plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=['No', 'Yes'])
plt.title("DecisionTreeClassfier")
plt.show()