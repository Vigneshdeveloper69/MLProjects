import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('csk_historical_data.csv')

df['Match_Outcome'] = df['Match_Outcome'].map({'Win':1,'Lose':0})

X = df[['Runs_Scored', 'Wickets_Taken', 'Toss_Win']]

y = df['Match_Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(f"Accuracy score :{accuracy_score(y_test, y_pred)*100}%")

print("Classification report :")
print(classification_report(y_test, y_pred))

new_match_pred = knn.predict(pd.DataFrame([[180, 10, 0]], columns=X.columns))

print(f"Next match csk (Scored_runs=180, Wickets_taken=10, Toss=Lose) will {'Win' if new_match_pred[0] == 1 else 'Lose'}")