import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('india_t20_scores.csv')

X = df[['team', 'opponent', 'venue', 'innings']]

y = df['score']

categorial_features = ['team', 'opponent', 'venue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_boost_reg = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, verbose=0)

cat_boost_reg.fit(X_train, y_train, cat_features=categorial_features)

y_pred = cat_boost_reg.predict(X_test)

rmse = mean_squared_error(y_test, y_pred)

print("Mean_Squared_Error : ",round(rmse, 2))

cat_boost_reg.save_model('cat_boost_model.cbm')

print("Model was successfully saved!")

cat_boost_reg = CatBoostRegressor()

cat_boost_reg.load_model('cat_boost_model.cbm')

new_score_pred = cat_boost_reg.predict(pd.DataFrame([['India', 'Ireland', 'New York City', 1]], columns=X.columns))

print("Next match (India vs Ireland) India can score : ",int(round(new_score_pred[0], 0)))


