#House price prediction using Simple LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv('house_sqft.csv')

X = df[['Square_Footage']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

new_pred_value = 1500

new_pred = model.predict(pd.DataFrame([[new_pred_value]], columns=['Square_Footage']))

print(f"Prediction for {new_pred_value}sqft : {new_pred[0]}$")