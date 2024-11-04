import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score

data = pd.read_csv('energy_consumption_data.csv')

data['date'] = pd.to_datetime(data['date'])

data.set_index('date', inplace = True)

features = data[['temperature', 'humidity']]
target = data['energy_consumption']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

gbm.fit(X_train, y_train)

y_pred = gbm.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2_score = r2_score(y_test, y_pred)

new_pred = gbm.predict(pd.DataFrame([[27.0, 75]], columns=['temperature', 'humidity']))

print("New predicition for energy consumption : ",round(new_pred[0]))