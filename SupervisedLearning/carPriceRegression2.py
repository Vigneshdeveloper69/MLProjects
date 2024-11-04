import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Creating a DataFrame
df = pd.read_csv('car_price2.csv')

X = df[['Mileage']]

y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

model = LinearRegression()

model.fit(X_poly_train, y_train)

y_pred = model.predict(X_poly_test)

print("MSE : ",mean_squared_error(y_test, y_pred))
print("MAE : ",mean_absolute_error(y_test, y_pred))
print("R² : ",r2_score(y_test, y_pred))

plt.scatter(X, y, color='blue', label='Actual data')
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot(X_range, model.predict(poly.transform(X_range)), color='red', label='Polynomial fit')
plt.xlabel('Mileage')
plt.ylabel('Price ($1000s)')
plt.legend()
plt.show()

new_mile = pd.DataFrame([[45000]], columns=['Mileage'])

new_mile_poly = poly.transform(new_mile)

new_mile_pred = model.predict(new_mile_poly)

print("New Vehicle (Mileage = 45000) : ",round(new_mile_pred[0], 2),"₹")

