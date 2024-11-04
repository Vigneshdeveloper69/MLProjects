import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

df = pd.read_csv('real_estate_data.csv')

X = df.drop('price', axis=1)

y = df['price']

numerical_features = ['area', 'bedrooms', 'bathrooms', 'floors']
categorial_features = ['location']

numerical_transformer = StandardScaler()
categorial_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(

    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat',categorial_transformer, categorial_features)
    ]

)

model = xgb.XGBRegressor()

pipline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipline.fit(X_train, y_train)

y_pred = pipline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE : {mse:.2f}")
print(f"R2 Score : {r2:.2f}")

new_pred = pipline.predict(pd.DataFrame([[1500, 3, 2, 1, 'Suburb A']], columns=X.columns))

print("A price prediction with (area=1500, bedrooms=3, bathrooms=2, location=Suburb A) : ",int(round(new_pred[0], 0)))