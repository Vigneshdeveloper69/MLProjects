import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('user_adsclicks_data.csv')

X = df.drop(['click', 'ad_id', 'user_id'], axis=1)
y = df['click']

label_encoders = {}
for column in ['gender', 'country', 'device', 'time_of_day']:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_data = lgbm.Dataset(X_train, label=y_train)
test_data = lgbm.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'verbose': -1 
}

lgbm_model = lgbm.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data]
)


y_pred = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

accuracy = accuracy_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else "undefined"

print(f"Accuracy score: {accuracy * 100:.2f}%")
print(f"ROC AUC score: {roc_auc:.4f}" if roc_auc != "undefined" else "ROC AUC score cannot be computed")

new_user = pd.DataFrame({
    'age': [29], 
    'gender': ['Female'], 
    'country': ['Canada'],  
    'device': ['Mobile'],  
    'ad_position': [1],  
    'time_of_day': ['Afternoon'],  
    'impressions': [1000],  
})

for column in ['gender', 'country', 'device', 'time_of_day']:
    new_user[column] = label_encoders[column].transform(new_user[column])


next_click_prediction = lgbm_model.predict(new_user, num_iteration=lgbm_model.best_iteration)
next_click_binary = np.where(next_click_prediction > 0.5, 1, 0)

print("Next predicted click of user:", next_click_binary[0])
