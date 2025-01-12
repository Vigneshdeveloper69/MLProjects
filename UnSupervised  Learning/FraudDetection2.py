import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Create Synthetic Dataset
np.random.seed(42)

# Normal transactions
normal_data = {
    "TransactionAmount": np.random.normal(500, 100, 100),  # Avg amount: 500
    "TransactionType": np.random.choice([0, 1], 100, p=[0.6, 0.4]),  # 60% credit, 40% debit
    "AccountAge": np.random.normal(36, 5, 100)  # Avg account age: 36 months
}

# Fraudulent transactions
fraud_data = {
    "TransactionAmount": np.random.uniform(2000, 5000, 50),  # Unusually high amounts
    "TransactionType": np.random.choice([0, 1], 50, p=[0.2, 0.8]),  # Mostly debit
    "AccountAge": np.random.uniform(0, 5, 50)  # Very new accounts
}

# Combine normal and fraud data
df_normal = pd.DataFrame(normal_data)
df_fraud = pd.DataFrame(fraud_data)
df = pd.concat([df_normal, df_fraud], ignore_index=True)

# Add labels (0 = normal, 1 = fraud)
df["Label"] = [0] * len(df_normal) + [1] * len(df_fraud)

# Save dataset to CSV (optional)
df.to_csv("synthetic_banking_data.csv", index=False)

# Step 2: Preprocess Data
features = ["TransactionAmount", "TransactionType", "AccountAge"]
X = df[features].values
y = df["Label"].values

# Normalize data
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split data into train (normal only) and test sets
X_train = X[y == 0]  # Only normal data for training
X_test = X  # Full dataset for testing
y_test = y

# Step 3: Build Autoencoder
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation="relu")(input_layer)
encoded = Dense(4, activation="relu")(encoded)  # Bottleneck layer
decoded = Dense(8, activation="relu")(encoded)
output_layer = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adam", loss="mse")

# Step 4: Train Autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Step 5: Evaluate Reconstruction Error
reconstructed = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - reconstructed), axis=1)

# Step 6: Set Threshold and Detect Anomalies
threshold = np.percentile(reconstruction_error[y_test == 0], 95)  # 95th percentile of normal errors
anomalies = reconstruction_error > threshold

# Results
print(f"Detected Anomalies: {sum(anomalies)} / {len(y_test)}")
print(f"Accuracy: {np.mean(anomalies == y_test):.2f}")

# Step 7: Visualization
plt.hist(reconstruction_error[y_test == 0], bins=50, alpha=0.7, label="Normal")
plt.hist(reconstruction_error[y_test == 1], bins=50, alpha=0.7, label="Fraud")
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()
