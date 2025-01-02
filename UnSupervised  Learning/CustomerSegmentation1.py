import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load Data
df = pd.read_csv('customer_data1.csv')

# 2. Data Preprocessing (Standardizing the data)
scaler = StandardScaler()
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
X_scaled = scaler.fit_transform(X)

# 3. Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Trying to segment customers into 3 groups
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. Visualize the Results
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
plt.title('Customer Segmentation Using K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster')
plt.show()

# Optionally, print the cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Show the customers and their corresponding cluster
print("\nCustomer Data with Cluster Labels:")
print(df)
