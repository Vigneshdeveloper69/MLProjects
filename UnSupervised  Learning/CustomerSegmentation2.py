import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load customer data from the CSV file
df = pd.read_csv('customer_data2.csv')

# Preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Age', 'Annual_Income', 'Spending_Score']])

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
reduced_data = tsne.fit_transform(scaled_data)

# Optional: Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

print(kmeans.cluster_centers_)

df['Cluster'] = clusters

print(df)

print(df['Cluster'].value_counts())
# Plot t-SNE results
plt.figure(figsize=(12, 8))
for cluster in np.unique(clusters):
    cluster_data = reduced_data[clusters == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')

plt.title("t-SNE Visualization of Customer Segmentation")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True)
plt.show()
