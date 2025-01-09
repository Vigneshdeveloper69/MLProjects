import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from geopy.distance import great_circle

# Sample dataset (latitude and longitude)

df = pd.read_csv('gps_data.csv')

# Function to calculate haversine distance (in kilometers)
def haversine(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return great_circle(coords_1, coords_2).km

# Preprocess the data: scaling for distance calculation
coordinates = df[['latitude', 'longitude']].values
scaler = StandardScaler()
scaled_coordinates = scaler.fit_transform(coordinates)

# DBSCAN clustering
db = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')  # eps is the distance threshold
labels = db.fit_predict(scaled_coordinates)

# Adding the DBSCAN labels to the DataFrame
df['cluster'] = labels

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['longitude'], df['latitude'], c=df['cluster'], cmap='viridis', marker='o')
plt.title('DBSCAN Clustering of GPS Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster')
plt.show()

# Classifying regions based on density (cluster size)
cluster_counts = df['cluster'].value_counts()
high_traffic_clusters = cluster_counts[cluster_counts > 1]  # More than 1 point in the cluster = high traffic

# Show results
print("Cluster sizes:\n", cluster_counts)
print("\nHigh Traffic Clusters:\n", high_traffic_clusters)
