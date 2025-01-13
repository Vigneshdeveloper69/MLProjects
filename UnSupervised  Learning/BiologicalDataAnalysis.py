import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the gene expression dataset (Assume we have a dataset with genes as rows and samples as columns)
data = pd.read_csv('gene_expression.csv')

# Separate the Gene column (string identifiers)
gene_ids = data['Gene']  # Keep for reference
features = data.drop('Gene', axis=1)  # Drop the Gene column for clustering
# Normalize the gene expression data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Check the shape of the scaled data
print(f"Scaled data shape: {scaled_data.shape}")

# Fit GMM to the gene expression data
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(scaled_data)

# Predict the clusters for each gene
labels = gmm.predict(scaled_data)

# Combine gene identifiers and cluster labels
data = pd.DataFrame({
    'Gene': gene_ids,
    'cluster': labels
})

print(data.head())

from sklearn.decomposition import PCA

# Perform PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# Plot the data points colored by their GMM cluster
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=labels, palette='viridis', s=100, alpha=0.7)
plt.title('Gene Expression Data Clustering with GMM')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()
# Check which genes are in each cluster
for cluster_num in range(3):
    cluster_genes = data[data['cluster'] == cluster_num].index
    print(f"Genes in Cluster {cluster_num}:")
    print(cluster_genes)
    print()

