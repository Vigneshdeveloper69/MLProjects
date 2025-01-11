import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Assuming you have your data in 'transaction_detail.csv'
df = pd.read_csv('transaction_detail.csv')

# Initialize LabelEncoder
label_encode = LabelEncoder()

# Apply LabelEncoder to each column individually
df['Transaction_ID'] = label_encode.fit_transform(df['Transaction_ID'])
df['Device_Used'] = label_encode.fit_transform(df['Device_Used'])
df['User_Location'] = label_encode.fit_transform(df['User_Location'])
df['IP_Address'] = label_encode.fit_transform(df['IP_Address'])
df['Shipping_Billing_Match'] = label_encode.fit_transform(df['Shipping_Billing_Match'])
df['Payment_Method'] = label_encode.fit_transform(df['Payment_Method'])
df['Failed_Attempts'] = label_encode.fit_transform(df['Failed_Attempts'])

# Data preprocessing: Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Transaction_Amount', 'Transaction_Time', 'Device_Used', 'User_Location', 
                                       'IP_Address', 'Shipping_Billing_Match', 'Payment_Method', 'Failed_Attempts']])

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(scaled_data, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Transactions")
plt.ylabel("Euclidean Distance")
plt.show()

# Perform Agglomerative Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
clusters = hc.fit_predict(scaled_data)

# Add clusters to the dataframe
df['Cluster'] = clusters
print(df)

# Visualize clusters with fraud highlighted
plt.figure(figsize=(10, 7))

# Assuming that cluster 0 represents fraud (you may need to adjust based on your clustering)
for i in range(len(df)):
    if df['Cluster'][i] == 0:  # Assuming cluster 0 is fraudulent
        plt.scatter(df['Transaction_Amount'][i], df['Transaction_Time'][i], color='red', label='Fraud' if 'Fraud' not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(df['Transaction_Amount'][i], df['Transaction_Time'][i], color='blue', label='Legitimate' if 'Legitimate' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("Clusters of Transactions")
plt.xlabel("Transaction Amount")
plt.ylabel("Transaction Time")
plt.legend()
plt.show()
