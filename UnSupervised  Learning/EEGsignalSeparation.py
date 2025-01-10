import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.io import loadmat  # To load EEG dataset

# Step 1: Load EEG Data
# Example EEG dataset in .mat format (replace with your own dataset)
# Download public datasets, e.g., from PhysioNet: https://physionet.org/
eeg_data = loadmat('eeg_data.mat')  # Replace 'eeg_data.mat' with your EEG dataset file

# Assuming 'data' key contains EEG signals (e.g., shape: [channels, time])
eeg_signals = eeg_data['data']  # Shape: (n_channels, n_samples)
n_channels, n_samples = eeg_signals.shape

# Step 2: Preprocess the Data (if required)
# Center and standardize signals
eeg_signals = eeg_signals - np.mean(eeg_signals, axis=1, keepdims=True)  # Mean center
eeg_signals = eeg_signals / np.std(eeg_signals, axis=1, keepdims=True)  # Normalize

# Step 3: Apply ICA to Separate Signals
ica = FastICA(n_components=n_channels)  # Number of components = number of channels
sources = ica.fit_transform(eeg_signals.T).T  # Fit and transform, then transpose

# Step 4: Visualize the Results
plt.figure(figsize=(12, 8))

# Plot original mixed EEG signals
for i in range(n_channels):
    plt.subplot(n_channels, 2, 2 * i + 1)
    plt.plot(eeg_signals[i])
    plt.title(f"Mixed Signal {i+1}")

# Plot separated independent components
for i in range(n_channels):
    plt.subplot(n_channels, 2, 2 * i + 2)
    plt.plot(sources[i])
    plt.title(f"Independent Component {i+1}")

plt.tight_layout()
plt.show()
