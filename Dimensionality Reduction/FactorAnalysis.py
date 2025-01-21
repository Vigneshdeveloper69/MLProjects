from sklearn.decomposition import FactorAnalysis
import numpy as np

# Dummy psychological test scores
scores = np.array([
    [5, 3, 4, 3],
    [4, 2, 5, 3],
    [3, 3, 4, 5],
    [2, 1, 2, 4]
])

fa = FactorAnalysis(n_components=2)

transformed_scores = fa.fit_transform(scores)

print("Laten Factors :\n", transformed_scores)