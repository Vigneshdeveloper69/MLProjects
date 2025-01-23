import numpy as np
from sklearn.decomposition import TruncatedSVD

#user-item interaction matrix (ratings)
ratings = np.array([
    
    [5, 4, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 4, 4],
    [0, 0, 5, 4]
    
])

svd = TruncatedSVD(n_components=2)

transformed_ratings = svd.fit_transform(ratings)

print("Reduced Dimensionality : \n",transformed_ratings)