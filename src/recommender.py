# src/recommender.py
import numpy as np
from sklearn.neighbors import NearestNeighbors

class AssetRecommender:
    def __init__(self, n_neighbors=4):
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')

    def fit(self, X_pca):
        """Fits KNN on the PCA-reduced feature space."""
        self.nn_model.fit(X_pca)

    def find_similar_assets(self, asset_vector):
        """Finds nearest neighbor indices and distances for a given vector."""
        distances, indices = self.nn_model.kneighbors(asset_vector)
        return distances[0], indices[0]