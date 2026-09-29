# src/features.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureProcessor:
    def __init__(self, df, target_col):
        self.df = df
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.pca_model = None

    def extract_and_scale_features(self, feature_cols):
        
        valid_cols = [col for col in feature_cols if col in self.df.columns]
        
        if len(valid_cols) == 0:
            raise ValueError("None of the specified feature columns were found in the dataset.")
            
        X = self.df[valid_cols]
        y = self.df[self.target_col].values
        
        X_scaled = self.scaler.fit_transform(X)
        print(f"[*] Extracted and scaled {len(valid_cols)} features.")
        return X_scaled, y

    def apply_pca(self, X_scaled, n_components=3):
        
        self.pca_model = PCA(n_components=n_components)
        X_pca = self.pca_model.fit_transform(X_scaled)
        
        explained_variance = np.sum(self.pca_model.explained_variance_ratio_) * 100
        print(f"[*] PCA applied. Reduced to {n_components} components. Variance Retained: {explained_variance:.2f}%")
        
        return X_pca