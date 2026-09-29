# src/cleaner.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

class DataCleaner:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        """Loads raw CSV data and drops identifiers and leaky targets."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Could not find dataset at: {self.filepath}")
            
        self.df = pd.read_csv(self.filepath)
        
        # 1. Drop the useless ID column (pure noise)
        if 'CustomerID' in self.df.columns:
            self.df = self.df.drop('CustomerID', axis=1)
            print("[*] Dropped 'CustomerID' (noise identifier).")
            
        # 2. Prevent Target Leakage 
        # (Dropping alternative success metrics so the model doesn't cheat during training)
        leakage_columns = ['ConversionRate', 'ClickThroughRate']
        for col in leakage_columns:
            if col in self.df.columns:
                self.df = self.df.drop(col, axis=1)
                print(f"[*] Dropped '{col}' to prevent target leakage.")
                
        print(f"[*] Data loaded successfully. Final Shape: {self.df.shape}")
        return self

    def clean_missing_values(self):
        """Fills missing numerical values with the median and categorical with the mode."""
        # Clean numerical columns
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # Clean categorical columns
        cat_cols = self.df.select_dtypes(include=[object]).columns
        for col in cat_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            
        print("[*] Missing values handled successfully.")
        return self

    def encode_categorical_data(self):
        """Converts text columns (like CampaignChannel) into numbers for NumPy math."""
        encoder = LabelEncoder()
        cat_cols = self.df.select_dtypes(include=[object]).columns
        
        for col in cat_cols:
            self.df[col] = encoder.fit_transform(self.df[col])
            
        print("[*] Categorical features encoded into numbers.")
        return self

    def get_cleaned_data(self):
        """Returns the fully cleaned DataFrame ready for PCA and Regression."""
        return self.df

# --- Quick Test to see if it works ---
if __name__ == "__main__":
    file_path = r"c:\Users\hardi\OneDrive\Pictures\Desktop\optiasset\data\raw_marketing_data.csv"
    cleaner = DataCleaner(filepath=file_path)
    cleaned_df = cleaner.load_data().clean_missing_values().encode_categorical_data().get_cleaned_data()
    
    print("\nSample of Cleaned & Filtered Data Columns:")
    print(cleaned_df.columns.tolist())
    print("\nFirst 3 Rows:")
    print(cleaned_df.head(3))