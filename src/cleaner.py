import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


class DataCleaner:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

        # Store encoder for every categorical column
        self.encoders = {}

    def load_data(self):

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Could not find dataset at: {self.filepath}"
            )

        self.df = pd.read_csv(self.filepath)

        if 'CustomerID' in self.df.columns:
            self.df = self.df.drop('CustomerID', axis=1)
            print("[*] Dropped 'CustomerID' (noise identifier).")

        leakage_columns = [
            'ConversionRate',
            'ClickThroughRate'
        ]

        for col in leakage_columns:
            if col in self.df.columns:
                self.df = self.df.drop(col, axis=1)
                print(
                    f"[*] Dropped '{col}' to prevent target leakage."
                )

        print(
            f"[*] Data loaded successfully. Final Shape: {self.df.shape}"
        )

        return self

    def clean_missing_values(self):

        # Clean numerical columns
        num_cols = self.df.select_dtypes(
            include=[np.number]
        ).columns

        for col in num_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(
                    self.df[col].median()
                )

        # Clean categorical columns
        cat_cols = self.df.select_dtypes(
            include=[object]
        ).columns

        for col in cat_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(
                    self.df[col].mode()[0]
                )

        print("[*] Missing values handled successfully.")

        return self

    def encode_categorical_data(self):

        cat_cols = self.df.select_dtypes(
            include=[object]
        ).columns

        for col in cat_cols:

            encoder = LabelEncoder()

            # Fit encoder on original category names
            self.df[col] = encoder.fit_transform(
                self.df[col]
            )

            # Save encoder for later user input
            self.encoders[col] = encoder

            print(f"[*] Encoded '{col}'")

            mapping = dict(
                zip(
                    encoder.classes_,
                    encoder.transform(encoder.classes_)
                )
            )

            print(f"    Mapping: {mapping}")

        print(
            "[*] Categorical features encoded into numbers."
        )

        return self

    def get_cleaned_data(self):
        return self.df


if __name__ == "__main__":

    file_path = (
        r"c:\Users\hardi\OneDrive\Pictures\Desktop"
        r"\optiasset\data\raw_marketing_data.csv"
    )

    cleaner = DataCleaner(filepath=file_path)

    cleaned_df = (
        cleaner
        .load_data()
        .clean_missing_values()
        .encode_categorical_data()
        .get_cleaned_data()
    )

    print("\nSample of Cleaned & Filtered Data Columns:")
    print(cleaned_df.columns.tolist())

    print("\nFirst 3 Rows:")
    print(cleaned_df.head(3))