import numpy as np
import pandas as pd


def load_data(file_path):
    """Load and return dataset"""
    return pd.read_csv(file_path)


def clean_data(df):
    """Clean the dataset"""
    # Create a copy to avoid SettingWithCopyWarning
    df_clean = df.copy()

    # Remove duplicates
    df_clean = df_clean.drop_duplicates()

    # Handle missing values in numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    median_values = df_clean[numeric_cols].median()

    # Use .loc to avoid warning
    for col in numeric_cols:
        df_clean.loc[:, col] = df_clean[col].fillna(median_values[col])

    return df_clean


def get_basic_stats(df):
    """Get basic statistics"""
    return df.describe()
