import pandas as pd
import numpy as np

def load_data(file_path):
    """Load and return dataset"""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the dataset"""
    # Remove duplicates
    df_clean = df.drop_duplicates()
    
    # Handle missing values in numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    return df_clean

def get_basic_stats(df):
    """Get basic statistics"""
    return df.describe()
    