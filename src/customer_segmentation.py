# src/customer_segmentation.py
import pandas as pd
from sklearn.cluster import KMeans
from typing import Tuple


def prepare_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare aggregated customer-level features for clustering.
    Expects 'customer_id', 'order_value', 'orders', 'visits' columns (or similar).
    """
    # Example: aggregate by customer_id
    if "customer_id" not in df.columns:
        raise ValueError("Missing 'customer_id' column")

    agg = df.groupby("customer_id").agg(
        total_spend=("order_value", "sum"),
        total_orders=("orders", "sum"),
        total_visits=("visits", "sum"),
    ).fillna(0)
    return agg


def cluster_customers(features: pd.DataFrame, n_clusters: int = 3) -> Tuple[pd.DataFrame, KMeans]:
    """
    Perform KMeans clustering and return dataframe with 'cluster' column and the model.
    """
    # simple scaling — using raw values for simplicity; in real project use StandardScaler
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(features)
    out = features.copy()
    out["cluster"] = labels
    return out, model
