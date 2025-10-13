# src/customer_segmentation.py
from typing import Tuple

import pandas as pd
from sklearn.cluster import KMeans


def cluster_customers(
    features: pd.DataFrame, n_clusters: int = 3
) -> Tuple[pd.DataFrame, KMeans]:
    """
    Perform KMeans clustering and return dataframe with
    'cluster' column and the model.
    """
    # simple scaling — using raw values for simplicity
    # in real project use StandardScaler
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(features)

    features_clustered = features.copy()
    features_clustered["cluster"] = labels
    return features_clustered, model
