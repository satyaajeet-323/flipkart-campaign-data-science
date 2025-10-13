# src/campaign_analyzer.py
from typing import Tuple, Dict
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset CSV. Returns DataFrame.
    """
    df = pd.read_csv(path)
    return df


def summarize_campaigns(df: pd.DataFrame) -> Dict[str, float]:
    """
    Returns summary metrics:
      - total_spend
      - total_orders
      - avg_ctr   (click-through rate)
      - conv_rate (conversion rate)
    Assumes columns: 'spend', 'impressions', 'clicks', 'orders'
    """
    # defensive column checks
    required = {"spend", "impressions", "clicks", "orders"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    total_spend = float(df["spend"].sum())
    total_orders = int(df["orders"].sum())
    total_clicks = float(df["clicks"].sum())
    total_impr = float(df["impressions"].sum())

    avg_ctr = (total_clicks / total_impr) if total_impr > 0 else 0.0
    conv_rate = (total_orders / total_clicks) if total_clicks > 0 else 0.0

    return {
        "total_spend": total_spend,
        "total_orders": total_orders,
        "avg_ctr": avg_ctr,
        "conv_rate": conv_rate,
    }


def top_campaigns_by_roi(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Compute ROI-like metric = orders / spend and return top N campaigns.
    Assumes column 'campaign_id' exists.
    """
    if "campaign_id" not in df.columns:
        raise ValueError("Missing 'campaign_id' column")

    # avoid division by zero
    df = df.copy()
    df["spend_safe"] = df["spend"].replace(0, 1e-9)
    df["roi_est"] = df["orders"] / df["spend_safe"]
    return df.sort_values("roi_est", ascending=False).head(top_n)[
        ["campaign_id", "spend", "orders", "roi_est"]
    ]
