# src/campaign_analyzer.py
from typing import Dict

import pandas as pd


def summarize_campaigns(df: pd.DataFrame) -> Dict:
    """Calculate basic campaign summary metrics."""
    total_spend = df["spend"].sum()
    total_orders = df["orders"].sum()
    total_clicks = df["clicks"].sum()
    total_impressions = df["impressions"].sum()

    overall_roi = (total_orders * 100) / total_spend if total_spend > 0 else 0
    ctr = (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0
    conversion_rate = (total_orders / total_clicks) * 100 if total_clicks > 0 else 0

    return {
        "total_campaigns": len(df),
        "total_spend": total_spend,
        "total_orders": total_orders,
        "total_clicks": total_clicks,
        "total_impressions": total_impressions,
        "overall_roi": round(overall_roi, 2),
        "ctr": round(ctr, 2),
        "conversion_rate": round(conversion_rate, 2),
    }


def top_campaigns_by_roi(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return top N campaigns by ROI."""
    df = df.copy()
    df["roi"] = (df["orders"] * 100) / df["spend"]
    df["roi"] = df["roi"].replace([float("inf"), -float("inf")], 0)
    return df.nlargest(top_n, "roi")[["campaign_id", "roi", "spend", "orders"]]
