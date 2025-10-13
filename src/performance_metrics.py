# src/performance_metrics.py
from typing import Dict
import pandas as pd


def calculate_cpa(df: pd.DataFrame) -> float:
    """
    Calculate Cost Per Acquisition (CPA) = total_spend / total_orders.
    Handles zero-order case.
    """
    spend = df["spend"].sum()
    orders = df["orders"].sum()
    if orders == 0:
        return float("inf")
    return float(spend / orders)


def kpi_over_time(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Returns KPIs aggregated by date (sum of spend, orders, clicks, impressions).
    'date_col' should be parseable by pandas.to_datetime.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    agg = (
        df.groupby(df[date_col].dt.date)[["spend", "orders", "clicks", "impressions"]]
        .sum()
        .reset_index()
        .rename(columns={date_col: "date"})
    )
    return agg
