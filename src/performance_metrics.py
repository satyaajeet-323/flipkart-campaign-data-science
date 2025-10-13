# src/performance_metrics.py
import pandas as pd


def calculate_cpa(df: pd.DataFrame) -> pd.Series:
    """Calculate Cost Per Acquisition (CPA) for each campaign."""
    return df["spend"] / df["orders"]


def kpi_over_time(
    df: pd.DataFrame, date_col: str = "date", freq: str = "D"
) -> pd.DataFrame:
    """Calculate KPIs aggregated over time."""
    df[date_col] = pd.to_datetime(df[date_col])
    df_time = df.set_index(date_col)

    kpi_df = df_time.resample(freq).agg(
        {
            "spend": "sum",
            "orders": "sum",
            "clicks": "sum",
            "impressions": "sum",
        }
    )

    kpi_df["cpa"] = kpi_df["spend"] / kpi_df["orders"]
    kpi_df["ctr"] = (kpi_df["clicks"] / kpi_df["impressions"]) * 100
    kpi_df["conversion_rate"] = (kpi_df["orders"] / kpi_df["clicks"]) * 100

    return kpi_df.fillna(0)
