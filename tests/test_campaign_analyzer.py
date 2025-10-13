# tests/test_campaign_analyzer.py
import pandas as pd
from src.campaign_analyzer import summarize_campaigns, top_campaigns_by_roi


def sample_df():
    return pd.DataFrame(
        [
            {"campaign_id": "c1", "spend": 100.0, "impressions": 10000, "clicks": 200, "orders": 10},
            {"campaign_id": "c2", "spend": 50.0, "impressions": 5000, "clicks": 50, "orders": 5},
            {"campaign_id": "c3", "spend": 0.0, "impressions": 1000, "clicks": 0, "orders": 0},
        ]
    )


def test_summarize_campaigns():
    df = sample_df()
    metrics = summarize_campaigns(df)
    assert metrics["total_spend"] == 150.0
    assert metrics["total_orders"] == 15
    # CTR = total_clicks / total_impressions = 250 / 16000
    assert abs(metrics["avg_ctr"] - (250 / 16000)) < 1e-9


def test_top_campaigns_by_roi():
    df = sample_df()
    top = top_campaigns_by_roi(df, top_n=2)
    assert "c2" in top["campaign_id"].values or "c1" in top["campaign_id"].values
