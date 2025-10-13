# tests/test_campaign_analyzer.py
import pandas as pd

from src.campaign_analyzer import summarize_campaigns, top_campaigns_by_roi


def sample_campaign_data():
    return pd.DataFrame(
        [
            {
                "campaign_id": "c1",
                "spend": 100.0,
                "impressions": 10000,
                "clicks": 200,
                "orders": 10,
            },
            {
                "campaign_id": "c2",
                "spend": 50.0,
                "impressions": 5000,
                "clicks": 50,
                "orders": 5,
            },
            {
                "campaign_id": "c3",
                "spend": 0.0,
                "impressions": 1000,
                "clicks": 0,
                "orders": 0,
            },
        ]
    )


def test_summarize_campaigns():
    df = sample_campaign_data()
    summary = summarize_campaigns(df)

    assert summary["total_campaigns"] == 3
    assert summary["total_spend"] == 150.0
    assert summary["total_orders"] == 15


def test_top_campaigns_by_roi():
    df = sample_campaign_data()
    top_campaigns = top_campaigns_by_roi(df, top_n=2)

    assert len(top_campaigns) == 2
    assert "roi" in top_campaigns.columns
