# tests/test_performance_metrics.py
import pandas as pd

from src.performance_metrics import calculate_cpa, kpi_over_time


def sample_performance_data():
    df = pd.DataFrame(
        [
            {
                "date": "2023-01-01",
                "spend": 100,
                "orders": 1,
                "clicks": 10,
                "impressions": 100,
            },
            {
                "date": "2023-01-01",
                "spend": 50,
                "orders": 2,
                "clicks": 5,
                "impressions": 50,
            },
            {
                "date": "2023-01-02",
                "spend": 20,
                "orders": 0,
                "clicks": 1,
                "impressions": 10,
            },
        ]
    )
    return df


def test_calculate_cpa():
    df = sample_performance_data()
    cpa_values = calculate_cpa(df)

    assert len(cpa_values) == 3
    assert cpa_values.iloc[0] == 100.0  # 100/1


def test_kpi_over_time():
    df = sample_performance_data()
    kpi_df = kpi_over_time(df)

    assert "cpa" in kpi_df.columns
    assert "ctr" in kpi_df.columns
    assert "conversion_rate" in kpi_df.columns
