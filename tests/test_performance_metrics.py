# tests/test_performance_metrics.py
import pandas as pd
from src.performance_metrics import calculate_cpa, kpi_over_time


def test_calculate_cpa():
    df = pd.DataFrame([{"spend": 100, "orders": 2}, {"spend": 50, "orders": 1}])
    assert calculate_cpa(df) == 150.0 / 3


def test_kpi_over_time():
    df = pd.DataFrame(
        [
            {"date": "2023-01-01", "spend": 100, "orders": 1, "clicks": 10, "impressions": 100},
            {"date": "2023-01-01", "spend": 50, "orders": 2, "clicks": 5, "impressions": 50},
            {"date": "2023-01-02", "spend": 20, "orders": 0, "clicks": 1, "impressions": 10},
        ]
    )
    kpi = kpi_over_time(df)
    assert len(kpi) == 2
