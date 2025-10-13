import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import clean_data, get_basic_stats

def test_clean_data():
    """Test data cleaning function"""
    sample_data = pd.DataFrame({
        'Type': ['Sale', 'Sale', 'Discount'],
        'Total_amt_of_sale': [1000, 1000, 2000],
        'conversion_rate': [3.5, 3.5, 4.0]
    })
    
    cleaned = clean_data(sample_data)
    assert len(cleaned) == 2  # Duplicates removed

def test_get_basic_stats():
    """Test statistics function"""
    sample_data = pd.DataFrame({
        'Total_amt_of_sale': [1000, 2000, 3000],
        'conversion_rate': [3.5, 4.0, 4.5]
    })
    
    stats = get_basic_stats(sample_data)
    assert 'mean' in stats.index