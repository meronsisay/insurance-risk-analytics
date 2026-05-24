import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eda_utils import EDAUtils


class TestEDAUtils:

    @pytest.fixture
    def sample_df(self):
        """Create a small sample dataframe for testing"""
        return pd.DataFrame(
            {
                "TotalPremium": [100, 200, 0, 50, -10, 150],
                "TotalClaims": [10, 20, 0, 5, 0, 100],
                "Province": ["Gauteng", "WC", "Gauteng", "KZN", "Gauteng", "WC"],
                "Gender": ["Male", "Female", "Not specified", "Male", "Female", "Male"],
                "TransactionMonth": [
                    "2015-01-01",
                    "2015-02-01",
                    "2015-03-01",
                    "2015-01-01",
                    "2015-02-01",
                    "2015-03-01",
                ],
                "make": ["Toyota", "VW", "Toyota", "BMW", "Toyota", "VW"],
                "Model": ["Corolla", "Golf", "Corolla", "X5", "Corolla", "Golf"],
            }
        )

    def test_fix_invalid_values_removes_negatives(self, sample_df):
        """Test that negative premiums are removed"""
        df_clean = EDAUtils.fix_invalid_values(sample_df)
        assert (df_clean["TotalPremium"] >= 0).all()
        assert len(df_clean) < len(sample_df)

    def test_fix_invalid_values_removes_claims_without_premium(self, sample_df):
        """Test that claims with zero premium are removed"""
        df_clean = EDAUtils.fix_invalid_values(sample_df)
        invalid = df_clean[(df_clean["TotalClaims"] > 0) & (df_clean["TotalPremium"] == 0)]
        assert len(invalid) == 0

    def test_calculate_loss_ratio_excludes_zero_premium(self, sample_df):
        """Test that loss ratio only uses policies with premium > 0"""
        result = EDAUtils.calculate_loss_ratio(sample_df)
        assert result["overall"]["total_premium"] == 100 + 200 + 50 + 150  # excludes 0 and -10
        assert result["overall"]["total_claims"] == 10 + 20 + 5 + 100

    def test_calculate_loss_ratio_returns_dict(self, sample_df):
        """Test that calculate_loss_ratio returns correct structure"""
        result = EDAUtils.calculate_loss_ratio(sample_df)
        assert "overall" in result
        assert "by_province" in result
        assert "total_premium" in result["overall"]
        assert "total_claims" in result["overall"]
        assert "loss_ratio" in result["overall"]

    def test_get_missing_summary_returns_dataframe(self, sample_df):
        """Test that missing summary returns DataFrame"""
        result = EDAUtils.get_missing_summary(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert "Count" in result.columns
        assert "Percentage" in result.columns

    def test_get_descriptive_stats_returns_stats(self, sample_df):
        """Test that descriptive stats returns DataFrame with statistics"""
        result = EDAUtils.get_descriptive_stats(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert "mean" in result.index
        assert "std" in result.index

    def test_calculate_loss_ratio_by_gender(self, sample_df):
        """Test gender loss ratio calculation"""
        result = EDAUtils.calculate_loss_ratio_by_gender(sample_df)
        assert result is not None
        assert "Male" in result.index or "Female" in result.index


def test_import_eda_utils():
    """Test that eda_utils can be imported"""
    from src.eda_utils import EDAUtils

    assert EDAUtils is not None
