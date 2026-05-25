import pytest
import pandas as pd
import numpy as np
from src.hypothesis_tests import HypothesisTests


class TestHypothesisTests:

    def test_numerical_kpi_normal(self):
        """Test numerical KPI with normal data"""
        group_a = pd.Series([10, 11, 12, 13, 14])
        group_b = pd.Series([15, 16, 17, 18, 19])

        result = HypothesisTests.test_numerical_kpi(group_a, group_b, "Test KPI")

        assert result['kpi'] == "Test KPI"
        assert 'p_value' in result
        assert 'decision' in result

    def test_numerical_kpi_empty_groups(self):
        """Test numerical KPI with empty groups"""
        group_a = pd.Series([])
        group_b = pd.Series([1, 2, 3])

        result = HypothesisTests.test_numerical_kpi(group_a, group_b)

        assert result['decision'] == "Insufficient Data"
        assert np.isnan(result['p_value'])

    def test_numerical_kpi_both_empty(self):
        """Test numerical KPI with both groups empty"""
        group_a = pd.Series([])
        group_b = pd.Series([])

        result = HypothesisTests.test_numerical_kpi(group_a, group_b)

        assert result['decision'] == "Insufficient Data"

    def test_numerical_kpi_force_t_test(self):
        """Test numerical KPI with forced t-test"""
        group_a = pd.Series([10, 11, 12, 13, 14])
        group_b = pd.Series([15, 16, 17, 18, 19])

        result = HypothesisTests.test_numerical_kpi(group_a, group_b, force_t_test=True)

        assert result['test_used'] == "Welch's t-test (forced for margin analysis)"
        assert 'Welch' in result['test_used']

    def test_categorical_kpi(self):
        """Test categorical KPI with chi-squared test"""
        contingency = pd.DataFrame([[100, 10], [80, 5]])

        result = HypothesisTests.test_categorical_kpi(contingency, "Test Frequency")

        assert result['kpi'] == "Test Frequency"
        assert result['test_used'] == "Chi-squared test"
        assert 'statistic' in result
        assert 'p_value' in result

    def test_calculate_portfolio_loss_ratio(self):
        """Test portfolio loss ratio calculation"""
        df = pd.DataFrame({
            'TotalPremium': [100, 200, 300],
            'TotalClaims': [50, 100, 150],
            'group': ['A', 'A', 'B']
        })

        result = HypothesisTests.calculate_portfolio_loss_ratio(df, 'group', 'A')

        # Total Premium = 300, Total Claims = 150, Loss Ratio = 0.5
        assert result == 0.5

    def test_calculate_portfolio_loss_ratio_zero_premium(self):
        """Test portfolio loss ratio when premium is zero"""
        df = pd.DataFrame({
            'TotalPremium': [0, 0, 0],
            'TotalClaims': [50, 100, 150],
            'group': ['A', 'A', 'A']
        })

        result = HypothesisTests.calculate_portfolio_loss_ratio(df, 'group', 'A')

        assert result == 0

    def test_calculate_claim_frequency(self):
        """Test claim frequency calculation"""
        df = pd.DataFrame({
            'TotalClaims': [0, 100, 0, 200, 0],
            'group': ['A', 'A', 'A', 'B', 'B']
        })

        result = HypothesisTests.calculate_claim_frequency(df, 'group', 'A')

        # Claims > 0: 1 out of 3 policies (100 is claim, 0, 0) = 0.3333
        assert round(result, 4) == 0.3333

    def test_calculate_claim_frequency_no_claims(self):
        """Test claim frequency when no claims exist"""
        df = pd.DataFrame({
            'TotalClaims': [0, 0, 0, 0, 0],
            'group': ['A', 'A', 'A', 'B', 'B']
        })

        result = HypothesisTests.calculate_claim_frequency(df, 'group', 'A')

        assert result == 0

    def test_calculate_margin(self):
        """Test margin calculation"""
        df = pd.DataFrame({
            'TotalPremium': [100, 200, 300],
            'TotalClaims': [50, 100, 150],
            'group': ['A', 'A', 'B']
        })

        result = HypothesisTests.calculate_margin(df, 'group', 'A')

        # Margins: [50, 100], Average = 75
        assert result == 75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])