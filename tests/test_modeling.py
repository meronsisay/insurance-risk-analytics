"""Simple tests for modeling.py - No heavy computation, just API validation."""

import pandas as pd
import numpy as np
from src.modeling import ModelingData


class TestModelingData:
    """Basic tests for ModelingData class methods."""

    def test_impute_gender_from_title(self):
        """Test gender imputation from titles."""
        df = pd.DataFrame({
            'Title': ['Mr', 'Mrs', 'Ms', 'Miss', 'Dr'],
            'Gender': ['Not Specified'] * 5
        })
        result = ModelingData.impute_gender_from_title(df)
        
        assert result.loc[0, 'Gender'] == 'Male'
        assert result.loc[1, 'Gender'] == 'Female'
        assert result.loc[2, 'Gender'] == 'Female'
        assert result.loc[3, 'Gender'] == 'Female'

    def test_engineer_features(self):
        """Test feature engineering creates expected columns."""
        df = pd.DataFrame({
            'RegistrationYear': [2010, 2015],
            'TransactionMonth': ['2015-01-01', '2015-06-01'],
            'kilowatts': [100, 150],
            'cubiccapacity': [1600, 2000]
        })
        result = ModelingData.engineer_features(df)
        
        assert 'vehicle_age' in result.columns
        assert 'policy_duration' in result.columns
        assert 'power_density' in result.columns

    def test_handle_missing_values(self):
        """Test missing value imputation."""
        df = pd.DataFrame({
            'num_col': [1, 2, None, 4, 5],
            'cat_col': ['A', 'B', None, 'A', 'B']
        })
        result = ModelingData.handle_missing_values(df)
        
        assert result['num_col'].isnull().sum() == 0
        assert result['cat_col'].isnull().sum() == 0

    def test_get_feature_columns(self):
        """Test feature/target separation with exclusions."""
        df = pd.DataFrame({
            'TotalClaims': [100, 200, 300],
            'UnderwrittenCoverID': [1, 2, 3],
            'PolicyID': [101, 102, 103],
            'IsVATRegistered': ['Yes', 'No', 'Yes'],
            'Citizenship': ['SA', 'SA', 'Other']
        })
        X, y, features = ModelingData.get_feature_columns(df, target_col='TotalClaims')
        
        assert 'TotalClaims' not in X.columns
        assert 'UnderwrittenCoverID' not in X.columns
        assert 'PolicyID' not in X.columns
        assert len(X.columns) == 2

    def test_get_column_types(self):
        """Test column type identification."""
        X = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [1.5, 2.5, 3.5],
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z']
        })
        numerical, categorical = ModelingData.get_column_types(X)
        
        assert 'num1' in numerical
        assert 'cat1' in categorical

    def test_calculate_risk_premium(self):
        """Test premium calculation formula."""
        premium = ModelingData.calculate_risk_premium(
            claim_probability=0.00264,
            avg_severity=21870
        )
        expected = 0.00264 * 21870 * 1.30
        assert abs(premium - expected) < 0.01

    def test_downsample_data(self):
        """Test downsampling creates balanced dataset."""
        df = pd.DataFrame({
            'HasClaim': [1] * 10 + [0] * 100,
            'feature': range(110)
        })
        balanced_df, w = ModelingData.downsample_data(df, 'HasClaim', sampling_ratio=5)
        
        # Check that we got a balanced dataset
        assert 0.1 < balanced_df['HasClaim'].mean() < 0.3
        assert 0 < w < 1