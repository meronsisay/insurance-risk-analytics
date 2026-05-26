import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


class ModelingData:
    """Handle data preparation for modeling - each method does ONE thing cleanly"""

    @staticmethod
    def impute_gender_from_title(df):
        """Override Gender using Title clues."""
        df = df.copy()
        title_to_gender = {'Mr': 'Male', 'Mrs': 'Female', 'Ms': 'Female', 'Miss': 'Female'}
        mask = df['Title'].isin(title_to_gender.keys())
        df.loc[mask, 'Gender'] = df.loc[mask, 'Title'].map(title_to_gender)
        df['Gender'] = df['Gender'].fillna('Not Specified')
        return df

    @staticmethod
    def engineer_features(df):
        """Create new features."""
        df = df.copy()
        if 'RegistrationYear' in df.columns:
            df['vehicle_age'] = 2015 - df['RegistrationYear']
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['policy_duration'] = (pd.to_datetime('2015-08-01') - df['TransactionMonth']).dt.days
        return df

    @staticmethod
    def handle_missing_values(df):
        """Impute missing values."""
        df = df.copy()
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna('Not Specified')
        return df

    @staticmethod
    def get_feature_columns(df, target_col):
        """Separate features and target."""
        df = df.copy()
        exclude_cols = [
            target_col, 'UnderwrittenCoverID', 'PolicyID', 'TransactionMonth',
            'VehicleIntroDate', 'TotalPremium', 'Margin', 'LossRatio', 'HasClaim',
            'CalculatedPremiumPerTerm', 'RegistrationYear'
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols]
        y = df[target_col]
        return X, y, feature_cols

    @staticmethod
    def get_column_types(X):
        """Identify numerical and categorical columns."""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        return numerical_cols, categorical_cols

    @staticmethod
    def apply_label_encoding(X, categorical_cols):
        """Apply label encoding to categorical columns."""
        X = X.copy()
        encoder_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoder_dict[col] = le
        return X, encoder_dict

    @staticmethod
    def prepare_label_encoding_data(X_train, X_test, numerical_cols, categorical_cols):
        """
        Prepare data with Label Encoding for tree-based models (RF, XGB).
        Returns ready-to-use data.
        """
        # Label encode categorical columns
        X_train_enc, encoders = ModelingData.apply_label_encoding(X_train, categorical_cols)
        X_test_enc, _ = ModelingData.apply_label_encoding(X_test, categorical_cols)
        
        # Scale numerical columns
        scaler = StandardScaler()
        X_train_enc[numerical_cols] = scaler.fit_transform(X_train_enc[numerical_cols])
        X_test_enc[numerical_cols] = scaler.transform(X_test_enc[numerical_cols])
        
        return X_train_enc, X_test_enc, encoders, scaler

    @staticmethod
    def prepare_onehot_data(X_train, X_test, numerical_cols, categorical_cols):
        """
        Prepare data with One-Hot Encoding for Linear Regression.
        Returns numpy arrays ready for modeling.
        """
        # One-hot encode categorical columns
        onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_cat = onehot.fit_transform(X_train[categorical_cols])
        X_test_cat = onehot.transform(X_test[categorical_cols])
        
        # Scale numerical columns
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train[numerical_cols])
        X_test_num = scaler.transform(X_test[numerical_cols])
        
        # Combine numerical + one-hot
        X_train_lr = np.hstack([X_train_num, X_train_cat])
        X_test_lr = np.hstack([X_test_num, X_test_cat])
        
        return X_train_lr, X_test_lr, onehot, scaler

    @staticmethod
    def train_models_label_encoding(X_train, y_train):
        """
        Train Random Forest and XGBoost on label-encoded data.
        """
        models = {}
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
        
        # XGBoost
        xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb
        
        return models

    @staticmethod
    def train_linear_regression(X_train, y_train, use_ridge=True, alpha=1.0):
        """
        Train Linear or Ridge Regression on one-hot encoded data.
        """
        if use_ridge:
            model = Ridge(alpha=alpha, random_state=42)
        else:
            model = LinearRegression()
        
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_models(models, X_test, y_test):
        """Evaluate all models and return results DataFrame."""
        results = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            results.append({'Model': name, 'RMSE': rmse, 'R²': r2})
        return pd.DataFrame(results)

    @staticmethod
    def save_model(model, path):
        """Save model to file."""
        joblib.dump(model, path)

    @staticmethod
    def load_model(path):
        """Load model from file."""
        return joblib.load(path)