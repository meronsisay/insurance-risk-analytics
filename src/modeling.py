import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import shap
import joblib


class ModelingData:
    """Handle data preparation for modeling - each method does ONE thing cleanly"""

    @staticmethod
    def impute_gender_from_title(df):
        """Override Gender using Title clues."""
        df = df.copy()
        title_to_gender = {"Mr": "Male", "Mrs": "Female", "Ms": "Female", "Miss": "Female"}
        mask = df["Title"].isin(title_to_gender.keys())
        df.loc[mask, "Gender"] = df.loc[mask, "Title"].map(title_to_gender)
        df["Gender"] = df["Gender"].fillna("Not Specified")
        return df

    @staticmethod
    def engineer_features(df):
        """Create new features."""
        df = df.copy()
        if "RegistrationYear" in df.columns:
            df["vehicle_age"] = 2015 - df["RegistrationYear"]
            df["vehicle_age"] = df["vehicle_age"].clip(0, 20)
        df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"])
        df["policy_duration"] = (pd.to_datetime("2015-08-01") - df["TransactionMonth"]).dt.days

        # Add interaction features
        if "kilowatts" in df.columns and "cubiccapacity" in df.columns:
            df["power_density"] = df["kilowatts"] / (df["cubiccapacity"] + 1)

        return df

    @staticmethod
    def handle_missing_values(df):
        """Impute missing values."""
        df = df.copy()
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        cat_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna("Not Specified")
        return df

    @staticmethod
    def get_feature_columns(df, target_col):
        """Separate features and target."""
        df = df.copy()
        exclude_cols = [
            target_col,
            "UnderwrittenCoverID",
            "PolicyID",
            "TransactionMonth",
            "VehicleIntroDate",
            "TotalPremium",
            "TotalClaims",
            "Margin",
            "LossRatio",
            "HasClaim",
            "CalculatedPremiumPerTerm",
            "RegistrationYear",
            "PostalCode",
            "mmcode",
            "MainCrestaZone",
            "SubCrestaZone",
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols]
        y = df[target_col]
        return X, y, feature_cols

    @staticmethod
    def get_column_types(X):
        """Identify numerical and categorical columns."""
        categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
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
        """Prepares data with Label Encoding for tree-based models."""
        X_train_enc = X_train.copy()
        X_test_enc = X_test.copy()
        encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            X_train_enc[col] = le.fit_transform(X_train_enc[col].astype(str))
            encoders[col] = le

            # --- FIX: Replaced the slow, bugged line-by-line lambda search with a fast vectorized lookup dictionary ---
            lookup = dict(zip(le.classes_, le.transform(le.classes_)))
            X_test_enc[col] = X_test_enc[col].astype(str).map(lambda s: lookup.get(s, -1))

        scaler = StandardScaler()
        X_train_enc[numerical_cols] = scaler.fit_transform(X_train_enc[numerical_cols])
        X_test_enc[numerical_cols] = scaler.transform(X_test_enc[numerical_cols])

        return X_train_enc, X_test_enc, encoders, scaler

    @staticmethod
    def prepare_onehot_data(X_train, X_test, numerical_cols, categorical_cols):
        """Prepare data with One-Hot Encoding for Linear Regression."""
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_train_cat = onehot.fit_transform(X_train[categorical_cols])
        X_test_cat = onehot.transform(X_test[categorical_cols])

        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train[numerical_cols])
        X_test_num = scaler.transform(X_test[numerical_cols])

        X_train_lr = np.hstack([X_train_num, X_train_cat])
        X_test_lr = np.hstack([X_test_num, X_test_cat])

        return X_train_lr, X_test_lr, onehot, scaler

    @staticmethod
    def train_model(X_train, y_train):
        """Train Random Forest and XGBoost with hyperparameter tuning."""
        models = {}

        print("Training Random Forest with tuning...")
        rf_params = {
            "n_estimators": [100, 150],
            "max_depth": [8, 10, 12],
            "min_samples_split": [10, 20, 30],
            "min_samples_leaf": [5, 10, 15],
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="r2", n_jobs=-1, verbose=0)
        rf_grid.fit(X_train, y_train)
        models["Random Forest"] = rf_grid.best_estimator_
        print(f"  Best RF: {rf_grid.best_params_}, CV R²: {rf_grid.best_score_:.4f}")

        print("Training XGBoost with tuning...")
        xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

        xgb_params = {
            "n_estimators": [100, 150],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.7, 0.8],
            "reg_alpha": [0.5, 1],
            "reg_lambda": [3, 5],
        }

        xgb_grid = GridSearchCV(xgb, xgb_params, cv=5, scoring="r2", n_jobs=-1, verbose=0)
        xgb_grid.fit(X_train, y_train)
        models["XGBoost"] = xgb_grid.best_estimator_
        print(f"  Best XGB: {xgb_grid.best_params_}, CV R²: {xgb_grid.best_score_:.4f}")

        return models

    @staticmethod
    def train_linear_regression(X_train, y_train):
        """Train Ridge Regression with cross-validated alpha."""
        alphas = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]
        best_alpha = 1.0
        best_score = -np.inf

        for alpha in alphas:
            ridge = Ridge(alpha=alpha, random_state=42)
            scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring="r2")
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_alpha = alpha

        model = Ridge(alpha=best_alpha, random_state=42)
        model.fit(X_train, y_train)
        print(f"Best Ridge alpha: {best_alpha}, CV R²: {best_score:.4f}")

        return model

    # --- KEPT EXACTLY THE SAME (No Log Transform processing handled here) ---
    @staticmethod
    def evaluate_model(model, X_test, y_test_original, model_name):
        """Evaluate model trained"""
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # No negative claims
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
        r2 = r2_score(y_test_original, y_pred)
        return {"Model": model_name, "RMSE": rmse, "R²": r2}

    @staticmethod
    def downsample_data(df, target_col, sampling_ratio=20):
        """
        Down-sample non-claims to balance the dataset.
        sampling_ratio = number of non-claims per claim (default 20:1)
        """
        claims = df[df[target_col] == 1]
        non_claims = df[df[target_col] == 0]

        # Calculate sampling
        n_claims = len(claims)
        n_non_claims_target = n_claims * sampling_ratio
        non_claims_sampled = non_claims.sample(n=min(n_non_claims_target, len(non_claims)), random_state=42)

        # Calculate weight factor for calibration
        w = len(non_claims_sampled) / len(non_claims)

        # Combine and shuffle
        balanced_df = pd.concat([claims, non_claims_sampled]).sample(frac=1, random_state=42)

        return balanced_df, w

    @staticmethod
    def calibrate_probabilities(raw_proba, w):
        """
        Calibrate predicted probabilities from down-sampled data.
        w = (sampled_non_claims / total_non_claims)
        """
        raw_proba = np.clip(raw_proba, 1e-7, 1 - 1e-7)
        # --- FIX: Aligned the math step so the down-sampling weight factor 'w' corrects the negative class pool ---
        calibrated = (raw_proba * w) / (raw_proba * w + (1 - raw_proba))
        return calibrated

    @staticmethod
    def calculate_risk_premium(claim_probability, avg_severity, expense_loading=0.20, profit_margin=0.10):
        """Calculate risk-based premium."""
        loading = 1 + expense_loading + profit_margin
        return claim_probability * avg_severity * loading

    @staticmethod
    def get_claim_probability(classifier, X_test, w=None):
        """Get calibrated claim probability from classifier."""
        raw_proba = classifier.predict_proba(X_test)[:, 1]
        if w is not None:
            calibrated = ModelingData.calibrate_probabilities(raw_proba, w)
            return calibrated.mean(), calibrated
        else:
            return raw_proba.mean(), raw_proba

    @staticmethod
    def train_classifier(X_train, y_train):
        """Train classifiers for claim probability prediction."""
        models = {}

        print("Training Random Forest Classifier...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=15,
            class_weight="balanced",  # ← ADD THIS BACK
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        models["Random Forest Classifier"] = rf

        print("Training XGBoost Classifier...")
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos_weight,  # ← ADD THIS
            random_state=42,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        models["XGBoost Classifier"] = xgb
        return models

    @staticmethod
    def evaluate_classifier(model, X_test, y_test, model_name):
        """Evaluate classifier performance."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        return {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "AUC-ROC": roc_auc_score(y_test, y_pred_proba),
        }

    @staticmethod
    def shap_analysis(model, X_sample, feature_names, model_name):
        """Generate SHAP analysis for model interpretability using full dataset."""
        print(f"\nGenerating SHAP analysis for {model_name}...")
        print(f"  Using {len(X_sample)} samples for SHAP analysis")

        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values is a list or 3D tensor
        if isinstance(shap_values, list):
            shap_matrix = shap_values[1]  # Use positive class (claims)
        elif len(shap_values.shape) == 3:
            shap_matrix = shap_values[:, :, 1]
        else:
            shap_matrix = shap_values

        # Get top features (no plots)
        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": np.abs(shap_matrix).mean(axis=0)}
        ).sort_values("importance", ascending=False)

        return feature_importance.head(10)

    @staticmethod
    def save_model(model, path):
        """Save model to file."""
        joblib.dump(model, path)

    @staticmethod
    def load_model(path):
        """Load model from file."""
        return joblib.load(path)
