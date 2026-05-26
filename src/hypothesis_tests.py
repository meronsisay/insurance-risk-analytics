import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency


class HypothesisTests:

    @staticmethod
    def test_numerical_kpi(group_a, group_b, kpi_name="KPI", force_t_test=False):
        """
        Test numerical KPIs using appropriate statistical test.

        Parameters:
        - force_t_test: Set to True for margin/profitability analysis where
                        mean differences matter more than rank differences
        """
        # Convert to pandas Series if needed and clean data
        if not isinstance(group_a, pd.Series):
            group_a = pd.Series(group_a)
        if not isinstance(group_b, pd.Series):
            group_b = pd.Series(group_b)

        # Clean data
        group_a = group_a.dropna()
        group_b = group_b.dropna()

        if len(group_a) == 0 or len(group_b) == 0:
            return {"kpi": kpi_name, "decision": "Insufficient Data", "p_value": np.nan}

        # FORCE T-TEST for margin/profitability analysis
        if force_t_test:
            stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
            test_used = "Welch's t-test (forced for margin analysis)"
            decision = "Reject H₀" if p_value < 0.05 else "Fail to reject H₀"

            return {
                "kpi": kpi_name,
                "test_used": test_used,
                "statistic": round(stat, 4),
                "p_value": p_value if p_value < 0.0001 else round(p_value, 4),
                "decision": decision,
                "significant": p_value < 0.05,
            }

        # Original logic for other cases
        zero_pct_a = (group_a == 0).mean()
        zero_pct_b = (group_b == 0).mean()
        is_zero_inflated = zero_pct_a > 0.5 or zero_pct_b > 0.5

        if len(group_a) > 5000 or len(group_b) > 5000 or is_zero_inflated:
            stat, p_value = stats.mannwhitneyu(group_a, group_b)
            test_used = "Mann-Whitney U test"
        else:
            # Small samples - check normality
            if len(group_a) >= 3 and len(group_b) >= 3:
                try:
                    _, p_norm_a = stats.shapiro(group_a)
                    _, p_norm_b = stats.shapiro(group_b)
                    is_normal = p_norm_a > 0.05 and p_norm_b > 0.05
                except (ValueError, TypeError):
                    is_normal = False
            else:
                is_normal = False

            if is_normal:
                stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
                test_used = "Welch's t-test"
            else:
                stat, p_value = stats.mannwhitneyu(group_a, group_b)
                test_used = "Mann-Whitney U test"

        decision = "Reject H₀" if p_value < 0.05 else "Fail to reject H₀"

        return {
            "kpi": kpi_name,
            "test_used": test_used,
            "statistic": round(stat, 4),
            "p_value": p_value if p_value < 0.0001 else round(p_value, 4),
            "decision": decision,
            "significant": p_value < 0.05,
        }
    @staticmethod
    def impute_gender_from_title(df):
        """
        Impute missing Gender using Title.
        Mr → Male, Mrs/Ms/Miss → Female. Excludes Dr and unmapped titles.
        """
        df = df.copy()
        title_to_gender = {'Mr': 'Male', 'Mrs': 'Female', 'Ms': 'Female', 'Miss': 'Female'}
        
        df['Gender_final'] = df['Gender'].replace('Not specified', np.nan)
        df['Gender_final'] = df['Gender_final'].fillna(df['Title'].map(title_to_gender))
        
        return df[df['Gender_final'].isin(['Male', 'Female'])]

    @staticmethod
    def test_categorical_kpi(contingency_table, kpi_name="Claim Frequency"):
        """Test categorical KPIs using Chi-squared test."""
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        decision = "Reject H₀" if p_value < 0.05 else "Fail to reject H₀"

        return {
            "kpi": kpi_name,
            "test_used": "Chi-squared test",
            "statistic": round(chi2, 4),
            "p_value": p_value,
            "decision": decision,
            "significant": p_value < 0.05,
        }

    @staticmethod
    def calculate_portfolio_loss_ratio(df, group_col, group_value):
        """Calculate overall portfolio loss ratio for a segment."""
        subset = df[df[group_col] == group_value]
        total_premium = subset["TotalPremium"].sum()
        total_claims = subset["TotalClaims"].sum()
        return total_claims / total_premium if total_premium > 0 else 0

    @staticmethod
    def calculate_claim_frequency(df, group_col, group_value):
        """Calculate claim frequency for a segment."""
        subset = df[df[group_col] == group_value]
        return (subset["TotalClaims"] > 0).mean()

    @staticmethod
    def calculate_margin(df, group_col, group_value):
        """Calculate average margin for a segment."""
        subset = df[df[group_col] == group_value]
        return (subset["TotalPremium"] - subset["TotalClaims"]).mean()
