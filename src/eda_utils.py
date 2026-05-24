import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EDAUtils:
    @staticmethod
    def fix_dtypes(df):
        """Fix data type issues cleanly without creating 'nan' string values."""
        df = df.copy()
        
        # IDs - Cast to string handling floats and nulls cleanly
        id_cols = ['UnderwrittenCoverID', 'PolicyID', 'PostalCode', 'mmcode', 'Mmcode']
        for col in id_cols:
            if col in df.columns:
                # Convert float-looking strings to standard integers first where possible, ignoring nulls
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64').astype(str)
                df[col] = df[col].replace('<NA>', np.nan)
        
        # Case-insensitive correction for columns like mmcode vs Mmcode
        numeric_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate', 'CapitalOutstanding', 
                        'Cylinders', 'cubiccapacity', 'Cubiccapacity', 'kilowatts', 'Kilowatts', 
                        'NumberOfDoors', 'SumInsured', 'CalculatedPremiumPerTerm', 'RegistrationYear']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Date columns
        date_cols = ['TransactionMonth', 'VehicleIntroDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df.dropna(axis=1, how='all')
    
    @staticmethod
    def get_descriptive_stats(df):
        """Get descriptive statistics for all numeric columns"""
        return df.select_dtypes(include=['number']).describe().round(2)
    
    @staticmethod
    def get_missing_summary(df):
        """Return missing values count and percentage"""
        missing = df.isnull().sum()
        pct = (missing / len(df)) * 100
        result = pd.DataFrame({'Count': missing, 'Percentage': pct})
        return result[result['Count'] > 0].sort_values('Percentage', ascending=False)
    
    @staticmethod
    def handle_missing_values(df):
        """Handle missing values safely, returning transformation mappings."""
        df = df.copy()
        missing_pct = (df.isnull().sum() / len(df)) * 100
        
        # Drop completely empty/useless columns (>60%)
        useless_cols = missing_pct[missing_pct > 60].index.tolist()
        if useless_cols:
            df = df.drop(columns=useless_cols)
            
        for col in df.columns:
            if not df[col].isnull().any():
                continue
                
            pct = missing_pct[col]
            
            if col in ['TotalPremium', 'TotalClaims'] or pct < 1:
                df = df.dropna(subset=[col])
            elif col in ['NewVehicle', 'IsVATRegistered', 'AlarmImmobiliser', 'TrackingDevice']:
                df[col] = df[col].fillna(0)
            elif df[col].dtype in ['float64', 'int64', 'Int64']:
                if pct < 30:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(0)
            else: # Categorical
                if pct < 30:
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
                else:
                    df[col] = df[col].fillna('Unknown')
        return df
    
    @staticmethod
    def fix_invalid_values(df):
        """Remove mathematically impossible values from insurance data."""
        df = df.copy()
        
        # Filter negative assets
        if 'TotalPremium' in df.columns:
            df = df[df['TotalPremium'] >= 0]
        if 'TotalClaims' in df.columns:
            df = df[df['TotalClaims'] >= 0]
            
        # Claims requiring a policy transaction validation
        if 'TotalClaims' in df.columns and 'TotalPremium' in df.columns:
            df = df[~((df['TotalClaims'] > 0) & (df['TotalPremium'] == 0))]
            
        return df
    
    @staticmethod
    def plot_histograms(df, columns, figsize=(15, 5)):
        """Plot histograms with clean aesthetic scaling using Seaborn context."""
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, len(columns), figsize=figsize)
        if len(columns) == 1: axes = [axes]
        
        for i, col in enumerate(columns):
            data = df[col].dropna()
            if len(data) > 0:
                sns.histplot(data, bins=50, ax=axes[i], kde=True, color='#2b5c8f')
                axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                axes[i].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
                axes[i].axvline(data.median(), color='green', linestyle='-', label=f'Med: {data.median():.2f}')
                axes[i].legend()
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_bar_charts(df, columns, figsize=(15, 5)):
        """Plot dynamic bar charts converting to horizontal layouts for long text labels."""
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, len(columns), figsize=figsize)
        if len(columns) == 1: axes = [axes]
        
        for i, col in enumerate(columns):
            if col in df.columns:
                counts = df[col].value_counts().head(10)
                
                # If labels are lengthy or there are many categories, swap to a horizontal bar chart
                if counts.index.astype(str).str.len().max() > 10:
                    sns.barplot(x=counts.values, y=counts.index.astype(str), ax=axes[i], palette="viridis", hue=counts.index.astype(str), legend=False)
                    axes[i].set_title(f'Top 10 {col}', fontsize=12, fontweight='bold')
                else:
                    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=axes[i], palette="viridis", hue=counts.index.astype(str), legend=False)
                    axes[i].set_title(f'Top 10 {col}', fontsize=12, fontweight='bold')
                    axes[i].tick_params(axis='x', rotation=30)
                    
        plt.tight_layout()
        return fig
    # Add these to your existing EDAUtils class in eda_utils.py

    @staticmethod
    def calculate_loss_ratio(df):
        """
        Calculate loss ratio overall and by segments
        
        Returns:
            Dictionary with loss ratio by different segments
        """
        # Use only valid policies (premium > 0)
        valid = df[df['TotalPremium'] > 0]
        
        # Overall
        overall = {
            'total_premium': valid['TotalPremium'].sum(),
            'total_claims': valid['TotalClaims'].sum(),
            'loss_ratio': valid['TotalClaims'].sum() / valid['TotalPremium'].sum()
        }
        
        # By Province
        province = valid.groupby('Province').apply(
            lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum()
        ).sort_values(ascending=False)
        
        # By Vehicle Type (if exists)
        vehicle = None
        if 'VehicleType' in valid.columns:
            vehicle = valid.groupby('VehicleType').apply(
                lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum()
            ).sort_values(ascending=False)
        
        return {
            'overall': overall,
            'by_province': province,
            'by_vehicle': vehicle
            }
    
    @staticmethod
    def analyze_temporal_trends(df):
        """
        Analyze claim frequency, severity, and loss ratio over time
        """
        df = df.copy()
        
        if 'TransactionMonth' not in df.columns:
            return None
        
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['YearMonth'] = df['TransactionMonth'].dt.to_period('M')
        df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
        
        # Simple aggregation
        freq = df.groupby('YearMonth')['HasClaim'].mean().reset_index()
        freq.columns = ['YearMonth', 'Claim_Frequency']
        
        severity = df.groupby('YearMonth')['TotalClaims'].mean().reset_index()
        severity.columns = ['YearMonth', 'Avg_Claim']
        
        premium = df.groupby('YearMonth')['TotalPremium'].sum().reset_index()
        premium.columns = ['YearMonth', 'Total_Premium']
        
        claims = df.groupby('YearMonth')['TotalClaims'].sum().reset_index()
        claims.columns = ['YearMonth', 'Total_Claims']
        
        # Merge all
        monthly = freq.merge(severity, on='YearMonth')
        monthly = monthly.merge(premium, on='YearMonth')
        monthly = monthly.merge(claims, on='YearMonth')
        
        # Calculate loss ratio safely
        monthly['LossRatio'] = monthly.apply(
            lambda row: row['Total_Claims'] / row['Total_Premium'] if row['Total_Premium'] > 0 else 0,
            axis=1
        )
        monthly['MonthStr'] = monthly['YearMonth'].astype(str)
        
        # Get first period with claims (not first row of data)
        first_claim_row = monthly[monthly['Claim_Frequency'] > 0].iloc[0] if len(monthly[monthly['Claim_Frequency'] > 0]) > 0 else None
        first_loss = monthly['LossRatio'].iloc[0]
        last_loss = monthly['LossRatio'].iloc[-1]
        
        # Better trend description
        if first_loss == 0 and last_loss > 0:
            trend = 'increasing (claims started after initial period)'
            change_pct = last_loss * 100  # meaningful: from 0 to last
            change_desc = f"Loss ratio increased from 0 to {last_loss:.2%}"
        elif first_loss == 0 and last_loss == 0:
            trend = 'stable (no claims recorded)'
            change_pct = 0
            change_desc = "No claims in dataset"
        else:
            change_pct = ((last_loss - first_loss) / first_loss) * 100
            trend = 'increasing' if last_loss > first_loss else 'decreasing'
            change_desc = f"{trend} by {abs(change_pct):.1f}%"
        
        return {
            'monthly_data': monthly,
            'trend': trend,
            'change_pct': change_pct,
            'change_description': change_desc,
            'first_month_with_claims': first_claim_row['YearMonth'] if first_claim_row is not None else None
        }
    
    @staticmethod
    def plot_temporal_trends(df, main_title="Temporal Trends"):
        """
        Create time series plots for claim frequency, severity, and loss ratio
        """
        trends = EDAUtils.analyze_temporal_trends(df)
        if trends is None:
            return None
        
        monthly = trends['monthly_data']
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Claim Frequency
        axes[0].plot(monthly['MonthStr'], monthly['Claim_Frequency'], 'b-o', linewidth=2, markersize=4)
        axes[0].set_title('Claim Frequency Over Time', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Proportion with Claims')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Claim Severity
        axes[1].plot(monthly['MonthStr'], monthly['Avg_Claim'], 'r-o', linewidth=2, markersize=4)
        axes[1].set_title('Claim Severity Over Time', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Average Claim (R)')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Loss Ratio
        axes[2].plot(monthly['MonthStr'], monthly['LossRatio'], 'g-o', linewidth=2, markersize=4)
        axes[2].axhline(y=1.0, color='red', linestyle='--', label='Breakeven (100%)')
        axes[2].set_title('Loss Ratio Over Time', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Loss Ratio')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].tick_params(axis='x', rotation=45)
        
        # FIX: Use suptitle with padding to avoid overlap
        plt.suptitle(main_title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def analyze_vehicle_claims(df, top_n=10):
        """
        Analyze which vehicle makes/models have highest/lowest claims
        
        Returns:
            Dictionary with top and bottom makes by claims
        """
        # Filter valid vehicle data
        vehicle_data = df[
            (df['make'].notna()) & 
            (df['make'] != '') & 
            (df['make'] != 'nan') &
            (df['TotalPremium'] > 0)
        ]
        
        if len(vehicle_data) == 0:
            return None
        
        # Claims by make
        make_claims = vehicle_data.groupby('make').agg({
            'TotalClaims': ['sum', 'mean', 'count']
        }).round(2)
        make_claims.columns = ['Total_Claims', 'Avg_Claim', 'Policy_Count']
        make_claims = make_claims.sort_values('Total_Claims', ascending=False)
        
        # Top models for highest risk makes
        top_makes = make_claims.head(3).index.tolist()
        top_models = {}
        
        for make in top_makes:
            models = vehicle_data[vehicle_data['make'] == make].groupby('Model').agg({
                'TotalClaims': 'sum'
            }).round(2)
            models = models.sort_values('TotalClaims', ascending=False).head(5)
            top_models[make] = models
        
        return {
            'top_makes': make_claims.head(top_n),
            'bottom_makes': make_claims.tail(top_n),
            'top_models': top_models
        }


    @staticmethod
    def calculate_loss_ratio_by_gender(df):
        """
        Calculate loss ratio by Gender
        """
        valid = df[df['TotalPremium'] > 0]
        
        if 'Gender' not in valid.columns:
            return None
        
        gender_loss = valid.groupby('Gender').apply(
            lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum()
        ).sort_values(ascending=False)
        
        return gender_loss