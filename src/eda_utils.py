import pandas as pd
import matplotlib.pyplot as plt

class EDAUtils:
    
    @staticmethod
    def get_missing_summary(df):
        """Return missing values count and percentage"""
        missing = df.isnull().sum()
        pct = (missing / len(df)) * 100
        return pd.DataFrame({'Count': missing, 'Percentage': pct})[missing > 0].sort_values('Percentage', ascending=False)
    
    @staticmethod
    def plot_histograms(df, columns, figsize=(15, 4)):
        """Plot histograms for given columns"""
        fig, axes = plt.subplots(1, len(columns), figsize=figsize)
        if len(columns) == 1:
            axes = [axes]
        
        for i, col in enumerate(columns):
            df[col].hist(bins=50, ax=axes[i], edgecolor='black', alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].axvline(df[col].mean(), color='red', linestyle='--', label='mean')
            axes[i].legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_bar_charts(df, columns, figsize=(15, 4)):
        """Plot bar charts for categorical columns"""
        fig, axes = plt.subplots(1, len(columns), figsize=figsize)
        if len(columns) == 1:
            axes = [axes]
        
        for i, col in enumerate(columns):
            df[col].value_counts().head(10).plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig