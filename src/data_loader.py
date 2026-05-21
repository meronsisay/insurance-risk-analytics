"""
Load and prepare the insurance data
Separates data loading logic from analysis
"""

from logging import info

import pandas as pd
from pathlib import Path

class DataLoader:
    """Load insurance data from pipe-delimited file"""
    
    def __init__(self, data_path):
        """
        Initialize with path to data file
        
        Args:
            data_path: Path to the .txt file (e.g., "data/MachineLearningRating_v3.txt")
        """
        self.data_path = Path(data_path)
        self.df = None
    def load_data(self):
        """Load the pipe-delimited data file"""
        print(f"Loading data from: {self.data_path}")
        
        # Define column names (from your data sample)
        column_names = [
            'UnderwrittenCoverID', 'PolicyID', 'TransactionMonth', 'IsVATRegistered',
            'Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 'AccountType',
            'MaritalStatus', 'Gender', 'Country', 'Province', 'PostalCode',
            'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'mmcode', 'VehicleType',
            'RegistrationYear', 'make', 'Model', 'Cylinders', 'cubiccapacity',
            'kilowatts', 'bodytype', 'NumberOfDoors', 'VehicleIntroDate',
            'CustomValueEstimate', 'AlarmImmobiliser', 'TrackingDevice',
            'CapitalOutstanding', 'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted',
            'CrossBorder', 'NumberOfVehiclesInFleet', 'SumInsured', 'TermFrequency',
            'CalculatedPremiumPerTerm', 'ExcessSelected', 'CoverCategory', 'CoverType',
            'CoverGroup', 'Section', 'Product', 'StatutoryClass', 'StatutoryRiskType',
            'TotalPremium', 'TotalClaims'
        ]
        
        self.df = pd.read_csv(
            self.data_path,
            delimiter='|',
            names=column_names,  
            low_memory=False
        )
        
        print(f"Loaded: {len(self.df):,} rows, {len(self.df.columns)} columns")
        return self.df
    
    def get_data_info(self):
        """Return basic information about the data"""
        if self.df is None:
            return None
        
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
    def print_info(self, info):
        """Print data info in nice format"""
        print("=" * 50)
        print("DATASET INFORMATION")
        print("=" * 50)
        print(f"Rows:   {info['rows']:,}")
        print(f"Cols:   {info['columns']}")
        print(f"Memory: {float(info['memory_mb']):.2f} MB")
        print("\nColumns:")
        for i, col in enumerate(info['column_names'], 1):
            print(f"  {i:2}. {col}")
        print("=" * 50)