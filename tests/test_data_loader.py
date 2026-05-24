import pytest
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import DataLoader


class TestDataLoader:

    @pytest.fixture
    def mock_csv_file(self, tmp_path):
        """Create a mock CSV file for testing"""
        csv_content = """145249|12827|2015-07-01|True||Close Corporation|Mr|English|First National Bank|Current account|Not specified|Not specified|South Africa|Gauteng|1459|Rand East|Rand East|Mobility - Motor|44069150|Passenger Vehicle|2004|MERCEDES-BENZ|E 240|6|2597|130|S/D|4|6/2002|119300|Yes|No|119300|More than 6 months||||||0.01|Monthly|25|Mobility - Windscreen|Windscreen|Windscreen|Comprehensive - Taxi|Motor Comprehensive|Mobility Metered Taxis: Monthly|Commercial|IFRS Constant|21.93|0.0"""

        csv_file = tmp_path / "test_data.txt"
        csv_file.write_text(csv_content)
        return csv_file

    def test_data_loader_initialization(self):
        """Test DataLoader initializes correctly"""
        loader = DataLoader("some/path.txt")
        assert loader.data_path == Path("some/path.txt")
        assert loader.df is None

    def test_load_data_returns_dataframe(self, mock_csv_file):
        """Test that load_data returns a DataFrame"""
        loader = DataLoader(mock_csv_file)
        df = loader.load_data()

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_data_has_correct_columns(self, mock_csv_file):
        """Test that loaded data has expected columns"""
        expected_columns = [
            "UnderwrittenCoverID",
            "PolicyID",
            "TransactionMonth",
            "IsVATRegistered",
            "Citizenship",
            "LegalType",
            "Title",
            "Language",
            "Bank",
            "AccountType",
            "MaritalStatus",
            "Gender",
            "Country",
            "Province",
            "PostalCode",
            "MainCrestaZone",
            "SubCrestaZone",
            "ItemType",
            "mmcode",
            "VehicleType",
            "RegistrationYear",
            "make",
            "Model",
            "Cylinders",
            "cubiccapacity",
            "kilowatts",
            "bodytype",
            "NumberOfDoors",
            "VehicleIntroDate",
            "CustomValueEstimate",
            "AlarmImmobiliser",
            "TrackingDevice",
            "CapitalOutstanding",
            "NewVehicle",
            "WrittenOff",
            "Rebuilt",
            "Converted",
            "CrossBorder",
            "NumberOfVehiclesInFleet",
            "SumInsured",
            "TermFrequency",
            "CalculatedPremiumPerTerm",
            "ExcessSelected",
            "CoverCategory",
            "CoverType",
            "CoverGroup",
            "Section",
            "Product",
            "StatutoryClass",
            "StatutoryRiskType",
            "TotalPremium",
            "TotalClaims",
        ]

        loader = DataLoader(mock_csv_file)
        df = loader.load_data()

        for col in expected_columns[:10]:  # Check first 10 columns
            assert col in df.columns

    def test_get_data_info_returns_dict(self, mock_csv_file):
        """Test that get_data_info returns correct dictionary"""
        loader = DataLoader(mock_csv_file)
        loader.load_data()
        info = loader.get_data_info()

        assert info is not None
        assert isinstance(info, dict)
        assert "rows" in info
        assert "columns" in info
        assert "column_names" in info
        assert "memory_mb" in info
        assert info["rows"] > 0
        assert info["columns"] > 0

    def test_get_data_info_returns_none_when_no_data(self):
        """Test that get_data_info returns None when no data loaded"""
        loader = DataLoader("nonexistent.txt")
        info = loader.get_data_info()
        assert info is None

    def test_print_info_runs_without_error(self, mock_csv_file, capsys):
        """Test that print_info executes without errors"""
        loader = DataLoader(mock_csv_file)
        loader.load_data()
        info = loader.get_data_info()

        loader.print_info(info)
        captured = capsys.readouterr()

        assert "DATASET INFORMATION" in captured.out
        assert "Rows:" in captured.out
        assert "Columns:" in captured.out

    def test_load_data_with_different_delimiter(self, mock_csv_file):
        """Test that data loads with pipe delimiter"""
        loader = DataLoader(mock_csv_file)
        df = loader.load_data()

        # Check that data is properly split (should have 52 columns)
        assert len(df.columns) == 52


def test_import_data_loader():
    """Test that data_loader can be imported"""
    from src.data_loader import DataLoader

    assert DataLoader is not None
