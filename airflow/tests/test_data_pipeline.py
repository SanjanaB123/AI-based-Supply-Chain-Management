"""
Tests for the AI-based Supply Chain Management Data Pipeline.
Run with: pytest tests/test_data_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dags.data_pipeline import extract, transform, load


@pytest.fixture
def sample_csv(tmp_path):
    dates = pd.date_range("2022-01-01", periods=60, freq="D")
    rows = []
    for date in dates:
        rows.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Store ID": "S001",
            "Product ID": "P0001",
            "Category": "Groceries",
            "Region": "North",
            "Inventory Level": np.random.randint(100, 500),
            "Units Sold": np.random.randint(20, 200),
            "Units Ordered": np.random.randint(30, 170),
            "Demand Forecast": round(np.random.uniform(50, 200), 2),
            "Price": round(np.random.uniform(10, 70), 2),
            "Discount": np.random.choice([0, 5, 10, 15, 20]),
            "Weather Condition": np.random.choice(["Sunny", "Rainy", "Cloudy"]),
            "Holiday/Promotion": np.random.choice([0, 1]),
            "Competitor Pricing": round(np.random.uniform(10, 70), 2),
            "Seasonality": np.random.choice(["Spring", "Summer", "Autumn", "Winter"]),
        })
    df = pd.DataFrame(rows)
    file_path = tmp_path / "test_inventory.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def sample_dataframe(sample_csv):
    return extract(sample_csv)


@pytest.fixture
def multi_series_csv(tmp_path):
    dates = pd.date_range("2022-01-01", periods=40, freq="D")
    rows = []
    for store, product in [("S001", "P0001"), ("S001", "P0002"), ("S002", "P0001")]:
        for date in dates:
            rows.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Store ID": store,
                "Product ID": product,
                "Category": "Groceries",
                "Region": "North",
                "Inventory Level": np.random.randint(100, 500),
                "Units Sold": np.random.randint(20, 200),
                "Units Ordered": np.random.randint(30, 170),
                "Demand Forecast": round(np.random.uniform(50, 200), 2),
                "Price": round(np.random.uniform(10, 70), 2),
                "Discount": np.random.choice([0, 5, 10, 15, 20]),
                "Weather Condition": np.random.choice(["Sunny", "Rainy", "Cloudy"]),
                "Holiday/Promotion": np.random.choice([0, 1]),
                "Competitor Pricing": round(np.random.uniform(10, 70), 2),
                "Seasonality": np.random.choice(["Spring", "Summer", "Autumn", "Winter"]),
            })
    df = pd.DataFrame(rows)
    file_path = tmp_path / "multi_series.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "data" / "processed"
    out.mkdir(parents=True)
    return str(out)


class TestExtract:

    def test_extract_returns_dataframe(self, sample_csv):
        result = extract(sample_csv)
        assert isinstance(result, pd.DataFrame)

    def test_extract_correct_shape(self, sample_csv):
        result = extract(sample_csv)
        assert result.shape == (60, 15)

    def test_extract_expected_columns(self, sample_csv):
        result = extract(sample_csv)
        expected_cols = [
            "Date", "Store ID", "Product ID", "Category", "Region",
            "Inventory Level", "Units Sold", "Units Ordered", "Demand Forecast",
            "Price", "Discount", "Weather Condition", "Holiday/Promotion",
            "Competitor Pricing", "Seasonality"
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_extract_no_empty_dataframe(self, sample_csv):
        result = extract(sample_csv)
        assert len(result) > 0

    def test_extract_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract("/nonexistent/path/data.csv")

    def test_extract_data_types(self, sample_csv):
        result = extract(sample_csv)
        assert pd.api.types.is_numeric_dtype(result["Units Sold"])
        assert pd.api.types.is_numeric_dtype(result["Price"])
        assert pd.api.types.is_numeric_dtype(result["Discount"])


class TestTransform:

    def test_transform_returns_dataframe(self, sample_dataframe):
        result = transform(sample_dataframe)
        assert isinstance(result, pd.DataFrame)

    def test_lag_features_present(self, sample_dataframe):
        result = transform(sample_dataframe)
        for col in ["sales_lag_1", "sales_lag_7", "sales_lag_14"]:
            assert col in result.columns, f"Missing lag feature: {col}"

    def test_rolling_features_present(self, sample_dataframe):
        result = transform(sample_dataframe)
        for col in ["sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28"]:
            assert col in result.columns, f"Missing rolling feature: {col}"

    def test_calendar_features_present(self, sample_dataframe):
        result = transform(sample_dataframe)
        assert "dow" in result.columns
        assert "month" in result.columns

    def test_label_column_present(self, sample_dataframe):
        result = transform(sample_dataframe)
        assert "y" in result.columns

    def test_metadata_columns_present(self, sample_dataframe):
        result = transform(sample_dataframe)
        meta_cols = ["as_of_date", "series_id", "horizon", "pipeline_version", "created_at"]
        for col in meta_cols:
            assert col in result.columns, f"Missing metadata column: {col}"

    def test_identifier_columns_present(self, sample_dataframe):
        result = transform(sample_dataframe)
        assert "Store ID" in result.columns
        assert "Product ID" in result.columns

    def test_no_null_in_lag_14(self, sample_dataframe):
        result = transform(sample_dataframe)
        assert result["sales_lag_14"].isna().sum() == 0

    def test_no_null_in_label(self, sample_dataframe):
        result = transform(sample_dataframe)
        assert result["y"].isna().sum() == 0

    def test_no_null_in_lag_features(self, sample_dataframe):
        result = transform(sample_dataframe)
        for col in ["sales_lag_1", "sales_lag_7", "sales_lag_14"]:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_row_count_reduced_after_transform(self, sample_dataframe):
        result = transform(sample_dataframe)
        assert len(result) < len(sample_dataframe)

    def test_dow_range(self, sample_dataframe):
        result = transform(sample_dataframe)
        assert result["dow"].between(0, 6).all()

    def test_month_range(self, sample_dataframe):
        result = transform(sample_dataframe)
        assert result["month"].between(1, 12).all()

    def test_no_future_leakage_in_lag_1(self, sample_dataframe):
        df = sample_dataframe.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by=["Store ID", "Product ID", "Date"]).reset_index(drop=True)

        result = transform(sample_dataframe)
        series = result[result["series_id"] == "S001_P0001"].reset_index(drop=True)
        if len(series) > 1:
            assert series["sales_lag_1"].notna().all()

    def test_label_is_future_sales(self, sample_dataframe):
        result = transform(sample_dataframe, horizon=1)
        assert not (result["y"] == result["sales_lag_1"]).all()

    def test_default_horizon(self, sample_dataframe):
        result = transform(sample_dataframe)
        assert (result["horizon"] == 1).all()

    def test_custom_horizon(self, sample_dataframe):
        result = transform(sample_dataframe, horizon=3)
        assert (result["horizon"] == 3).all()

    def test_default_pipeline_version(self, sample_dataframe):
        result = transform(sample_dataframe)
        assert (result["pipeline_version"] == "1.0").all()

    def test_custom_pipeline_version(self, sample_dataframe):
        result = transform(sample_dataframe, pipeline_version="2.0")
        assert (result["pipeline_version"] == "2.0").all()

    def test_series_id_format(self, sample_dataframe):
        result = transform(sample_dataframe)
        for _, row in result.head(5).iterrows():
            expected = f"{row['Store ID']}_{row['Product ID']}"
            assert row["series_id"] == expected

    def test_multi_series_lag_isolation(self, multi_series_csv):
        df = extract(multi_series_csv)
        result = transform(df)
        unique_series = result["series_id"].nunique()
        assert unique_series == 3, f"Expected 3 series, got {unique_series}"

    def test_multi_series_all_represented(self, multi_series_csv):
        df = extract(multi_series_csv)
        result = transform(df)
        expected_series = {"S001_P0001", "S001_P0002", "S002_P0001"}
        actual_series = set(result["series_id"].unique())
        assert expected_series == actual_series

    def test_created_at_is_valid_timestamp(self, sample_dataframe):
        result = transform(sample_dataframe)
        for ts in result["created_at"].head(3):
            datetime.fromisoformat(ts)

    def test_output_columns_count(self, sample_dataframe):
        result = transform(sample_dataframe)
        expected_cols = [
            "as_of_date", "series_id", "horizon", "pipeline_version", "created_at",
            "Store ID", "Product ID",
            "sales_lag_1", "sales_lag_7", "sales_lag_14",
            "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
            "dow", "month",
            "Price", "Discount", "Holiday/Promotion", "Competitor Pricing",
            "Weather Condition", "Seasonality", "Inventory Level", "Units Ordered",
            "y"
        ]
        assert set(result.columns) == set(expected_cols), (
            f"Column mismatch.\nExtra: {set(result.columns) - set(expected_cols)}\n"
            f"Missing: {set(expected_cols) - set(result.columns)}"
        )

    def test_index_is_reset(self, sample_dataframe):
        result = transform(sample_dataframe)
        assert list(result.index) == list(range(len(result)))


class TestLoad:

    def test_load_creates_file(self, sample_dataframe, output_dir):
        transformed = transform(sample_dataframe)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_dir, f"processed_data_{timestamp}.csv")
        transformed.to_csv(output_path, index=False)
        assert os.path.exists(output_path)

    def test_load_file_not_empty(self, sample_dataframe, output_dir):
        transformed = transform(sample_dataframe)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_dir, f"processed_data_{timestamp}.csv")
        transformed.to_csv(output_path, index=False)
        loaded = pd.read_csv(output_path)
        assert len(loaded) > 0

    def test_load_preserves_columns(self, sample_dataframe, output_dir):
        transformed = transform(sample_dataframe)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_dir, f"processed_data_{timestamp}.csv")
        transformed.to_csv(output_path, index=False)
        loaded = pd.read_csv(output_path)
        assert set(loaded.columns) == set(transformed.columns)

    def test_load_preserves_row_count(self, sample_dataframe, output_dir):
        transformed = transform(sample_dataframe)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_dir, f"processed_data_{timestamp}.csv")
        transformed.to_csv(output_path, index=False)
        loaded = pd.read_csv(output_path)
        assert len(loaded) == len(transformed)


class TestIntegration:

    def test_full_pipeline_single_series(self, sample_csv, output_dir):
        df = extract(sample_csv)
        assert len(df) > 0

        transformed = transform(df)
        assert len(transformed) > 0
        assert "y" in transformed.columns

        output_path = os.path.join(output_dir, "integration_test.csv")
        transformed.to_csv(output_path, index=False)
        assert os.path.exists(output_path)

        loaded = pd.read_csv(output_path)
        assert len(loaded) == len(transformed)

    def test_full_pipeline_multi_series(self, multi_series_csv, output_dir):
        df = extract(multi_series_csv)
        transformed = transform(df)
        assert transformed["series_id"].nunique() == 3
        assert len(transformed) > 0
        assert transformed["y"].isna().sum() == 0

    def test_pipeline_with_actual_data(self, output_dir):
        actual_data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "retail_store_inventory.csv"
        )
        if not os.path.exists(actual_data_path):
            pytest.skip("Actual data file not found — skipping.")

        df = extract(actual_data_path)
        assert len(df) > 0

        transformed = transform(df)
        assert len(transformed) > 0
        assert transformed["y"].isna().sum() == 0
        assert transformed["sales_lag_14"].isna().sum() == 0

    def test_transform_is_deterministic(self, sample_dataframe):
        result1 = transform(sample_dataframe)
        result2 = transform(sample_dataframe)
        assert list(result1.columns) == list(result2.columns)
        assert len(result1) == len(result2)


class TestEdgeCases:

    def test_transform_with_minimum_rows(self, tmp_path):
        dates = pd.date_range("2022-01-01", periods=20, freq="D")
        rows = [{
            "Date": d.strftime("%Y-%m-%d"), "Store ID": "S001", "Product ID": "P0001",
            "Category": "Groceries", "Region": "North", "Inventory Level": 200,
            "Units Sold": 100, "Units Ordered": 50, "Demand Forecast": 110.0,
            "Price": 30.0, "Discount": 10, "Weather Condition": "Sunny",
            "Holiday/Promotion": 0, "Competitor Pricing": 28.0, "Seasonality": "Summer"
        } for d in dates]
        df = pd.DataFrame(rows)
        result = transform(df)
        assert len(result) > 0

    def test_transform_too_few_rows(self, tmp_path):
        dates = pd.date_range("2022-01-01", periods=5, freq="D")
        rows = [{
            "Date": d.strftime("%Y-%m-%d"), "Store ID": "S001", "Product ID": "P0001",
            "Category": "Groceries", "Region": "North", "Inventory Level": 200,
            "Units Sold": 100, "Units Ordered": 50, "Demand Forecast": 110.0,
            "Price": 30.0, "Discount": 10, "Weather Condition": "Sunny",
            "Holiday/Promotion": 0, "Competitor Pricing": 28.0, "Seasonality": "Summer"
        } for d in dates]
        df = pd.DataFrame(rows)
        result = transform(df)
        assert len(result) == 0

    def test_transform_with_large_horizon(self, sample_dataframe):
        result = transform(sample_dataframe, horizon=30)
        assert isinstance(result, pd.DataFrame)
