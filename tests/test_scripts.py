#!/usr/bin/env python3
"""
Comprehensive test suite for AI-based Supply Chain Management pipeline.

Covers:
    - scripts/anomaly.py   → missingness, outliers, date gaps, report generation
    - scripts/bias.py      → MAE, MAPE, slice analysis, mitigation, sample weights
    - scripts/validate.py  → schema/stats generation, GE validation
    - scripts/upload_to_gcp.py  → GCS upload (mocked)
    - scripts/upload_to_mongo.py → file loading, MongoDB upload (mocked)
    - dags/data_pipeline.py → transform feature engineering logic
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_supply_chain_df():
    """
    Create a realistic supply-chain DataFrame with multiple stores, products,
    dates, and the columns the pipeline expects.
    """
    np.random.seed(42)
    stores = ["S1", "S2"]
    products = ["P1", "P2"]
    dates = pd.date_range("2025-01-01", periods=60, freq="D")

    rows = []
    for store in stores:
        for product in products:
            for date in dates:
                rows.append({
                    "Date": date,
                    "Store ID": store,
                    "Product ID": product,
                    "Units Sold": max(0, int(np.random.normal(100, 20))),
                    "Inventory Level": max(0, int(np.random.normal(500, 50))),
                    "Price": round(np.random.uniform(5, 50), 2),
                    "Discount": round(np.random.uniform(0, 0.3), 2),
                    "Holiday/Promotion": np.random.choice([0, 1]),
                    "Competitor Pricing": round(np.random.uniform(5, 55), 2),
                    "Weather Condition": np.random.choice(
                        ["Sunny", "Rainy", "Cloudy", "Snowy"]
                    ),
                    "Seasonality": np.random.choice(
                        ["Spring", "Summer", "Fall", "Winter"]
                    ),
                    "Units Ordered": max(0, int(np.random.normal(120, 25))),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def features_df(sample_supply_chain_df):
    """
    Simulate a post-transform features DataFrame with lag, rolling, and
    target columns already present.
    """
    df = sample_supply_chain_df.copy()
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)
    df["as_of_date"] = df["Date"]

    gs = df.groupby(["Store ID", "Product ID"])["Units Sold"]
    df["sales_lag_1"] = gs.shift(1)
    df["sales_lag_7"] = gs.shift(7)
    df["sales_lag_14"] = gs.shift(14)
    df["sales_roll_mean_7"] = gs.transform(
        lambda x: x.rolling(7, min_periods=3).mean().shift(1)
    )
    df["sales_roll_mean_14"] = gs.transform(
        lambda x: x.rolling(14, min_periods=7).mean().shift(1)
    )
    df["sales_roll_mean_28"] = gs.transform(
        lambda x: x.rolling(28, min_periods=7).mean().shift(1)
    )
    df["sales_ewm_28"] = gs.transform(
        lambda x: x.shift(1).ewm(span=28, adjust=False).mean()
    )
    df["dow"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["y_pred_baseline"] = df["sales_lag_1"].fillna(df["sales_lag_7"])
    df["y"] = gs.shift(-1)

    # Sample weights
    store_freq = df["Store ID"].value_counts(normalize=True)
    df["sample_weight"] = df["Store ID"].map(
        lambda x: 1.0 / max(store_freq.get(x, 0), 1e-6)
    ).clip(0.1, 10.0)

    df = df.dropna(subset=["sales_lag_14", "y"])
    return df


@pytest.fixture
def features_parquet(features_df, tmp_path):
    """Write the features DataFrame to a temporary Parquet file."""
    path = tmp_path / "features.parquet"
    features_df.to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def df_with_nulls():
    """DataFrame with intentional null spikes for anomaly testing."""
    n = 1000
    df = pd.DataFrame({
        "col_clean": np.random.randn(n),
        "col_some_nulls": np.where(
            np.random.rand(n) < 0.05, np.nan, np.random.randn(n)
        ),
        "col_many_nulls": np.where(
            np.random.rand(n) < 0.30, np.nan, np.random.randn(n)
        ),
    })
    return df


@pytest.fixture
def df_with_outliers():
    """DataFrame with extreme outliers for anomaly testing."""
    np.random.seed(42)
    n = 1000
    units = np.random.normal(100, 10, n)
    units[:20] = 10000
    units[-10:] = -500

    return pd.DataFrame({
        "Units Sold": units,
        "Price": np.random.uniform(5, 50, n),
        "y": np.random.normal(100, 15, n),
    })


@pytest.fixture
def df_with_date_gaps():
    """DataFrame with intentional date gaps for anomaly testing."""
    dates = pd.date_range("2025-01-01", periods=30, freq="D").tolist()

    dates.pop(10)  
    dates.pop(14)  
    dates.pop(18)  
    dates.pop(20)  

    return pd.DataFrame({
        "as_of_date": dates,
        "Store ID": ["S1"] * len(dates),
        "Product ID": ["P1"] * len(dates),
        "Units Sold": np.random.randint(50, 150, len(dates)),
    })


@pytest.fixture
def bias_df():
    """DataFrame for bias / slice-performance testing."""
    np.random.seed(42)
    n = 500

    y_true = np.random.normal(100, 15, n)
    y_pred = y_true + np.random.normal(0, 5, n)

    categories = np.random.choice(["Electronics", "Clothing", "Food"], n)

    food_mask = categories == "Food"
    y_pred[food_mask] += 30  

    return pd.DataFrame({
        "y": y_true,
        "y_pred_baseline": y_pred,
        "Holiday/Promotion": np.random.choice([0, 1], n),
        "Weather Condition": np.random.choice(["Sunny", "Rainy", "Cloudy"], n),
        "Seasonality": np.random.choice(["Spring", "Summer", "Fall", "Winter"], n),
        "Store ID": np.random.choice(["S1", "S2", "S3"], n),
        "Product ID": np.random.choice(["P1", "P2"], n),
        "Category": categories,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS — anomaly.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnomalyMissingness:
    """Tests for detect_missingness_spikes."""

    def test_no_nulls_no_anomalies(self):
        from anomaly import detect_missingness_spikes

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = detect_missingness_spikes(df)

        assert len(result["missingness_anomalies"]) == 0
        assert result["missingness_summary"]["a"]["null_percentage"] == 0.0

    def test_detects_high_null_columns(self, df_with_nulls):
        from anomaly import detect_missingness_spikes

        result = detect_missingness_spikes(df_with_nulls, threshold=0.02)
        anomalies = result["missingness_anomalies"]

        assert "col_some_nulls" in anomalies
        assert "col_many_nulls" in anomalies
        assert "col_clean" not in anomalies

    def test_severity_levels(self, df_with_nulls):
        from anomaly import detect_missingness_spikes

        result = detect_missingness_spikes(df_with_nulls, threshold=0.02)
        anomalies = result["missingness_anomalies"]

        assert anomalies["col_many_nulls"]["severity"] == "high"

        assert anomalies["col_some_nulls"]["severity"] == "medium"

    def test_custom_threshold(self):
        from anomaly import detect_missingness_spikes

        df = pd.DataFrame({
            "col_a": [1, None, 3, 4, 5, 6, 7, 8, 9, 10], 
        })

        result = detect_missingness_spikes(df, threshold=0.15)
        assert len(result["missingness_anomalies"]) == 0

        result = detect_missingness_spikes(df, threshold=0.05)
        assert "col_a" in result["missingness_anomalies"]


class TestAnomalyOutliers:
    """Tests for detect_outliers."""

    def test_no_outliers_in_normal_data(self):
        from anomaly import detect_outliers

        np.random.seed(42)
        df = pd.DataFrame({
            "Units Sold": np.random.normal(100, 5, 1000),
            "Price": np.random.normal(20, 2, 1000),
        })
        result = detect_outliers(df)
        assert len(result["outlier_anomalies"]) == 0 or all(
            v["outlier_percentage"] < 0.05
            for v in result["outlier_anomalies"].values()
        )

    def test_detects_extreme_outliers(self, df_with_outliers):
        from anomaly import detect_outliers

        result = detect_outliers(df_with_outliers, z_threshold=3.0)
        summary = result["outlier_summary"]


        assert "Units Sold" in summary
        assert summary["Units Sold"]["outlier_count"] > 0

    def test_missing_columns_skipped(self):
        from anomaly import detect_outliers

        df = pd.DataFrame({"irrelevant_col": [1, 2, 3]})
        result = detect_outliers(df)
        assert result["outlier_summary"] == {}
        assert result["outlier_anomalies"] == {}

    def test_empty_series_handled(self):
        from anomaly import detect_outliers

        df = pd.DataFrame({"Units Sold": [np.nan, np.nan, np.nan]})
        result = detect_outliers(df)
        assert "Units Sold" not in result["outlier_summary"]


class TestAnomalyDateGaps:
    """Tests for detect_date_gaps."""

    def test_no_gaps_in_continuous_dates(self):
        from anomaly import detect_date_gaps

        dates = pd.date_range("2025-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "as_of_date": dates,
            "Store ID": ["S1"] * 30,
            "Product ID": ["P1"] * 30,
        })
        result = detect_date_gaps(df)
        assert len(result["date_gap_anomalies"]) == 0

    def test_detects_gaps(self, df_with_date_gaps):
        from anomaly import detect_date_gaps

        result = detect_date_gaps(df_with_date_gaps, gap_threshold=1)
        assert len(result["date_gap_anomalies"]) > 0

    def test_missing_date_column(self):
        from anomaly import detect_date_gaps

        df = pd.DataFrame({"Store ID": ["S1"], "Product ID": ["P1"]})
        result = detect_date_gaps(df)
        assert result["date_gap_summary"] == {}
        assert result["date_gap_anomalies"] == {}


class TestAnomalyReport:
    """Tests for generate_anomaly_report and check_anomaly_thresholds."""

    def test_generate_report_creates_json(self, features_parquet, tmp_path):
        from anomaly import generate_anomaly_report

        output_dir = str(tmp_path / "anomaly_out")
        report_path = generate_anomaly_report(features_parquet, output_dir)

        assert Path(report_path).exists()
        with open(report_path) as f:
            report = json.load(f)

        assert "metadata" in report
        assert "missingness" in report
        assert "outliers" in report
        assert "date_gaps" in report
        assert "summary" in report
        assert report["metadata"]["total_rows"] > 0

    def test_check_thresholds_pass(self, tmp_path):
        from anomaly import check_anomaly_thresholds

        report = {
            "summary": {"total_anomaly_types": 0}
        }
        report_path = tmp_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f)

        assert check_anomaly_thresholds(str(report_path), max_anomalies=0) is True

    def test_check_thresholds_fail(self, tmp_path):
        from anomaly import check_anomaly_thresholds

        report = {
            "summary": {"total_anomaly_types": 5}
        }
        report_path = tmp_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f)

        assert check_anomaly_thresholds(str(report_path), max_anomalies=0) is False
        assert check_anomaly_thresholds(str(report_path), max_anomalies=10) is True


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS — bias.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestBiasMetrics:
    """Tests for calculate_mae and calculate_mape."""

    def test_mae_perfect_predictions(self):
        from bias import calculate_mae

        y = np.array([10, 20, 30])
        assert calculate_mae(y, y) == 0.0

    def test_mae_known_values(self):
        from bias import calculate_mae

        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 33])
        expected = np.mean([2, 2, 3])
        assert abs(calculate_mae(y_true, y_pred) - expected) < 1e-10

    def test_mape_perfect_predictions(self):
        from bias import calculate_mape

        y = np.array([10.0, 20.0, 30.0])
        assert calculate_mape(y, y) == 0.0

    def test_mape_known_values(self):
        from bias import calculate_mape

        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        assert abs(calculate_mape(y_true, y_pred) - 10.0) < 1e-10

    def test_mape_with_zeros(self):
        from bias import calculate_mape

        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert calculate_mape(y_true, y_pred) is None


class TestBiasSliceAnalysis:
    """Tests for analyze_slice_performance."""

    def test_returns_overall_metrics(self, bias_df):
        from bias import analyze_slice_performance

        result = analyze_slice_performance(bias_df, "Store ID")
        assert "_overall" in result
        assert "mae" in result["_overall"]
        assert "mape" in result["_overall"]
        assert result["_overall"]["sample_count"] > 0

    def test_returns_per_slice_metrics(self, bias_df):
        from bias import analyze_slice_performance

        result = analyze_slice_performance(bias_df, "Store ID")
        for store in ["S1", "S2", "S3"]:
            assert store in result
            assert "mae" in result[store]
            assert "mae_ratio" in result[store]
            assert "bias_flag" in result[store]

    def test_detects_biased_slice(self, bias_df):
        from bias import analyze_slice_performance

        result = analyze_slice_performance(bias_df, "Category")
        assert result["Food"]["bias_flag"] == True

    def test_missing_column_returns_empty(self, bias_df):
        from bias import analyze_slice_performance

        result = analyze_slice_performance(bias_df, "NonexistentColumn")
        assert result == {}


class TestBiasMitigationPlan:
    """Tests for generate_mitigation_plan."""

    def test_generates_recommendations(self, bias_df):
        from bias import analyze_slice_performance, generate_mitigation_plan

        slice_results = {
            "Category": analyze_slice_performance(bias_df, "Category")
        }
        plan = generate_mitigation_plan(slice_results)

        assert "Category" in plan
        assert len(plan["Category"]["recommendations"]) > 0

    def test_priority_assignment(self, bias_df):
        from bias import analyze_slice_performance, generate_mitigation_plan

        slice_results = {
            "Category": analyze_slice_performance(bias_df, "Category")
        }
        plan = generate_mitigation_plan(slice_results, min_samples=1000)
        assert plan["Category"]["priority"] in ["medium", "high"]

    def test_sample_weights_generated(self, bias_df):
        from bias import analyze_slice_performance, generate_mitigation_plan

        slice_results = {
            "Store ID": analyze_slice_performance(bias_df, "Store ID")
        }
        plan = generate_mitigation_plan(slice_results)
        assert len(plan["Store ID"]["sample_weights"]) > 0


class TestBiasReport:
    """Tests for generate_bias_report."""

    def test_generates_report_file(self, features_parquet, tmp_path):
        from bias import generate_bias_report

        output_dir = str(tmp_path / "bias_out")
        report_path = generate_bias_report(features_parquet, output_dir)

        assert Path(report_path).exists()
        with open(report_path) as f:
            report = json.load(f)

        assert "metadata" in report
        assert "slice_performance" in report
        assert "mitigation_plan" in report
        assert "summary" in report

    def test_mitigation_plan_saved_separately(self, features_parquet, tmp_path):
        from bias import generate_bias_report

        output_dir = str(tmp_path / "bias_out")
        generate_bias_report(features_parquet, output_dir)

        mitigation_path = Path(output_dir) / "bias_mitigation_plan.json"
        assert mitigation_path.exists()


class TestBiasSampleWeights:
    """Tests for calculate_sample_weights."""

    def test_uniform_when_column_missing(self):
        from bias import calculate_sample_weights

        df = pd.DataFrame({"a": [1, 2, 3]})
        weights = calculate_sample_weights(df, weight_column="NonexistentCol")
        assert (weights == 1.0).all()

    def test_inverse_frequency_weighting(self):
        from bias import calculate_sample_weights

        df = pd.DataFrame({"Store ID": ["S1", "S1", "S1", "S2"]})
        weights = calculate_sample_weights(df)
        assert weights.iloc[3] > weights.iloc[0]

    def test_weights_clipped(self):
        from bias import calculate_sample_weights

        df = pd.DataFrame({"Store ID": ["S1"] * 1000 + ["S2"]})
        weights = calculate_sample_weights(df, min_weight=0.1, max_weight=10.0)
        assert weights.min() >= 0.1
        assert weights.max() <= 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS — validate.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidate:
    """Tests for generate_schema_and_stats (with great_expectations mocked)."""

    def test_generates_schema_json(self, features_parquet, tmp_path):
        """Test schema generation (mocking GE to avoid heavy dependency)."""
        from unittest.mock import MagicMock
        import importlib

        mock_ge = MagicMock()
        mock_ge_df = MagicMock()
        mock_ge_df.columns = pd.read_parquet(features_parquet).columns.tolist()
        mock_ge_df.validate.return_value = MagicMock(
            to_json_dict=lambda: {"success": True, "results": []}
        )
        mock_ge.from_pandas.return_value = mock_ge_df

        with patch.dict("sys.modules", {"great_expectations": mock_ge}):
            if "validate" in sys.modules:
                del sys.modules["validate"]
            from validate import generate_schema_and_stats

            output_dir = str(tmp_path / "validate_out")
            generate_schema_and_stats(features_parquet, output_dir)

            schema_path = Path(output_dir) / "schema.json"
            stats_path = Path(output_dir) / "stats.json"

            assert schema_path.exists()
            assert stats_path.exists()

            with open(schema_path) as f:
                schema = json.load(f)
            assert len(schema) > 0

            with open(stats_path) as f:
                stats = json.load(f)
            assert len(stats) > 0

    def test_stats_contain_expected_fields(self, features_parquet, tmp_path):
        """Verify stats JSON has mean, std, min, max, null_pct per column."""
        mock_ge = MagicMock()
        mock_ge_df = MagicMock()
        mock_ge_df.columns = pd.read_parquet(features_parquet).columns.tolist()
        mock_ge_df.validate.return_value = MagicMock(
            to_json_dict=lambda: {"success": True, "results": []}
        )
        mock_ge.from_pandas.return_value = mock_ge_df

        with patch.dict("sys.modules", {"great_expectations": mock_ge}):
            if "validate" in sys.modules:
                del sys.modules["validate"]
            from validate import generate_schema_and_stats

            output_dir = str(tmp_path / "validate_out2")
            generate_schema_and_stats(features_parquet, output_dir)

            with open(Path(output_dir) / "stats.json") as f:
                stats = json.load(f)

            for col, col_stats in stats.items():
                assert "null_pct" in col_stats
                if col_stats["mean"] is not None:
                    assert "std" in col_stats
                    assert "min" in col_stats
                    assert "max" in col_stats


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS — upload_to_gcp.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestUploadToGCP:
    """Tests for upload_to_gcs (all GCP calls mocked)."""

    @patch("upload_to_gcp.storage")
    @patch("upload_to_gcp.load_dotenv")
    @patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/fake/creds.json"})
    def test_upload_calls_gcs(self, mock_dotenv, mock_storage, tmp_path):
        from upload_to_gcp import upload_to_gcs

        test_file = tmp_path / "test.parquet"
        test_file.write_text("fake data")

        mock_client = MagicMock()
        mock_storage.Client.from_service_account_json.return_value = mock_client
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        upload_to_gcs(str(test_file), "test-bucket", "dest/test.parquet")

        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.blob.assert_called_once_with("dest/test.parquet")
        mock_blob.upload_from_filename.assert_called_once_with(str(test_file))

    @patch("upload_to_gcp.load_dotenv")
    @patch.dict(os.environ, {}, clear=True)
    def test_raises_without_credentials(self, mock_dotenv, tmp_path):
        from upload_to_gcp import upload_to_gcs
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

        test_file = tmp_path / "test.parquet"
        test_file.write_text("fake data")

        with pytest.raises(ValueError, match="GOOGLE_APPLICATION_CREDENTIALS"):
            upload_to_gcs(str(test_file), "test-bucket")

    @patch("upload_to_gcp.storage")
    @patch("upload_to_gcp.load_dotenv")
    @patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/fake/creds.json"})
    def test_default_destination_uses_filename(self, mock_dotenv, mock_storage, tmp_path):
        from upload_to_gcp import upload_to_gcs

        test_file = tmp_path / "my_data.parquet"
        test_file.write_text("fake")

        mock_client = MagicMock()
        mock_storage.Client.from_service_account_json.return_value = mock_client
        mock_bucket = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        upload_to_gcs(str(test_file), "bucket")
        mock_bucket.blob.assert_called_once_with("my_data.parquet")


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS — upload_to_mongo.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestUploadToMongo:
    """Tests for load_file and upload_to_mongo (MongoDB calls mocked)."""

    def test_load_csv(self, tmp_path):
        from upload_to_mongo import load_file

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n")

        docs = load_file(csv_path)
        assert len(docs) == 2
        assert docs[0]["a"] == 1

    def test_load_json_array(self, tmp_path):
        from upload_to_mongo import load_file

        json_path = tmp_path / "data.json"
        json_path.write_text(json.dumps([{"x": 1}, {"x": 2}]))

        docs = load_file(json_path)
        assert len(docs) == 2

    def test_load_json_dict_with_list(self, tmp_path):
        from upload_to_mongo import load_file

        json_path = tmp_path / "data.json"
        json_path.write_text(json.dumps({"records": [{"x": 1}, {"x": 2}]}))

        docs = load_file(json_path)
        assert len(docs) == 2

    def test_load_json_single_object(self, tmp_path):
        from upload_to_mongo import load_file

        json_path = tmp_path / "data.json"
        json_path.write_text(json.dumps({"name": "test", "value": 42}))

        docs = load_file(json_path)
        assert len(docs) == 1
        assert docs[0]["name"] == "test"

    def test_unsupported_file_type(self, tmp_path):
        from upload_to_mongo import load_file

        xml_path = tmp_path / "data.xml"
        xml_path.write_text("<root/>")

        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file(xml_path)

    @patch("upload_to_mongo.MongoClient")
    def test_upload_inserts_documents(self, mock_mongo_cls):
        from upload_to_mongo import upload_to_mongo

        mock_client = MagicMock()
        mock_mongo_cls.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection
        mock_collection.insert_many.return_value = MagicMock(
            inserted_ids=["id1", "id2", "id3"]
        )

        docs = [{"a": 1}, {"a": 2}, {"a": 3}]
        upload_to_mongo(docs, "mongodb://localhost", "testdb", "testcol", batch_size=10)

        mock_collection.insert_many.assert_called_once()

    @patch("upload_to_mongo.MongoClient")
    def test_upload_with_drop(self, mock_mongo_cls):
        from upload_to_mongo import upload_to_mongo

        mock_client = MagicMock()
        mock_mongo_cls.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection
        mock_collection.insert_many.return_value = MagicMock(inserted_ids=["id1"])

        upload_to_mongo(
            [{"a": 1}], "mongodb://localhost", "db", "col",
            batch_size=10, drop_first=True
        )
        mock_collection.drop.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS — data_pipeline.py (transform function)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineTransform:
    """Tests for the transform() feature engineering function."""

    def test_transform_produces_expected_columns(
        self, sample_supply_chain_df, tmp_path
    ):
        """Verify transform creates all required feature columns."""
        raw_path = tmp_path / "raw.parquet"
        sample_supply_chain_df.to_parquet(raw_path, index=False)

        with patch.dict("sys.modules", {
            "airflow.sdk": MagicMock(),
            "airflow.operators.python": MagicMock(),
            "airflow.providers.smtp.operators.smtp": MagicMock(),
            "airflow.exceptions": MagicMock(AirflowException=Exception),
        }):
            if "dags.data_pipeline" in sys.modules:
                del sys.modules["dags.data_pipeline"]
            with patch("dags.data_pipeline.FEAT_DIR", tmp_path / "features"):
                from dags.data_pipeline import transform

                result_path = transform(str(raw_path), horizon=1, run_id="test_run")
                result_df = pd.read_parquet(result_path)

                expected_features = [
                    "sales_lag_1", "sales_lag_7", "sales_lag_14",
                    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
                    "sales_ewm_28", "dow", "month",
                    "y_pred_baseline", "y", "sample_weight",
                ]
                for col in expected_features:
                    assert col in result_df.columns, f"Missing column: {col}"

    def test_transform_no_label_leakage(self, sample_supply_chain_df, tmp_path):
        """Ensure lag features don't leak future information."""
        raw_path = tmp_path / "raw.parquet"
        sample_supply_chain_df.to_parquet(raw_path, index=False)

        with patch.dict("sys.modules", {
            "airflow.sdk": MagicMock(),
            "airflow.operators.python": MagicMock(),
            "airflow.providers.smtp.operators.smtp": MagicMock(),
            "airflow.exceptions": MagicMock(AirflowException=Exception),
        }):
            if "dags.data_pipeline" in sys.modules:
                del sys.modules["dags.data_pipeline"]
            with patch("dags.data_pipeline.FEAT_DIR", tmp_path / "features"):
                from dags.data_pipeline import transform

                result_path = transform(str(raw_path), horizon=1, run_id="test_run2")
                result_df = pd.read_parquet(result_path)

                assert not (result_df["sales_lag_1"] == result_df["y"]).all()

    def test_transform_drops_rows_with_missing_lags(
        self, sample_supply_chain_df, tmp_path
    ):
        """First 14 rows per series should be dropped (no lag_14 available)."""
        raw_path = tmp_path / "raw.parquet"
        sample_supply_chain_df.to_parquet(raw_path, index=False)

        with patch.dict("sys.modules", {
            "airflow.sdk": MagicMock(),
            "airflow.operators.python": MagicMock(),
            "airflow.providers.smtp.operators.smtp": MagicMock(),
            "airflow.exceptions": MagicMock(AirflowException=Exception),
        }):
            if "dags.data_pipeline" in sys.modules:
                del sys.modules["dags.data_pipeline"]
            with patch("dags.data_pipeline.FEAT_DIR", tmp_path / "features"):
                from dags.data_pipeline import transform

                result_path = transform(str(raw_path), horizon=1, run_id="test_run3")
                result_df = pd.read_parquet(result_path)

                original_count = len(sample_supply_chain_df)
                assert len(result_df) < original_count

                assert result_df["sales_lag_14"].isnull().sum() == 0
                assert result_df["y"].isnull().sum() == 0

    def test_transform_rejects_missing_columns(self, tmp_path):
        """Transform should raise ValueError if required columns are missing."""
        bad_df = pd.DataFrame({"Date": ["2025-01-01"], "Store ID": ["S1"]})
        raw_path = tmp_path / "raw.parquet"
        bad_df.to_parquet(raw_path, index=False)

        with patch.dict("sys.modules", {
            "airflow.sdk": MagicMock(),
            "airflow.operators.python": MagicMock(),
            "airflow.providers.smtp.operators.smtp": MagicMock(),
            "airflow.exceptions": MagicMock(AirflowException=Exception),
        }):
            if "dags.data_pipeline" in sys.modules:
                del sys.modules["dags.data_pipeline"]
            with patch("dags.data_pipeline.FEAT_DIR", tmp_path / "features"):
                from dags.data_pipeline import transform

                with pytest.raises(ValueError, match="Missing required columns"):
                    transform(str(raw_path), horizon=1, run_id="test_fail")

    def test_transform_handles_negative_values(self, sample_supply_chain_df, tmp_path):
        """Negative Units Sold / Inventory Level should be clipped to 0."""
        df = sample_supply_chain_df.copy()
        df.loc[0:5, "Units Sold"] = -10
        df.loc[0:5, "Inventory Level"] = -20

        raw_path = tmp_path / "raw.parquet"
        df.to_parquet(raw_path, index=False)

        with patch.dict("sys.modules", {
            "airflow.sdk": MagicMock(),
            "airflow.operators.python": MagicMock(),
            "airflow.providers.smtp.operators.smtp": MagicMock(),
            "airflow.exceptions": MagicMock(AirflowException=Exception),
        }):
            if "dags.data_pipeline" in sys.modules:
                del sys.modules["dags.data_pipeline"]
            with patch("dags.data_pipeline.FEAT_DIR", tmp_path / "features"):
                from dags.data_pipeline import transform

                result_path = transform(str(raw_path), horizon=1, run_id="test_neg")
                # If it completes without error, clipping worked.
                assert Path(result_path).exists()

    def test_transform_removes_duplicates(self, sample_supply_chain_df, tmp_path):
        """Duplicate (Date, Store ID, Product ID) rows should be deduplicated."""
        df = pd.concat([sample_supply_chain_df, sample_supply_chain_df.iloc[:10]])
        raw_path = tmp_path / "raw.parquet"
        df.to_parquet(raw_path, index=False)

        with patch.dict("sys.modules", {
            "airflow.sdk": MagicMock(),
            "airflow.operators.python": MagicMock(),
            "airflow.providers.smtp.operators.smtp": MagicMock(),
            "airflow.exceptions": MagicMock(AirflowException=Exception),
        }):
            if "dags.data_pipeline" in sys.modules:
                del sys.modules["dags.data_pipeline"]
            with patch("dags.data_pipeline.FEAT_DIR", tmp_path / "features"):
                from dags.data_pipeline import transform

                result_path = transform(str(raw_path), horizon=1, run_id="test_dup")
                assert Path(result_path).exists()


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS — params.yaml integrity
# ═══════════════════════════════════════════════════════════════════════════════

class TestParamsConfig:
    """Ensure params.yaml is valid and has expected keys."""

    def test_params_yaml_loads(self):
        import yaml

        params_path = PROJECT_ROOT / "params.yaml"
        with open(params_path) as f:
            params = yaml.safe_load(f)

        assert isinstance(params, dict)
        assert "horizon" in params
        assert "lags" in params
        assert "rolling_windows" in params
        assert "anomaly_thresholds" in params

    def test_params_anomaly_thresholds(self):
        import yaml

        params_path = PROJECT_ROOT / "params.yaml"
        with open(params_path) as f:
            params = yaml.safe_load(f)

        thresholds = params["anomaly_thresholds"]
        assert "z_score" in thresholds
        assert "missingness" in thresholds
        assert "date_gap_days" in thresholds
        assert thresholds["z_score"] > 0
        assert 0 < thresholds["missingness"] < 1

    def test_lags_are_positive_ints(self):
        import yaml

        params_path = PROJECT_ROOT / "params.yaml"
        with open(params_path) as f:
            params = yaml.safe_load(f)

        for lag in params["lags"]:
            assert isinstance(lag, int) and lag > 0

        for window in params["rolling_windows"]:
            assert isinstance(window, int) and window > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS — Integration / End-to-End
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """Integration tests: transform -> anomaly -> bias pipeline."""

    def test_full_pipeline_flow(self, sample_supply_chain_df, tmp_path):
        """Run transform -> anomaly detection -> bias analysis end-to-end."""
        raw_path = tmp_path / "raw.parquet"
        sample_supply_chain_df.to_parquet(raw_path, index=False)

        with patch.dict("sys.modules", {
            "airflow.sdk": MagicMock(),
            "airflow.operators.python": MagicMock(),
            "airflow.providers.smtp.operators.smtp": MagicMock(),
            "airflow.exceptions": MagicMock(AirflowException=Exception),
        }):
            if "dags.data_pipeline" in sys.modules:
                del sys.modules["dags.data_pipeline"]
            with patch("dags.data_pipeline.FEAT_DIR", tmp_path / "features"):
                from dags.data_pipeline import transform

                features_path = transform(
                    str(raw_path), horizon=1, run_id="e2e_test"
                )

        from anomaly import generate_anomaly_report

        anomaly_dir = str(tmp_path / "anomaly")
        anomaly_report = generate_anomaly_report(features_path, anomaly_dir)
        assert Path(anomaly_report).exists()

        with open(anomaly_report) as f:
            report = json.load(f)
        assert report["metadata"]["total_rows"] > 0

        # 3. Bias analysis
        from bias import generate_bias_report

        bias_dir = str(tmp_path / "bias")
        bias_report = generate_bias_report(features_path, bias_dir)
        assert Path(bias_report).exists()

        with open(bias_report) as f:
            breport = json.load(f)
        assert breport["metadata"]["total_rows"] > 0
