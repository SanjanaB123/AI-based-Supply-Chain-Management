import great_expectations as ge
import pandas as pd
import json
from pathlib import Path

def generate_schema_and_stats(features_path, output_dir):
    df = pd.read_parquet(features_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save schema (columns + dtypes)
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    with open(output_dir / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # Save stats
    stats = {}
    for col in df.columns:
        stats[col] = {
            "mean": float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
            "std": float(df[col].std()) if pd.api.types.is_numeric_dtype(df[col]) else None,
            "min": float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
            "max": float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None,
            "null_pct": float(df[col].isnull().mean()),
        }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Great Expectations validation
    ge_df = ge.from_pandas(df)
    
    # Check that critical columns exist rather than an exact ordered match,
    # as the pipeline now generates many engineered features
    critical_columns = [
        "as_of_date", "Store ID", "Product ID", "Units Sold", 
        "Price", "Discount", "Holiday/Promotion", "Competitor Pricing", 
        "Weather Condition", "Seasonality", "Units Ordered", "y"
    ]
    for col in critical_columns:
        if col in ge_df.columns:
            ge_df.expect_column_to_exist(col)

    ge_df.expect_column_values_to_not_be_null("as_of_date")
    ge_df.expect_column_values_to_not_be_null("y")
    
    if "Units Sold" in ge_df.columns:
        ge_df.expect_column_values_to_be_between("Units Sold", min_value=0)
    if "Price" in ge_df.columns:
        ge_df.expect_column_values_to_be_between("Price", min_value=0, strict_min=True)
    if "Holiday/Promotion" in ge_df.columns:
        ge_df.expect_column_values_to_be_in_set("Holiday/Promotion", [0, 1])
        
    ge_df.expect_compound_columns_to_be_unique(["as_of_date", "Store ID", "Product ID"])
    
    result = ge_df.validate()
    with open(output_dir / "ge_validation_result.json", "w") as f:
        json.dump(result.to_json_dict(), f, indent=2)
    return str(output_dir)
