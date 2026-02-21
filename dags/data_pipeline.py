# Airflow imports are optional for running tests locally
try:
    from airflow.decorators import dag
    from airflow.operators.python import PythonOperator
    _AIRFLOW_AVAILABLE = True
except Exception:
    _AIRFLOW_AVAILABLE = False

import pandas as pd
from datetime import datetime

# data extraction
def extract(file_path):
    return pd.read_csv(file_path)

# data transformation
def transform(df, horizon=1, pipeline_version="1.0"):
    # Step 1 — Sort properly
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Store ID', 'Product ID', 'Date']).reset_index(drop=True)
    
    # Step 2 & 3 — Create lag and rolling features
    # Group by Store and Product so past data is only from the same item
    grouped_sales = df.groupby(['Store ID', 'Product ID'])['Units Sold']
    
    # Lag Features (ONLY past data)
    df['sales_lag_1'] = grouped_sales.shift(1)
    df['sales_lag_7'] = grouped_sales.shift(7)
    df['sales_lag_14'] = grouped_sales.shift(14)
    
    # Rolling Features (ONLY past data)
    # The trick: Compute rolling mean, then shift by 1 so time t only sees t-1 and prior
    df['sales_roll_mean_7'] = grouped_sales.transform(lambda x: x.rolling(7, min_periods=1).mean().shift(1))
    df['sales_roll_mean_14'] = grouped_sales.transform(lambda x: x.rolling(14, min_periods=1).mean().shift(1))
    df['sales_roll_mean_28'] = grouped_sales.transform(lambda x: x.rolling(28, min_periods=1).mean().shift(1))

    # Step 4 — Calendar features
    df['dow'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    
    # Step 5 & 6 — Label Creation
    # y = Sales.shift(-1) -> Predict Support at t+1 based on t
    df['y'] = grouped_sales.shift(-horizon)
    
    # Step 7 — Drop invalid rows
    # Drop first 14 days (missing lag_14) and last day (missing label y) per series
    df = df.dropna(subset=['sales_lag_14', 'y'])
    
    # Step 8 — MLOps Metadata & Final column selection
    df['as_of_date'] = df['Date'] # Date at time t
    df['series_id'] = df['Store ID'] + "_" + df['Product ID']
    df['horizon'] = horizon
    df['pipeline_version'] = pipeline_version
    df['created_at'] = datetime.now().isoformat()
    
    # Keep requested features + known covariates (already in the dataframe)
    features = [
        'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 
        'sales_roll_mean_7', 'sales_roll_mean_14', 'sales_roll_mean_28',
        'dow', 'month', 
        'Price', 'Discount', 'Holiday/Promotion', 'Competitor Pricing', 'Weather Condition', 'Seasonality', 'Inventory Level', 'Units Ordered'
    ]
    meta = ['as_of_date', 'series_id', 'horizon', 'pipeline_version', 'created_at']
    labels = ['y']
    identifiers = ['Store ID', 'Product ID']
    
    # Select the final model-ready columns
    final_cols = meta + identifiers + features + labels
    return df[final_cols].reset_index(drop=True)

# data loading
def load(df):
    df.to_csv(f"data/processed/processed_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", index=False)

if _AIRFLOW_AVAILABLE:
    @dag()
    def data_pipeline():
        extract_task = PythonOperator(
            task_id='extract',
            python_callable=extract,
            op_kwargs={'file_path': 'data/retail_store_inventory.csv'},
        )
        transform_task = PythonOperator(
            task_id='transform',
            python_callable=transform,
            op_kwargs={'df': extract_task.output},
        )
        load_task = PythonOperator(
            task_id='load',
            python_callable=load,
            op_kwargs={'df': transform_task.output},
        )

        extract_task >> transform_task >> load_task