from mcp.server.fastmcp import FastMCP
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import io
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from datetime import datetime

# Load environment variables from root .env (two levels up from mcp/)
_root_env = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(_root_env)

# Resolve GOOGLE_APPLICATION_CREDENTIALS to an absolute path if it's relative
_gcp_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if _gcp_creds and not os.path.isabs(_gcp_creds):
    _abs_creds = os.path.join(os.path.dirname(__file__), '..', '..', _gcp_creds)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(_abs_creds)

# Initialize FastMCP server
PORT = int(os.getenv("PORT", 8000))
mcp = FastMCP("inventory_server", host="0.0.0.0", port=PORT)

# MongoDB connection setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["inventory_forecasting"]
collection = db["inventory_snapshot"]

# --- MLflow & GCS config ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")

XGBOOST_MODEL_NAME = "xgboost-supply-chain"
PROPHET_MODEL_NAME = "prophet-supply-chain"

# Numeric columns to scale for XGBoost (StandardScaler applied to these 15 cols)
# These must exactly match the feature names the scaler was fit with
NUMERIC_COLS = [
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
    "sales_roll_std_7", "sales_ewm_28", "demand_forecast_lag1",
    "price_vs_competitor", "effective_price",
    "Inventory Level", "lead_time_demand", "y_pred_baseline"
]

@mcp.tool()
def summarize_inventory() -> dict:
    """
    Summarize overall inventory information across all stores, categorized by products and stores.
    """
    try:
        pipeline = [
            {
                "$facet": {
                    "product_summary": [
                        {"$group": {
                            "_id": "$Product ID",
                            "total_stock": {"$sum": "$Current Stock"},
                            "total_sold": {"$sum": "$Total Units Sold"},
                            "total_received": {"$sum": "$Total Units Received"}
                        }}
                    ],
                    "store_summary": [
                        {"$group": {
                            "_id": "$Store ID",
                            "total_stock": {"$sum": "$Current Stock"},
                            "total_sold": {"$sum": "$Total Units Sold"},
                            "total_received": {"$sum": "$Total Units Received"}
                        }}
                    ],
                    "overall_stats": [
                        {"$group": {
                            "_id": None,
                            "total_stock": {"$sum": "$Current Stock"},
                            "total_sold": {"$sum": "$Total Units Sold"},
                            "total_received": {"$sum": "$Total Units Received"},
                            "unique_products": {"$addToSet": "$Product ID"},
                            "unique_stores": {"$addToSet": "$Store ID"}
                        }}
                    ]
                }
            }
        ]
        
        result = list(collection.aggregate(pipeline))
        if result:
            data = result[0]
            overall = data.get("overall_stats", [{}])[0]
            
            return {
                "status": "success",
                "overall": {
                    "total_products": len(overall.get("unique_products", [])),
                    "total_stores": len(overall.get("unique_stores", [])),
                    "total_stock": overall.get("total_stock", 0),
                    "total_sold": overall.get("total_sold", 0),
                    "total_received": overall.get("total_received", 0)
                },
                "product_wise": data.get("product_summary", []),
                "store_wise": data.get("store_summary", [])
            }
            
        return {"status": "error", "message": "No inventory data found to summarize."}
    except Exception as e:
        return {"status": "error", "message": f"Error summarizing inventory: {str(e)}"}

@mcp.tool()
def get_product_at_store(product_id: str, store_id: str) -> dict:
    """
    Get detailed information for a specific product at a specific store.
    
    Args:
        product_id: The unique identifier for the product.
        store_id: The unique identifier for the store.
    """
    try:
        item = collection.find_one({"Product ID": product_id, "Store ID": store_id})
        
        if not item:
            return {"status": "error", "message": f"No inventory found for Product '{product_id}' at Store '{store_id}'"}
            
        if "_id" in item:
            item["_id"] = str(item["_id"])
            
        return {"status": "success", "data": item}
    except Exception as e:
        return {"status": "error", "message": f"Error connecting to or querying MongoDB: {str(e)}"}

@mcp.tool()
def check_restocking_needs(store_id: str, threshold: int = 10) -> dict:
    """
    Check for products that require restocking at a particular store.
    
    Args:
        store_id: The unique identifier for the store.
        threshold: The stock level below which to trigger a restock alert (default: 10).
    """
    try:
        items_to_restock = list(collection.find(
            {"Store ID": store_id, "Current Stock": {"$lt": threshold}}
        ))
        
        if not items_to_restock:
            return {
                "status": "success",
                "message": f"No products require restocking at Store '{store_id}' (threshold < {threshold}).",
                "items": []
            }
            
        # Convert _id for all returned items
        for item in items_to_restock:
            if "_id" in item:
                item["_id"] = str(item["_id"])
                
        return {
            "status": "success",
            "store_id": store_id,
            "threshold": threshold,
            "items": items_to_restock
        }
    except Exception as e:
        return {"status": "error", "message": f"Error checking restock levels: {str(e)}"}

@mcp.tool()
def predict_demand(
    store_id: str,
    product_id: str,
    target_date: str,
    sales_lag_1: float,
    sales_lag_7: float,
    sales_lag_14: float,
    sales_lag_28: float,
    sales_roll_mean_7: float,
    sales_roll_mean_14: float,
    sales_roll_mean_28: float,
    sales_roll_std_7: float,
    sales_ewm_28: float,
    demand_forecast_lag1: float,
    price_vs_competitor: float,
    effective_price: float,
    holiday_promotion: int,
    discount: float,
    discount_x_holiday: float,
    inventory_level: int,
    stockout_flag: int,
    lead_time_demand: float,
    lead_time_days: int,
    reorder_event: int,
    category_enc: int,
    region_enc: int,
    seasonality_enc: int,
    y_pred_baseline: float
) -> dict:
    """
    Predict demand for a product at a store using the Production model from MLflow.
    Automatically detects whether XGBoost or Prophet is in Production and applies
    the correct preprocessing pipeline for each.

    Args:
        store_id: Store identifier (e.g. "S001")
        product_id: Product identifier (e.g. "P0001")
        target_date: Date for prediction in YYYY-MM-DD format
        sales_lag_1: Units sold 1 day ago
        sales_lag_7: Units sold 7 days ago
        sales_lag_14: Units sold 14 days ago
        sales_lag_28: Units sold 28 days ago
        sales_roll_mean_7: 7-day rolling mean of sales
        sales_roll_mean_14: 14-day rolling mean of sales
        sales_roll_mean_28: 28-day rolling mean of sales
        sales_roll_std_7: 7-day rolling std of sales
        sales_ewm_28: 28-day exponentially weighted mean of sales
        demand_forecast_lag1: Previous day's demand forecast
        price_vs_competitor: Ratio of product price to competitor price
        effective_price: Actual selling price after discounts
        holiday_promotion: 1 if holiday/promotion, else 0
        discount: Discount rate (0.0 to 1.0)
        discount_x_holiday: Interaction term (discount * holiday_promotion)
        inventory_level: Current stock level
        stockout_flag: 1 if stockout occurred, else 0
        lead_time_demand: Expected demand during lead time
        lead_time_days: Number of days for supplier lead time
        reorder_event: 1 if a reorder was placed, else 0
        category_enc: Integer-encoded product category
        region_enc: Integer-encoded store region
        seasonality_enc: Integer-encoded seasonality pattern
        y_pred_baseline: Baseline prediction (typically equals sales_lag_1)
    """
    try:
        if not MLFLOW_TRACKING_URI:
            return {"status": "error", "message": "MLFLOW_TRACKING_URI is not set in environment."}

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow_client = mlflow.tracking.MlflowClient()

        # 1. Detect which model has the @champion alias (XGBoost takes priority)
        model_name = None
        prod_version = None
        for name in [XGBOOST_MODEL_NAME, PROPHET_MODEL_NAME]:
            try:
                mv = mlflow_client.get_model_version_by_alias(name, "champion")
                model_name = name
                prod_version = mv.version
                break
            except Exception:
                continue

        if not model_name:
            return {"status": "error", "message": "No model found with @champion alias in MLflow registry."}

        # 2. Derive time features from target_date
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        dow = dt.weekday()       # 0=Monday, 6=Sunday
        month = dt.month
        is_weekend = int(dow >= 5)

        is_xgboost = model_name == XGBOOST_MODEL_NAME

        if is_xgboost:
            # 3a. Load scaler.pkl and series_mapping.json from GCS
            if not GCS_BUCKET:
                return {"status": "error", "message": "GCS_BUCKET is not set in environment."}

            from google.cloud import storage as gcs
            gcs_client = gcs.Client()
            bucket = gcs_client.bucket(GCS_BUCKET)

            mapping_blob = bucket.blob("preprocessing-artifacts/series_mapping.json")
            series_mapping = json.loads(mapping_blob.download_as_text())

            series_id = f"{store_id}_{product_id}"
            series_enc = series_mapping.get(series_id)
            if series_enc is None:
                return {"status": "error", "message": f"series_id '{series_id}' not found in series_mapping.json"}

            scaler_blob = bucket.blob("preprocessing-artifacts/scaler.pkl")
            scaler = joblib.load(io.BytesIO(scaler_blob.download_as_bytes()))

            # 4a. Build 28-column feature DataFrame in the exact column order the model expects
            MODEL_FEATURE_ORDER = [
                "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
                "sales_roll_mean_7", "sales_roll_mean_14", "sales_roll_mean_28",
                "sales_roll_std_7", "sales_ewm_28", "demand_forecast_lag1",
                "price_vs_competitor", "effective_price",
                "Holiday/Promotion", "Discount", "discount_x_holiday",
                "dow", "month", "is_weekend",
                "Inventory Level", "stockout_flag", "lead_time_demand",
                "Lead Time Days", "reorder_event",
                "Category_enc", "Region_enc", "Seasonality_enc",
                "y_pred_baseline", "series_enc"
            ]
            row = {
                "sales_lag_1": sales_lag_1, "sales_lag_7": sales_lag_7,
                "sales_lag_14": sales_lag_14, "sales_lag_28": sales_lag_28,
                "sales_roll_mean_7": sales_roll_mean_7, "sales_roll_mean_14": sales_roll_mean_14,
                "sales_roll_mean_28": sales_roll_mean_28, "sales_roll_std_7": sales_roll_std_7,
                "sales_ewm_28": sales_ewm_28, "demand_forecast_lag1": demand_forecast_lag1,
                "price_vs_competitor": price_vs_competitor, "effective_price": effective_price,
                "Holiday/Promotion": holiday_promotion, "Discount": discount,
                "discount_x_holiday": discount_x_holiday,
                "dow": dow, "month": month, "is_weekend": is_weekend,
                "Inventory Level": inventory_level, "stockout_flag": stockout_flag,
                "lead_time_demand": lead_time_demand, "Lead Time Days": lead_time_days,
                "reorder_event": reorder_event,
                "Category_enc": category_enc, "Region_enc": region_enc,
                "Seasonality_enc": seasonality_enc,
                "y_pred_baseline": y_pred_baseline, "series_enc": series_enc
            }
            input_df = pd.DataFrame([row])[MODEL_FEATURE_ORDER]

            # 5a. Apply StandardScaler to numeric columns
            cols_to_scale = [c for c in NUMERIC_COLS if c in input_df.columns]
            input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

            # 6a. Load model and predict
            model = mlflow.pyfunc.load_model(f"models:/{XGBOOST_MODEL_NAME}@champion")
            prediction = float(max(0.0, model.predict(input_df)[0]))

            return {
                "status": "success",
                "store_id": store_id,
                "product_id": product_id,
                "target_date": target_date,
                "model_used": XGBOOST_MODEL_NAME,
                "model_version": prod_version,
                "predicted_demand": round(prediction, 2)
            }

        else:
            # 3b. Prophet path — load per-series Production model
            model = mlflow.pyfunc.load_model(f"models:/{PROPHET_MODEL_NAME}@champion")

            # 4b. Build Prophet input: ds + 9 regressors (lag/rolling fields are ignored)
            future_df = pd.DataFrame({
                "ds": [pd.Timestamp(target_date)],
                "Holiday/Promotion": [holiday_promotion],
                "Discount": [discount],
                "discount_x_holiday": [discount_x_holiday],
                "Inventory Level": [inventory_level],
                "stockout_flag": [stockout_flag],
                "lead_time_demand": [lead_time_demand],
                "Lead Time Days": [lead_time_days],
                "reorder_event": [reorder_event],
                "price_vs_competitor": [price_vs_competitor]
            })

            # 5b. Predict and clip to non-negative
            forecast = model.predict(future_df)
            yhat = float(max(0.0, forecast["yhat"].iloc[0]))

            return {
                "status": "success",
                "store_id": store_id,
                "product_id": product_id,
                "target_date": target_date,
                "model_used": PROPHET_MODEL_NAME,
                "model_version": prod_version,
                "predicted_demand": round(yhat, 2)
            }

    except Exception as e:
        import traceback
        return {"status": "error", "message": f"Prediction failed: {str(e)}", "traceback": traceback.format_exc()}


@mcp.custom_route("/health", methods=["GET"])
async def health(request):
    """Health check for Cloud Run / load balancers."""
    from starlette.responses import JSONResponse
    try:
        client.admin.command("ping")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    return JSONResponse({"status": "ok", "database": db_status})


if __name__ == "__main__":
    mcp.run(transport='streamable-http')
