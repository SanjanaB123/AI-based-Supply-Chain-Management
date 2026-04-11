"""
Synthetic data generator for Stratos inventory system.

Generates:
  1. inventory_snapshot  (5 stores x 20 products = 100 rows)
  2. retail_snapshot     (30 days x 100 combos  = 3000 rows)
  3. vendors             (10 vendor records, 1+ per category)

Usage:
  python generate_data.py --mode mongo          # insert into MongoDB
  python generate_data.py --mode csv            # write CSV files
  python generate_data.py --mode both           # do both
  python generate_data.py --mode both --seed 42 # reproducible
"""

import argparse
import csv
import os
import random
import uuid
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# ── Constants ─────────────────────────────────────────────────────────────────

STORES = {
    "S001": "Northeast",
    "S002": "Southeast",
    "S003": "Midwest",
    "S004": "West",
    "S005": "Southwest",
}

PRODUCT_CATEGORIES = {
    "P0001": "Snacks",    "P0002": "Snacks",
    "P0003": "Beverages", "P0004": "Beverages",
    "P0005": "Dairy",     "P0006": "Dairy",
    "P0007": "Beverages", "P0008": "Frozen Foods",
    "P0009": "Personal Care", "P0010": "Snacks",
    "P0011": "Household", "P0012": "Household",
    "P0013": "Bakery",    "P0014": "Bakery",
    "P0015": "Produce",   "P0016": "Produce",
    "P0017": "Meat",      "P0018": "Meat",
    "P0019": "Canned Goods", "P0020": "Canned Goods",
}

CATEGORIES = list(set(PRODUCT_CATEGORIES.values()))

LEAD_TIME_RANGES = {
    "Snacks": (3, 7),
    "Beverages": (3, 7),
    "Dairy": (2, 5),
    "Frozen Foods": (4, 10),
    "Personal Care": (5, 14),
    "Household": (5, 14),
    "Bakery": (1, 3),
    "Produce": (1, 4),
    "Meat": (2, 5),
    "Canned Goods": (7, 21),
}

BASE_PRICES = {
    "Snacks": 4.99,
    "Beverages": 2.49,
    "Dairy": 3.99,
    "Frozen Foods": 6.99,
    "Personal Care": 8.49,
    "Household": 12.99,
    "Bakery": 3.49,
    "Produce": 2.99,
    "Meat": 9.99,
    "Canned Goods": 1.99,
}

WEATHER_CONDITIONS = ["Sunny", "Rainy", "Cloudy", "Snowy", "Windy"]

VENDOR_DATA = [
    {"name": "FreshCo Distributors",      "category": "Produce",       "region": "Northeast"},
    {"name": "Pacific Beverages Inc.",     "category": "Beverages",     "region": "West"},
    {"name": "Heartland Snacks Co.",       "category": "Snacks",        "region": "Midwest"},
    {"name": "Southern Dairy Farms",       "category": "Dairy",         "region": "Southeast"},
    {"name": "Arctic Frozen Supply",       "category": "Frozen Foods",  "region": "Northeast"},
    {"name": "CleanHome Products LLC",     "category": "Household",     "region": "Southwest"},
    {"name": "GlowUp Personal Care",      "category": "Personal Care", "region": "West"},
    {"name": "Golden Grain Bakery",        "category": "Bakery",        "region": "Midwest"},
    {"name": "Prime Cuts Meats",           "category": "Meat",          "region": "Southeast"},
    {"name": "Pantry Staples Wholesale",   "category": "Canned Goods",  "region": "Southwest"},
]


# ── Generators ────────────────────────────────────────────────────────────────

def generate_inventory_snapshot() -> list[dict]:
    """100 rows: 5 stores x 20 products."""
    rows = []
    for store_id, region in STORES.items():
        for product_id, category in PRODUCT_CATEGORIES.items():
            lt_lo, lt_hi = LEAD_TIME_RANGES[category]
            lead_time = random.randint(lt_lo, lt_hi)

            # ~30 % of items are deliberately low stock
            if random.random() < 0.30:
                current_stock = random.randint(5, 30)
            else:
                current_stock = random.randint(100, 2000)

            total_sold = random.randint(500, 15000)
            shrinkage = random.randint(0, 200)
            total_received = total_sold + current_stock + shrinkage

            rows.append({
                "Store ID": store_id,
                "Product ID": product_id,
                "Category": category,
                "Region": region,
                "Current Stock": current_stock,
                "Total Units Sold": total_sold,
                "Total Units Received": total_received,
                "Lead Time Days": lead_time,
            })
    return rows


def generate_retail_snapshot(inventory: list[dict], days: int = 30) -> list[dict]:
    """Daily rows for every store-product combo over `days` days."""
    rows = []
    end_date = datetime(2026, 4, 11)
    start_date = end_date - timedelta(days=days - 1)

    inv_lookup = {(r["Store ID"], r["Product ID"]): r for r in inventory}

    for day_offset in range(days):
        date = start_date + timedelta(days=day_offset)
        date_str = date.strftime("%Y-%m-%d")
        is_weekend = date.weekday() >= 5
        month = date.month

        # Seasonality label
        if month in (12, 1, 2):
            seasonality = "Winter"
        elif month in (3, 4, 5):
            seasonality = "Spring"
        elif month in (6, 7, 8):
            seasonality = "Summer"
        else:
            seasonality = "Fall"

        for (store_id, product_id), inv in inv_lookup.items():
            category = inv["Category"]
            region = inv["Region"]
            base_price = BASE_PRICES[category]

            # Daily sales — higher on weekends
            base_daily = inv["Total Units Sold"] / 365
            weekend_mult = 1.3 if is_weekend else 1.0
            units_sold = max(0, int(base_daily * weekend_mult * random.uniform(0.5, 1.5)))

            units_ordered = random.choice([0, 0, 0, random.randint(50, 500)])
            demand_forecast = max(0, int(units_sold * random.uniform(0.85, 1.15)))

            inventory_level = max(0, inv["Current Stock"] - units_sold * (day_offset + 1) // days + units_ordered)
            price = round(base_price * random.uniform(0.95, 1.05), 2)
            discount = round(random.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20]), 2)
            weather = random.choice(WEATHER_CONDITIONS)
            holiday = 1 if random.random() < 0.15 else 0
            competitor_pricing = round(base_price * random.uniform(0.80, 1.20), 2)

            rows.append({
                "Date": date_str,
                "Store ID": store_id,
                "Product ID": product_id,
                "Category": category,
                "Region": region,
                "Inventory Level": inventory_level,
                "Units Sold": units_sold,
                "Units Ordered": units_ordered,
                "Demand Forecast": demand_forecast,
                "Price": price,
                "Discount": discount,
                "Weather Condition": weather,
                "Holiday/Promotion": holiday,
                "Competitor Pricing": competitor_pricing,
                "Seasonality": seasonality,
            })
    return rows


def generate_vendors() -> list[dict]:
    """Vendor records — one per category."""
    vendors = []
    for v in VENDOR_DATA:
        cat = v["category"]
        lt_lo, lt_hi = LEAD_TIME_RANGES[cat]
        vendors.append({
            "vendor_id": f"V{uuid.uuid4().hex[:6].upper()}",
            "name": v["name"],
            "category": cat,
            "region": v["region"],
            "lead_time_days": random.randint(lt_lo, lt_hi),
            "min_order_qty": random.choice([50, 100, 200, 250, 500]),
            "price_per_unit": round(BASE_PRICES[cat] * random.uniform(0.4, 0.7), 2),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "contact_email": f"{v['name'].split()[0].lower()}@supplier.com",
        })
    return vendors


# ── Output helpers ────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], filename: str):
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path}")


def insert_mongo(rows: list[dict], db_name: str, collection_name: str, replace: bool = True):
    from pymongo import MongoClient

    uri = os.getenv("MONGO_URI")
    if not uri:
        raise RuntimeError("MONGO_URI not set — cannot insert into MongoDB")

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    coll = client[db_name][collection_name]

    if replace:
        coll.delete_many({})
        print(f"  Cleared {db_name}.{collection_name}")

    coll.insert_many(rows)
    print(f"  Inserted {len(rows)} docs → {db_name}.{collection_name}")
    client.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Stratos data")
    parser.add_argument("--mode", choices=["mongo", "csv", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Generating inventory_snapshot …")
    inventory = generate_inventory_snapshot()
    print(f"  {len(inventory)} records")

    print("Generating retail_snapshot …")
    retail = generate_retail_snapshot(inventory, days=30)
    print(f"  {len(retail)} records")

    print("Generating vendors …")
    vendors = generate_vendors()
    print(f"  {len(vendors)} records")

    if args.mode in ("csv", "both"):
        print("\n── Writing CSV files ──")
        write_csv(inventory, "inventory_snapshot.csv")
        write_csv(retail, "retail_snapshot.csv")
        write_csv(vendors, "vendors.csv")

    if args.mode in ("mongo", "both"):
        print("\n── Inserting into MongoDB ──")
        insert_mongo(inventory, "inventory_forecasting", "inventory_snapshot", replace=True)
        insert_mongo(retail, "inventory_forecasting", "retail_snapshot", replace=True)
        insert_mongo(vendors, "stratos_orders", "vendors", replace=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
