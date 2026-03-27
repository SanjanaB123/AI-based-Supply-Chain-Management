"""
Seed the inventory database with example data.

Run:  python -m mcp_server.seed_data
      (from the backend/ directory)

This creates/resets inventory.db with realistic sample data spanning
5 stores, 4 regions, 5 categories, and 20 products.
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "inventory.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

SEED_ROWS = [
    # store_id, product_id, category, region, current_stock, total_units_sold, total_units_received, lead_time_days
    # ── Store S001 (West) ──
    ("S001", "P001", "Electronics",    "West",   12,  980,  1000,  14),
    ("S001", "P002", "Electronics",    "West",  250,  600,   900,   7),
    ("S001", "P003", "Groceries",      "West",   45, 3200,  3300,   3),
    ("S001", "P004", "Clothing",       "West",   80,  450,   550,  10),
    ("S001", "P005", "Home & Garden",  "West",    5, 1200,  1220,  21),

    # ── Store S002 (East) ──
    ("S002", "P001", "Electronics",    "East",  300,  400,   700,  14),
    ("S002", "P002", "Electronics",    "East",   18,  850,   900,   7),
    ("S002", "P006", "Groceries",      "East",  500, 2800,  3500,   2),
    ("S002", "P007", "Clothing",       "East",   30,  700,   750,  12),
    ("S002", "P008", "Home & Garden",  "East",  150,  350,   520,  18),

    # ── Store S003 (North) ──
    ("S003", "P003", "Groceries",      "North",  10, 4100,  4200,   4),
    ("S003", "P009", "Electronics",    "North", 200,  550,   800,  10),
    ("S003", "P010", "Clothing",       "North",  60,  300,   380,   8),
    ("S003", "P011", "Automotive",     "North",  25,  180,   210,  30),
    ("S003", "P005", "Home & Garden",  "North",  90,  900,  1000,  21),

    # ── Store S004 (South) ──
    ("S004", "P001", "Electronics",    "South",   8, 1100,  1150,  14),
    ("S004", "P012", "Groceries",      "South", 600, 5000,  5800,   3),
    ("S004", "P013", "Clothing",       "South",  15,  920,   950,  11),
    ("S004", "P014", "Automotive",     "South",  40,  260,   310,  25),
    ("S004", "P008", "Home & Garden",  "South", 100,  480,   600,  18),

    # ── Store S005 (West) ──
    ("S005", "P015", "Electronics",    "West",   22, 1400,  1450,  12),
    ("S005", "P003", "Groceries",      "West",  180, 2600,  2800,   3),
    ("S005", "P016", "Clothing",       "West",    3,  800,   820,   9),
    ("S005", "P017", "Automotive",     "West",   70,  150,   230,  28),
    ("S005", "P005", "Home & Garden",  "West",   35,  750,   800,  21),
]


def seed():
    # Remove old DB
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)

    # Run schema
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    # Insert seed data
    conn.executemany(
        """
        INSERT INTO inventory
            (store_id, product_id, category, region,
             current_stock, total_units_sold, total_units_received, lead_time_days)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        SEED_ROWS,
    )
    conn.commit()
    conn.close()
    print(f"Seeded {len(SEED_ROWS)} rows into {DB_PATH}")


if __name__ == "__main__":
    seed()
