"""
Database access layer for the Inventory MCP Server.

- Read-only: every public function uses SELECT only.
- Parameterized queries: no string interpolation in SQL.
- Returns plain dicts so the tool layer can serialize to JSON.
"""

import os
import sqlite3
from typing import Any

DB_PATH = os.getenv(
    "INVENTORY_DB_PATH",
    os.path.join(os.path.dirname(__file__), "inventory.db"),
)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # dict-like access
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA query_only=ON")  # enforce read-only at DB level
    return conn


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# 1. get_inventory_record — exact lookup
# ------------------------------------------------------------------
def get_inventory_record(
    store_id: str, product_id: str
) -> dict[str, Any] | None:
    """Return a single inventory row for a store + product pair."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM inventory WHERE store_id = ? AND product_id = ?",
            (store_id, product_id),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ------------------------------------------------------------------
# 2. search_inventory — flexible filtered search
# ------------------------------------------------------------------
# Allowed columns for filtering, sorting
_FILTERABLE = {"store_id", "product_id", "category", "region"}
_NUMERIC_RANGE = {"current_stock", "lead_time_days", "total_units_sold", "total_units_received"}
_SORTABLE = {
    "store_id", "product_id", "category", "region",
    "current_stock", "total_units_sold", "total_units_received", "lead_time_days",
}


def search_inventory(
    *,
    store_id: str | None = None,
    product_id: str | None = None,
    category: str | None = None,
    region: str | None = None,
    min_current_stock: int | None = None,
    max_current_stock: int | None = None,
    min_lead_time_days: int | None = None,
    max_lead_time_days: int | None = None,
    sort_by: str = "store_id",
    sort_order: str = "asc",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Search inventory with optional filters, sorting, and pagination.
    Returns {results: [...], total: int, limit: int, offset: int}.
    """
    # Validate sort column
    if sort_by not in _SORTABLE:
        raise ValueError(f"sort_by must be one of {sorted(_SORTABLE)}")
    if sort_order.lower() not in ("asc", "desc"):
        raise ValueError("sort_order must be 'asc' or 'desc'")
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    where_clauses: list[str] = []
    params: list[Any] = []

    # Exact-match filters
    for col, val in [
        ("store_id", store_id),
        ("product_id", product_id),
        ("category", category),
        ("region", region),
    ]:
        if val is not None:
            where_clauses.append(f"{col} = ?")
            params.append(val)

    # Range filters
    for col, lo, hi in [
        ("current_stock", min_current_stock, max_current_stock),
        ("lead_time_days", min_lead_time_days, max_lead_time_days),
    ]:
        if lo is not None:
            where_clauses.append(f"{col} >= ?")
            params.append(lo)
        if hi is not None:
            where_clauses.append(f"{col} <= ?")
            params.append(hi)

    where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    conn = _connect()
    try:
        # Total count for pagination metadata
        total = conn.execute(
            f"SELECT COUNT(*) FROM inventory{where_sql}", params
        ).fetchone()[0]

        # Actual results — sort_by is validated above so safe to interpolate
        rows = conn.execute(
            f"SELECT * FROM inventory{where_sql} "
            f"ORDER BY {sort_by} {sort_order.upper()} "
            f"LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

        return {
            "results": _rows_to_dicts(rows),
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    finally:
        conn.close()


# ------------------------------------------------------------------
# 3. get_inventory_summary — aggregation by dimension
# ------------------------------------------------------------------
_GROUP_BY_ALLOWED = {"store_id", "category", "region"}


def get_inventory_summary(
    *,
    group_by: str = "store_id",
    store_id: str | None = None,
    category: str | None = None,
    region: str | None = None,
) -> list[dict[str, Any]]:
    """
    Aggregate inventory stats grouped by store_id, category, or region.
    Returns sum/avg/min/max for stock, sales, received, and lead time.
    """
    if group_by not in _GROUP_BY_ALLOWED:
        raise ValueError(f"group_by must be one of {sorted(_GROUP_BY_ALLOWED)}")

    where_clauses: list[str] = []
    params: list[Any] = []
    for col, val in [
        ("store_id", store_id),
        ("category", category),
        ("region", region),
    ]:
        if val is not None:
            where_clauses.append(f"{col} = ?")
            params.append(val)

    where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT
            {group_by},
            COUNT(*)                    AS product_count,
            SUM(current_stock)          AS total_stock,
            AVG(current_stock)          AS avg_stock,
            MIN(current_stock)          AS min_stock,
            MAX(current_stock)          AS max_stock,
            SUM(total_units_sold)       AS total_sold,
            SUM(total_units_received)   AS total_received,
            ROUND(AVG(lead_time_days), 1) AS avg_lead_time_days,
            MAX(lead_time_days)         AS max_lead_time_days
        FROM inventory{where_sql}
        GROUP BY {group_by}
        ORDER BY {group_by}
    """

    conn = _connect()
    try:
        rows = conn.execute(sql, params).fetchall()
        return _rows_to_dicts(rows)
    finally:
        conn.close()


# ------------------------------------------------------------------
# 4. get_restock_risks — heuristic risk scoring
# ------------------------------------------------------------------
def get_restock_risks(
    *,
    store_id: str | None = None,
    category: str | None = None,
    region: str | None = None,
    risk_level: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Score every product's restock risk and return sorted by severity.

    Heuristic:
      daily_sales_rate = total_units_sold / 365  (annualized)
      days_of_supply   = current_stock / daily_sales_rate
      coverage_ratio   = days_of_supply / lead_time_days

      HIGH   — coverage_ratio < 1.0  (stock won't last through lead time)
      MEDIUM — coverage_ratio < 2.0  (tight buffer)
      LOW    — coverage_ratio >= 2.0 (comfortable)

    Products with zero sales get LOW risk (no demand signal).
    """
    where_clauses: list[str] = []
    params: list[Any] = []
    for col, val in [
        ("store_id", store_id),
        ("category", category),
        ("region", region),
    ]:
        if val is not None:
            where_clauses.append(f"{col} = ?")
            params.append(val)

    where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    limit = max(1, min(limit, 200))

    conn = _connect()
    try:
        rows = conn.execute(
            f"SELECT * FROM inventory{where_sql}", params
        ).fetchall()

        scored = []
        for row in rows:
            r = dict(row)
            sold = r["total_units_sold"]
            stock = r["current_stock"]
            lead = r["lead_time_days"]

            if sold == 0 or lead == 0:
                daily_rate = 0.0
                days_of_supply = float("inf")
                coverage_ratio = float("inf")
                risk = "low"
            else:
                daily_rate = round(sold / 365, 2)
                days_of_supply = round(stock / daily_rate, 1) if daily_rate > 0 else float("inf")
                coverage_ratio = round(days_of_supply / lead, 2) if lead > 0 else float("inf")

                if coverage_ratio < 1.0:
                    risk = "high"
                elif coverage_ratio < 2.0:
                    risk = "medium"
                else:
                    risk = "low"

            r["daily_sales_rate"] = daily_rate
            r["days_of_supply"] = days_of_supply if days_of_supply != float("inf") else None
            r["coverage_ratio"] = coverage_ratio if coverage_ratio != float("inf") else None
            r["risk_level"] = risk
            scored.append(r)

        # Filter by risk level if requested
        if risk_level and risk_level.lower() in ("high", "medium", "low"):
            scored = [s for s in scored if s["risk_level"] == risk_level.lower()]

        # Sort: high first, then medium, then low
        risk_order = {"high": 0, "medium": 1, "low": 2}
        scored.sort(key=lambda x: (risk_order.get(x["risk_level"], 3), x.get("coverage_ratio") or 999))

        return scored[:limit]
    finally:
        conn.close()
