"""
MCP Tool definitions for the Inventory Server.

Each function here:
  1. Validates input from the LLM
  2. Delegates to the db layer
  3. Returns structured JSON-serializable responses
"""

from typing import Any

try:
    from . import db
except ImportError:
    import db


# ------------------------------------------------------------------
# Input validation helpers
# ------------------------------------------------------------------
def _require_str(params: dict, key: str) -> str:
    val = params.get(key)
    if not val or not isinstance(val, str) or not val.strip():
        raise ValueError(f"'{key}' is required and must be a non-empty string")
    return val.strip()


def _optional_str(params: dict, key: str) -> str | None:
    val = params.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise ValueError(f"'{key}' must be a string")
    return val.strip() or None


def _optional_int(params: dict, key: str, min_val: int | None = None, max_val: int | None = None) -> int | None:
    val = params.get(key)
    if val is None:
        return None
    try:
        val = int(val)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be an integer")
    if min_val is not None and val < min_val:
        raise ValueError(f"'{key}' must be >= {min_val}")
    if max_val is not None and val > max_val:
        raise ValueError(f"'{key}' must be <= {max_val}")
    return val


# ------------------------------------------------------------------
# Tool: get_inventory_record
# ------------------------------------------------------------------
GET_INVENTORY_RECORD_SCHEMA = {
    "name": "get_inventory_record",
    "description": (
        "Look up the inventory record for a specific product at a specific store. "
        "Returns current stock, units sold, units received, lead time, and category."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "store_id": {
                "type": "string",
                "description": "Store identifier, e.g. 'S001'"
            },
            "product_id": {
                "type": "string",
                "description": "Product identifier, e.g. 'P001'"
            },
        },
        "required": ["store_id", "product_id"],
    },
}


def handle_get_inventory_record(params: dict) -> dict[str, Any]:
    store_id = _require_str(params, "store_id")
    product_id = _require_str(params, "product_id")

    record = db.get_inventory_record(store_id, product_id)
    if record is None:
        return {
            "success": False,
            "error": f"No inventory record found for store '{store_id}', product '{product_id}'",
        }
    return {"success": True, "record": record}


# ------------------------------------------------------------------
# Tool: search_inventory
# ------------------------------------------------------------------
SEARCH_INVENTORY_SCHEMA = {
    "name": "search_inventory",
    "description": (
        "Search and filter inventory records. Supports filtering by store, product, "
        "category, region, stock range, and lead time range. Supports sorting and pagination. "
        "Use this to answer questions like 'which stores in the West are low on stock?' or "
        "'show all Electronics inventory sorted by stock level'."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "store_id": {
                "type": "string",
                "description": "Filter by store ID"
            },
            "product_id": {
                "type": "string",
                "description": "Filter by product ID"
            },
            "category": {
                "type": "string",
                "description": "Filter by category, e.g. 'Electronics', 'Groceries'"
            },
            "region": {
                "type": "string",
                "description": "Filter by region, e.g. 'West', 'East', 'North', 'South'"
            },
            "min_current_stock": {
                "type": "integer",
                "description": "Minimum current stock threshold"
            },
            "max_current_stock": {
                "type": "integer",
                "description": "Maximum current stock threshold (useful for finding low-stock items)"
            },
            "min_lead_time_days": {
                "type": "integer",
                "description": "Minimum lead time in days"
            },
            "max_lead_time_days": {
                "type": "integer",
                "description": "Maximum lead time in days"
            },
            "sort_by": {
                "type": "string",
                "description": "Column to sort by. One of: store_id, product_id, category, region, current_stock, total_units_sold, total_units_received, lead_time_days",
                "default": "store_id"
            },
            "sort_order": {
                "type": "string",
                "enum": ["asc", "desc"],
                "description": "Sort direction",
                "default": "asc"
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (1-200, default 50)",
                "default": 50
            },
            "offset": {
                "type": "integer",
                "description": "Pagination offset (default 0)",
                "default": 0
            },
        },
        "required": [],
    },
}


def handle_search_inventory(params: dict) -> dict[str, Any]:
    try:
        result = db.search_inventory(
            store_id=_optional_str(params, "store_id"),
            product_id=_optional_str(params, "product_id"),
            category=_optional_str(params, "category"),
            region=_optional_str(params, "region"),
            min_current_stock=_optional_int(params, "min_current_stock", min_val=0),
            max_current_stock=_optional_int(params, "max_current_stock", min_val=0),
            min_lead_time_days=_optional_int(params, "min_lead_time_days", min_val=0),
            max_lead_time_days=_optional_int(params, "max_lead_time_days", min_val=0),
            sort_by=params.get("sort_by", "store_id"),
            sort_order=params.get("sort_order", "asc"),
            limit=_optional_int(params, "limit", min_val=1, max_val=200) or 50,
            offset=_optional_int(params, "offset", min_val=0) or 0,
        )
        return {"success": True, **result}
    except ValueError as e:
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# Tool: get_inventory_summary
# ------------------------------------------------------------------
GET_INVENTORY_SUMMARY_SCHEMA = {
    "name": "get_inventory_summary",
    "description": (
        "Get aggregated inventory statistics grouped by store, category, or region. "
        "Returns product count, total/avg/min/max stock, total sold, total received, "
        "and average lead time for each group. "
        "Use this for questions like 'summarize stock by region' or "
        "'what is the average lead time per category?'."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "group_by": {
                "type": "string",
                "enum": ["store_id", "category", "region"],
                "description": "Dimension to group by",
                "default": "store_id"
            },
            "store_id": {
                "type": "string",
                "description": "Optional: limit summary to a specific store"
            },
            "category": {
                "type": "string",
                "description": "Optional: limit summary to a specific category"
            },
            "region": {
                "type": "string",
                "description": "Optional: limit summary to a specific region"
            },
        },
        "required": [],
    },
}


def handle_get_inventory_summary(params: dict) -> dict[str, Any]:
    try:
        rows = db.get_inventory_summary(
            group_by=params.get("group_by", "store_id"),
            store_id=_optional_str(params, "store_id"),
            category=_optional_str(params, "category"),
            region=_optional_str(params, "region"),
        )
        return {"success": True, "summary": rows, "count": len(rows)}
    except ValueError as e:
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# Tool: get_restock_risks
# ------------------------------------------------------------------
GET_RESTOCK_RISKS_SCHEMA = {
    "name": "get_restock_risks",
    "description": (
        "Identify products at risk of running out of stock based on a heuristic "
        "that compares current stock levels, daily sales rate, and lead time. "
        "Risk levels: 'high' (stock won't last through lead time), "
        "'medium' (tight buffer), 'low' (comfortable). "
        "Results are sorted by severity. "
        "Use this for questions like 'which products need restocking?' or "
        "'show high-risk items in the West region'."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "store_id": {
                "type": "string",
                "description": "Optional: filter to a specific store"
            },
            "category": {
                "type": "string",
                "description": "Optional: filter to a specific category"
            },
            "region": {
                "type": "string",
                "description": "Optional: filter to a specific region"
            },
            "risk_level": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Optional: show only this risk level"
            },
            "limit": {
                "type": "integer",
                "description": "Max results (1-200, default 50)",
                "default": 50
            },
        },
        "required": [],
    },
}


def handle_get_restock_risks(params: dict) -> dict[str, Any]:
    try:
        rows = db.get_restock_risks(
            store_id=_optional_str(params, "store_id"),
            category=_optional_str(params, "category"),
            region=_optional_str(params, "region"),
            risk_level=_optional_str(params, "risk_level"),
            limit=_optional_int(params, "limit", min_val=1, max_val=200) or 50,
        )

        # Count by risk level
        counts = {"high": 0, "medium": 0, "low": 0}
        for r in rows:
            counts[r["risk_level"]] += 1

        return {
            "success": True,
            "risks": rows,
            "count": len(rows),
            "risk_breakdown": counts,
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}


# ------------------------------------------------------------------
# Registry — maps tool name → (schema, handler)
# ------------------------------------------------------------------
TOOL_REGISTRY: dict[str, tuple[dict, callable]] = {
    "get_inventory_record": (GET_INVENTORY_RECORD_SCHEMA, handle_get_inventory_record),
    "search_inventory": (SEARCH_INVENTORY_SCHEMA, handle_search_inventory),
    "get_inventory_summary": (GET_INVENTORY_SUMMARY_SCHEMA, handle_get_inventory_summary),
    "get_restock_risks": (GET_RESTOCK_RISKS_SCHEMA, handle_get_restock_risks),
}
