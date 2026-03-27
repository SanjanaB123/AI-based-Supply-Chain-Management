"""
MCP Server for Inventory Database Queries.

Architecture:
  - Uses the official `mcp` Python SDK (stdio transport).
  - Exposes 4 read-only tools for querying a SQLite inventory database.
  - All queries are parameterized — no raw SQL from the model.
  - Tool layer validates input, DB layer enforces read-only via PRAGMA.

Run:
  python -m mcp_server.server

Or configure in Claude Desktop / claude_desktop_config.json:
  {
    "mcpServers": {
      "inventory": {
        "command": "python",
        "args": ["-m", "mcp_server.server"],
        "cwd": "<path-to-backend>"
      }
    }
  }
"""

import json
import logging
import sys
import os

from mcp.server.fastmcp import FastMCP

# Support both `python -m mcp_server` (relative import) and
# `mcp dev server.py` (direct execution, no package context).
try:
    from .tools import TOOL_REGISTRY
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from tools import TOOL_REGISTRY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inventory-mcp")

# Create the MCP server instance
mcp = FastMCP("Inventory Database")


# ------------------------------------------------------------------
# Register each tool from the registry
# ------------------------------------------------------------------
@mcp.tool(
    name="get_inventory_record",
    description=TOOL_REGISTRY["get_inventory_record"][0]["description"],
)
def get_inventory_record(store_id: str, product_id: str) -> str:
    """Look up inventory for a specific product at a specific store."""
    handler = TOOL_REGISTRY["get_inventory_record"][1]
    result = handler({"store_id": store_id, "product_id": product_id})
    return json.dumps(result, indent=2, default=str)


@mcp.tool(
    name="search_inventory",
    description=TOOL_REGISTRY["search_inventory"][0]["description"],
)
def search_inventory(
    store_id: str = "",
    product_id: str = "",
    category: str = "",
    region: str = "",
    min_current_stock: int | None = None,
    max_current_stock: int | None = None,
    min_lead_time_days: int | None = None,
    max_lead_time_days: int | None = None,
    sort_by: str = "store_id",
    sort_order: str = "asc",
    limit: int = 50,
    offset: int = 0,
) -> str:
    """Search and filter inventory records with sorting and pagination."""
    params = {
        "store_id": store_id or None,
        "product_id": product_id or None,
        "category": category or None,
        "region": region or None,
        "min_current_stock": min_current_stock,
        "max_current_stock": max_current_stock,
        "min_lead_time_days": min_lead_time_days,
        "max_lead_time_days": max_lead_time_days,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "limit": limit,
        "offset": offset,
    }
    handler = TOOL_REGISTRY["search_inventory"][1]
    result = handler(params)
    return json.dumps(result, indent=2, default=str)


@mcp.tool(
    name="get_inventory_summary",
    description=TOOL_REGISTRY["get_inventory_summary"][0]["description"],
)
def get_inventory_summary(
    group_by: str = "store_id",
    store_id: str = "",
    category: str = "",
    region: str = "",
) -> str:
    """Get aggregated inventory statistics grouped by a dimension."""
    params = {
        "group_by": group_by,
        "store_id": store_id or None,
        "category": category or None,
        "region": region or None,
    }
    handler = TOOL_REGISTRY["get_inventory_summary"][1]
    result = handler(params)
    return json.dumps(result, indent=2, default=str)


@mcp.tool(
    name="get_restock_risks",
    description=TOOL_REGISTRY["get_restock_risks"][0]["description"],
)
def get_restock_risks(
    store_id: str = "",
    category: str = "",
    region: str = "",
    risk_level: str = "",
    limit: int = 50,
) -> str:
    """Identify products at risk of running out of stock."""
    params = {
        "store_id": store_id or None,
        "category": category or None,
        "region": region or None,
        "risk_level": risk_level or None,
        "limit": limit,
    }
    handler = TOOL_REGISTRY["get_restock_risks"][1]
    result = handler(params)
    return json.dumps(result, indent=2, default=str)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    logger.info("Starting Inventory MCP Server (stdio transport)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
