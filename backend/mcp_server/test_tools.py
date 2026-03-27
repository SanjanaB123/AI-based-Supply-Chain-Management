"""
Interactive test script for all MCP inventory tools.
Run: python -m mcp_server.test_tools
"""

import json
from .tools import TOOL_REGISTRY


def run_test(tool_name: str, params: dict):
    print(f"\n{'='*60}")
    print(f"TOOL: {tool_name}")
    print(f"INPUT: {json.dumps(params)}")
    print("-" * 60)

    schema, handler = TOOL_REGISTRY[tool_name]
    result = handler(params)
    print(json.dumps(result, indent=2, default=str))
    status = "PASS" if result.get("success") else "FAIL"
    print(f"[{status}]")
    return result


def main():
    print("=" * 60)
    print("  MCP INVENTORY TOOL TEST SUITE")
    print("=" * 60)

    # ── 1. get_inventory_record ──
    run_test("get_inventory_record", {
        "store_id": "S001",
        "product_id": "P001",
    })

    # Not found case
    run_test("get_inventory_record", {
        "store_id": "S999",
        "product_id": "P999",
    })

    # ── 2. search_inventory ──
    # Low stock items in the West
    run_test("search_inventory", {
        "region": "West",
        "max_current_stock": 20,
        "sort_by": "current_stock",
        "sort_order": "asc",
    })

    # All Electronics sorted by sales desc
    run_test("search_inventory", {
        "category": "Electronics",
        "sort_by": "total_units_sold",
        "sort_order": "desc",
    })

    # Pagination test
    run_test("search_inventory", {
        "limit": 3,
        "offset": 0,
    })

    # ── 3. get_inventory_summary ──
    # By region
    run_test("get_inventory_summary", {
        "group_by": "region",
    })

    # By category, filtered to West
    run_test("get_inventory_summary", {
        "group_by": "category",
        "region": "West",
    })

    # By store, filtered to Electronics
    run_test("get_inventory_summary", {
        "group_by": "store_id",
        "category": "Electronics",
    })

    # ── 4. get_restock_risks ──
    # All high risk
    run_test("get_restock_risks", {
        "risk_level": "high",
    })

    # Risks in South region
    run_test("get_restock_risks", {
        "region": "South",
    })

    # Risks for Groceries
    run_test("get_restock_risks", {
        "category": "Groceries",
        "limit": 5,
    })

    # ── Edge cases ──
    # Invalid sort column
    run_test("search_inventory", {
        "sort_by": "nonexistent_column",
    })

    # Invalid group_by
    run_test("get_inventory_summary", {
        "group_by": "invalid",
    })

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
