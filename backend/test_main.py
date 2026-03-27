import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

STORES = ["S001", "S002", "S003", "S004", "S005"]
HEALTH_STATUSES = {"critical", "low", "healthy"}
CATEGORIES = {"Groceries", "Beverages", "Household", "Personal Care", "Snacks"}


# ── /api/stores ──────────────────────────────────────────


class TestStores:
    def test_returns_all_five_stores(self):
        resp = client.get("/api/stores")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stores"] == STORES

    def test_stores_are_sorted(self):
        data = client.get("/api/stores").json()
        assert data["stores"] == sorted(data["stores"])


# ── /api/stock-levels ────────────────────────────────────


class TestStockLevels:
    def test_valid_store(self):
        resp = client.get("/api/stock-levels", params={"store": "S001"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["store"] == "S001"
        assert isinstance(data["total_stock"], int)
        assert data["total_stock"] > 0

    def test_has_20_products(self):
        data = client.get("/api/stock-levels", params={"store": "S001"}).json()
        assert len(data["products"]) == 20

    def test_summary_counts_add_up(self):
        data = client.get("/api/stock-levels", params={"store": "S001"}).json()
        s = data["summary"]
        assert s["critical"] + s["low"] + s["healthy"] == 20

    def test_products_sorted_low_to_high(self):
        data = client.get("/api/stock-levels", params={"store": "S001"}).json()
        stocks = [p["current_stock"] for p in data["products"]]
        assert stocks == sorted(stocks)

    def test_product_shape(self):
        data = client.get("/api/stock-levels", params={"store": "S001"}).json()
        p = data["products"][0]
        assert "product_id" in p
        assert "category" in p
        assert "current_stock" in p
        assert p["stock_health"] in HEALTH_STATUSES
        assert p["category"] in CATEGORIES

    def test_total_stock_matches_sum(self):
        data = client.get("/api/stock-levels", params={"store": "S001"}).json()
        assert data["total_stock"] == sum(p["current_stock"] for p in data["products"])

    def test_invalid_store_404(self):
        resp = client.get("/api/stock-levels", params={"store": "S999"})
        assert resp.status_code == 404

    def test_missing_store_param_422(self):
        resp = client.get("/api/stock-levels")
        assert resp.status_code == 422

    def test_all_stores_return_data(self):
        for store in STORES:
            resp = client.get("/api/stock-levels", params={"store": store})
            assert resp.status_code == 200
            assert len(resp.json()["products"]) == 20


# ── /api/sell-through ────────────────────────────────────


class TestSellThrough:
    def test_valid_store(self):
        resp = client.get("/api/sell-through", params={"store": "S001"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["store"] == "S001"
        assert len(data["products"]) == 20

    def test_sorted_high_to_low(self):
        data = client.get("/api/sell-through", params={"store": "S001"}).json()
        rates = [p["sell_through_rate"] for p in data["products"]]
        assert rates == sorted(rates, reverse=True)

    def test_rate_between_0_and_200(self):
        data = client.get("/api/sell-through", params={"store": "S001"}).json()
        for p in data["products"]:
            assert 0 <= p["sell_through_rate"] <= 200

    def test_product_shape(self):
        data = client.get("/api/sell-through", params={"store": "S001"}).json()
        p = data["products"][0]
        assert "product_id" in p
        assert "category" in p
        assert "sell_through_rate" in p
        assert "total_sold" in p
        assert "total_received" in p

    def test_rate_formula(self):
        data = client.get("/api/sell-through", params={"store": "S001"}).json()
        for p in data["products"]:
            expected = round((p["total_sold"] / p["total_received"]) * 100, 2)
            assert abs(p["sell_through_rate"] - expected) < 0.02

    def test_invalid_store_404(self):
        resp = client.get("/api/sell-through", params={"store": "XXXX"})
        assert resp.status_code == 404


# ── /api/days-of-supply ──────────────────────────────────


class TestDaysOfSupply:
    def test_valid_store(self):
        resp = client.get("/api/days-of-supply", params={"store": "S001"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["store"] == "S001"
        assert len(data["products"]) == 20

    def test_thresholds_present(self):
        data = client.get("/api/days-of-supply", params={"store": "S001"}).json()
        assert data["thresholds"]["critical_below"] == 14
        assert data["thresholds"]["low_below"] == 45

    def test_sorted_ascending(self):
        data = client.get("/api/days-of-supply", params={"store": "S001"}).json()
        dos_values = [p["days_of_supply"] for p in data["products"]]
        assert dos_values == sorted(dos_values)

    def test_health_matches_thresholds(self):
        data = client.get("/api/days-of-supply", params={"store": "S001"}).json()
        for p in data["products"]:
            dos = p["days_of_supply"]
            if dos < 14:
                assert p["stock_health"] == "critical"
            elif dos < 45:
                assert p["stock_health"] == "low"
            else:
                assert p["stock_health"] == "healthy"

    def test_product_shape(self):
        data = client.get("/api/days-of-supply", params={"store": "S001"}).json()
        p = data["products"][0]
        assert "product_id" in p
        assert "days_of_supply" in p
        assert "stock_health" in p
        assert "current_stock" in p
        assert "daily_sales" in p
        assert p["daily_sales"] > 0

    def test_invalid_store_404(self):
        resp = client.get("/api/days-of-supply", params={"store": "NOPE"})
        assert resp.status_code == 404


# ── /api/stock-health ────────────────────────────────────


class TestStockHealth:
    def test_valid_store(self):
        resp = client.get("/api/stock-health", params={"store": "S001"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["store"] == "S001"
        assert data["total_products"] == 20

    def test_breakdown_has_three_statuses(self):
        data = client.get("/api/stock-health", params={"store": "S001"}).json()
        statuses = {b["status"] for b in data["breakdown"]}
        assert statuses == HEALTH_STATUSES

    def test_counts_add_to_total(self):
        data = client.get("/api/stock-health", params={"store": "S001"}).json()
        total = sum(b["count"] for b in data["breakdown"])
        assert total == data["total_products"]

    def test_percentages_add_to_100(self):
        data = client.get("/api/stock-health", params={"store": "S001"}).json()
        total_pct = sum(b["percentage"] for b in data["breakdown"])
        assert abs(total_pct - 100.0) < 0.5

    def test_breakdown_shape(self):
        data = client.get("/api/stock-health", params={"store": "S001"}).json()
        for b in data["breakdown"]:
            assert "status" in b
            assert "count" in b
            assert "percentage" in b
            assert isinstance(b["count"], int)
            assert isinstance(b["percentage"], float)

    def test_consistent_with_stock_levels(self):
        sl = client.get("/api/stock-levels", params={"store": "S002"}).json()
        sh = client.get("/api/stock-health", params={"store": "S002"}).json()
        health_map = {b["status"]: b["count"] for b in sh["breakdown"]}
        assert sl["summary"] == health_map

    def test_invalid_store_404(self):
        resp = client.get("/api/stock-health", params={"store": "BAD"})
        assert resp.status_code == 404


# ── /api/lead-time-risk ──────────────────────────────────


class TestLeadTimeRisk:
    def test_valid_store(self):
        resp = client.get("/api/lead-time-risk", params={"store": "S001"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["store"] == "S001"
        assert len(data["products"]) == 20

    def test_product_shape(self):
        data = client.get("/api/lead-time-risk", params={"store": "S001"}).json()
        p = data["products"][0]
        assert "product_id" in p
        assert "category" in p
        assert "lead_time_days" in p
        assert "days_of_supply" in p
        assert "stock_health" in p

    def test_lead_time_positive(self):
        data = client.get("/api/lead-time-risk", params={"store": "S001"}).json()
        for p in data["products"]:
            assert p["lead_time_days"] > 0

    def test_days_of_supply_non_negative(self):
        data = client.get("/api/lead-time-risk", params={"store": "S001"}).json()
        for p in data["products"]:
            assert p["days_of_supply"] >= 0

    def test_invalid_store_404(self):
        resp = client.get("/api/lead-time-risk", params={"store": "ZZZ"})
        assert resp.status_code == 404


# ── /api/shrinkage ───────────────────────────────────────


class TestShrinkage:
    def test_valid_store(self):
        resp = client.get("/api/shrinkage", params={"store": "S001"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["store"] == "S001"
        assert len(data["products"]) == 20

    def test_total_matches_sum(self):
        data = client.get("/api/shrinkage", params={"store": "S001"}).json()
        assert data["total_shrinkage"] == sum(p["shrinkage"] for p in data["products"])

    def test_shrinkage_formula(self):
        data = client.get("/api/shrinkage", params={"store": "S001"}).json()
        for p in data["products"]:
            expected = p["total_received"] - p["total_sold"] - p["current_stock"]
            assert p["shrinkage"] == expected

    def test_sorted_descending(self):
        data = client.get("/api/shrinkage", params={"store": "S001"}).json()
        values = [p["shrinkage"] for p in data["products"]]
        assert values == sorted(values, reverse=True)

    def test_product_shape(self):
        data = client.get("/api/shrinkage", params={"store": "S001"}).json()
        p = data["products"][0]
        assert "product_id" in p
        assert "category" in p
        assert "total_received" in p
        assert "total_sold" in p
        assert "current_stock" in p
        assert "shrinkage" in p

    def test_invalid_store_404(self):
        resp = client.get("/api/shrinkage", params={"store": "FAKE"})
        assert resp.status_code == 404


# ── Cross-endpoint consistency ───────────────────────────


class TestCrossEndpoint:
    def test_product_count_consistent_across_endpoints(self):
        for store in STORES:
            counts = []
            for ep in ["/api/stock-levels", "/api/sell-through", "/api/days-of-supply",
                       "/api/lead-time-risk", "/api/shrinkage"]:
                data = client.get(ep, params={"store": store}).json()
                counts.append(len(data["products"]))
            assert all(c == 20 for c in counts), f"Mismatch for {store}: {counts}"

    def test_product_ids_consistent(self):
        ids_sl = {p["product_id"] for p in client.get("/api/stock-levels", params={"store": "S001"}).json()["products"]}
        ids_st = {p["product_id"] for p in client.get("/api/sell-through", params={"store": "S001"}).json()["products"]}
        ids_sh = {p["product_id"] for p in client.get("/api/shrinkage", params={"store": "S001"}).json()["products"]}
        assert ids_sl == ids_st == ids_sh

    def test_stock_health_counts_match_stock_levels_summary(self):
        for store in STORES:
            sl = client.get("/api/stock-levels", params={"store": store}).json()
            sh = client.get("/api/stock-health", params={"store": store}).json()
            health_map = {b["status"]: b["count"] for b in sh["breakdown"]}
            assert sl["summary"] == health_map, f"Mismatch for {store}"
