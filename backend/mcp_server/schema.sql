-- ============================================================
-- Inventory table for the Supply Chain MCP Server
-- ============================================================
-- Each row represents a product's inventory snapshot at a store.
-- This is a read-optimized schema — no foreign keys for simplicity.

CREATE TABLE IF NOT EXISTS inventory (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id        TEXT    NOT NULL,
    product_id      TEXT    NOT NULL,
    category        TEXT    NOT NULL,
    region          TEXT    NOT NULL,
    current_stock   INTEGER NOT NULL DEFAULT 0,
    total_units_sold     INTEGER NOT NULL DEFAULT 0,
    total_units_received INTEGER NOT NULL DEFAULT 0,
    lead_time_days  INTEGER NOT NULL DEFAULT 0,

    -- Composite uniqueness: one row per store-product pair
    UNIQUE(store_id, product_id)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_inventory_store    ON inventory(store_id);
CREATE INDEX IF NOT EXISTS idx_inventory_product  ON inventory(product_id);
CREATE INDEX IF NOT EXISTS idx_inventory_category ON inventory(category);
CREATE INDEX IF NOT EXISTS idx_inventory_region   ON inventory(region);
CREATE INDEX IF NOT EXISTS idx_inventory_stock    ON inventory(current_stock);
