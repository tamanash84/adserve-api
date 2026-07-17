#!/usr/bin/env python
"""Build a lean training dataset for XGBoost using DuckDB (with Node2Vec product embeddings).

What this script does
- Reads your fact tables (orders + order lines) and joins dimensions (stores context, weather, prices, node2vec embeddings)
- Keeps only the columns you list in STORE_CTX_COLS / WEATHER_COLS (aggressive column pruning)
- Adds embedding columns from node2vec_embeddings.parquet
    * If EMB_COLS is empty, it auto-detects numeric embedding columns (best for wide emb_0..emb_n)
    * If embeddings are stored as a LIST/ARRAY column (e.g. "embedding"), it can expand the first EMB_LIST_DIM entries
- Writes out a single compressed parquet once (no materialized intermediate tables)

Run:
  python build_train_xgb_duckdb_node2vec.py
"""

import duckdb

# -------------------- Inputs --------------------
ORDERS   = r"../data/parquets/orders_multistore_2026.parquet"         # order-level: order_id, store_id, order_timestamp
OP_QTY   = r"../data/parquets/order_products__prior_qty.parquet"      # line-level: order_id, product_id, quantity (+ maybe aisle_id/department_id)
STORES   = r"../data/parquets/stores_fixed_context.parquet"           # store dimension (demography + POI features)
WEATHER  = r"../data/parquets/weather_hourly.parquet"                 # store_id, ts_hour (+ weather vars)
EVENTS   = r"../data/parquets/events_hourly.parquet"
PRICES   = r"../data/parquets/product_base_prices.parquet"            # product_id, base_price

# IMPORTANT: Node2Vec embeddings
EMBEDS   = r"../data/parquets/product_embeddings.parquet"             # product_id + embedding columns OR product_id + list column

OUT_PATH = r"../data/parquets/train_xgb.parquet"                      # output parquet

# -------------------- Resource controls --------------------
TEMP_DIR     = r"C:/Users/NH61FL/ML_Data/duckdb_spill"  # adjust
THREADS      = 8
MEMORY_LIMIT = "20GB"                                   # tune (6GB–20GB)

# -------------------- Choose ONLY the columns you need --------------------
DEMO_COLS_ESSENTIAL = [
    "total_population",
    "pct_age_0_25",
    "pct_age_25_65",
    "pct_age_gt_65",
    "pct_pop_dutch",
    "pct_pop_europe",
    "pct_pop_asia",
    "pct_pop_africa",
]

POI_COLS_ESSENTIAL = [
    # --- immediate (0–50m): “front door” micro-context ---
    "poi_shop_grocery_supermarket_count_b0_50",
    "poi_shop_grocery_convenience_count_b0_50",
    "poi_pt_access_bus_stop_count_b0_50",
    "poi_office_any_count_b0_50",
    "poi_landuse_residential_count_b0_50",
    "poi_building_apartments_count_b0_50",
    "poi_walkability_links_count_b0_50",
    "poi_walkability_crossing_count_b0_50",

    # --- near (50–200m): local neighborhood ---
    "poi_shop_grocery_supermarket_count_b50_200",
    "poi_shop_grocery_convenience_count_b50_200",
    "poi_shop_food_nonsupermarket_count_b50_200",
    "poi_pt_access_bus_stop_count_b50_200",
    "poi_pt_access_rail_count_b50_200",
    "poi_office_any_count_b50_200",
    "poi_landuse_residential_count_b50_200",
    "poi_building_apartments_count_b50_200",
    "poi_building_house_count_b50_200",
    "poi_leisure_park_count_b50_200",
    "poi_amenity_food_fastfood_count_b50_200",
    "poi_amenity_food_nonfastfood_count_b50_200",
    "poi_walkability_links_count_b50_200",
    "poi_walkability_crossing_count_b50_200",

    # --- catchment (200–800m): broader draw / competition field ---
    "poi_shop_grocery_supermarket_count_b200_800",
    "poi_shop_grocery_convenience_count_b200_800",
    "poi_shop_food_nonsupermarket_count_b200_800",
    "poi_pt_access_bus_stop_count_b200_800",
    "poi_pt_access_rail_count_b200_800",
    "poi_office_any_count_b200_800",
    "poi_landuse_residential_count_b200_800",
    "poi_landuse_industrial_count_b200_800",
    "poi_building_apartments_count_b200_800",
    "poi_building_house_count_b200_800",
    "poi_leisure_park_count_b200_800",
    "poi_amenity_food_fastfood_count_b200_800",
    "poi_amenity_food_nonfastfood_count_b200_800",
    "poi_walkability_links_count_b200_800",
    "poi_walkability_crossing_count_b200_800",
]

POI_MIN_DIST_ADDON = [
    "poi_shop_grocery_supermarket_min_dist_m_b0_50",
    "poi_shop_grocery_supermarket_min_dist_m_b50_200",
    "poi_shop_grocery_supermarket_min_dist_m_b200_800",
    "poi_pt_access_bus_stop_min_dist_m_b50_200",
    "poi_pt_access_rail_min_dist_m_b200_800",
]

# Store context columns (example: keep only what your model uses)
# Replace these with the real set you use in predict_reward_xgboost.py
STORE_CTX_COLS = (
    DEMO_COLS_ESSENTIAL
    +
    POI_COLS_ESSENTIAL
    +
    POI_MIN_DIST_ADDON
)

WEATHER_COLS = [
    "weather_temp2m_mean_t0_3h",
    "weather_prop_precip_t0_3h",
    "weather_prop_heavy_precip_t0_3h",
    "weather_prop_cloudy_t0_3h",
    "weather_wind10m_speed_max_t0_3h",
    "weather_is_day_t0_3h",
]


EVENT_COLS = [
    "events_count_b_b0_1km_t_b0_3h",
    "events_count_b_b0_1km_t_b3_24h",
    "events_weighted_sum_t0_24h",
    "unique_venues_t0_24h",
    "max_events_same_venue_t0_24h",
    "events_count_b_b1_3km_t_b0_3h",
    "events_min_dist_km_b_b0_1km_t_b0_3h",
    "events_music_count_b_b0_1km_t_b0_3h",
    "events_nightlife_count_b_b0_1km_t_b0_3h",
    "events_family_count_b_b0_1km_t_b0_3h",
]


# Embedding columns.
# Leave empty to auto-detect ALL numeric embedding columns.
# If embeddings are a list column, set EMB_LIST_COL and EMB_LIST_DIM.
EMB_COLS = [str(i) for i in range(128)]  # column names in parquet are "0".."127"

# -------------------- Optional early filters --------------------
STORE_FILTER = None   # e.g. [1,2,3]
START_TS     = None   # e.g. "2026-01-01"
END_TS       = None   # e.g. "2026-03-01"


def _sql_list(cols):
    return ",\n            ".join(cols)


def main():
    con = duckdb.connect()

    # Pragmas
    con.execute(f"PRAGMA temp_directory='{TEMP_DIR}';")
    con.execute(f"PRAGMA threads={THREADS};")
    con.execute(f"PRAGMA memory_limit='{MEMORY_LIMIT}';")
    con.execute("SET preserve_insertion_order=false;")

    # -------------------- dimension views --------------------
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW stores_proj AS
        SELECT { _sql_list(['store_id'] + STORE_CTX_COLS) if STORE_CTX_COLS else 'store_id' }
        FROM read_parquet('{STORES}');
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW weather_dedup AS
        SELECT { _sql_list(['store_id', 'ts_hour'] + WEATHER_COLS) if WEATHER_COLS else 'store_id, ts_hour' }
        FROM read_parquet('{WEATHER}')
        QUALIFY row_number() OVER (PARTITION BY store_id, ts_hour ORDER BY ts_hour) = 1;
    """)
    
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW events_dedup AS
        SELECT {_sql_list(['store_id', 'ts_hour'] + EVENT_COLS) if EVENT_COLS else 'store_id, ts_hour'}
        FROM read_parquet('{EVENTS}')
        QUALIFY row_number() OVER (PARTITION BY store_id, ts_hour ORDER BY ts_hour) = 1;
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW prices_dedup AS
        SELECT product_id, base_price
        FROM read_parquet('{PRICES}')
        QUALIFY row_number() OVER (PARTITION BY product_id ORDER BY product_id) = 1;
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW embeds_dedup AS
        SELECT *
        EXCLUDE (product_name, community_id),
        CAST(community_id AS INT) AS community_id
        FROM read_parquet('{EMBEDS}')
    """)
        
    # -------------------- orders view (with early filters) --------------------
    where = []
    if STORE_FILTER:
        where.append(f"store_id IN ({', '.join(map(str, STORE_FILTER))})")
    if START_TS:
        where.append(f"order_timestamp >= TIMESTAMP '{START_TS}'")
    if END_TS:
        where.append(f"order_timestamp < TIMESTAMP '{END_TS}'")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW orders_base AS
        SELECT
            order_id,
            store_id,
            order_timestamp,
            date_trunc('hour', order_timestamp) AS ts_hour
        FROM read_parquet('{ORDERS}')
        {where_sql};
    """)

    # -------------------- sql ready list of column names --------------------

    store_cols_sql = ", ".join([f"st.{c}" for c in STORE_CTX_COLS]) if STORE_CTX_COLS else "NULL AS _no_store_ctx"
    weather_cols_sql = ", ".join([f"w.{c}" for c in WEATHER_COLS]) if WEATHER_COLS else "NULL AS _no_weather"
    events_cols_sql = ", ".join([f"ev.{c}" for c in EVENT_COLS]) if EVENT_COLS else "NULL AS _no_events"
    embed_out_cols = [r[0] for r in con.execute("DESCRIBE SELECT * FROM embeds_dedup").fetchall()
                      if r[0] != 'product_id']
    emb_cols_sql = ", ".join([f"e.{c}" for c in embed_out_cols]) if embed_out_cols else "NULL AS _no_embeds"

    # -------------------- final join + write once --------------------
    con.execute(f"""
        COPY (
            WITH lines AS (
                SELECT order_id, product_id, quantity
                FROM read_parquet('{OP_QTY}')
            )
            SELECT
                ob.store_id,
                ob.order_id,
                l.product_id,
                ob.order_timestamp,
                l.quantity,
                {store_cols_sql},
                {weather_cols_sql},
                {events_cols_sql},
                {emb_cols_sql},

                p.base_price,
                ROUND(p.base_price * l.quantity, 2) AS line_amount

            FROM orders_base ob
            INNER JOIN lines l USING (order_id)
            LEFT JOIN prices_dedup p ON p.product_id = l.product_id
            LEFT JOIN embeds_dedup e ON e.product_id = l.product_id
            LEFT JOIN stores_proj st USING (store_id)
            LEFT JOIN weather_dedup w
                ON ob.store_id = w.store_id AND ob.ts_hour = w.ts_hour
            LEFT JOIN events_dedup ev
                ON ob.store_id = ev.store_id AND ob.ts_hour = ev.ts_hour

        ) TO '{OUT_PATH}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000);
    """)

    print("Wrote", OUT_PATH)


if __name__ == "__main__":
    main()
