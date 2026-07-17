#!/usr/bin/env python
from datetime import datetime
import duckdb
from config import Paths

stores_path = str(Paths.STORES)
weather_path  = str(Paths.WEATHER)                
events_path   = str(Paths.EVENTS)
embeds_path   = str(Paths.EMBEDS)            
prices_path   = str(Paths.PRICES)

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


def _sql_list(cols):
    return ",\n            ".join(cols)

def get_ctx_per_product(cand_prod_ids):
    cand_prod_ids = list(map(int, cand_prod_ids))
    with duckdb.connect() as con:
        
        product_ctx = con.execute(f"""
            WITH ids AS (SELECT * FROM UNNEST(?) AS t(product_id)),
            p AS (
              SELECT product_id, base_price
              FROM read_parquet('{prices_path}')
              WHERE product_id IN (SELECT product_id FROM ids)
              QUALIFY row_number() OVER (PARTITION BY product_id ORDER BY product_id) = 1
            )
            SELECT
              e.* EXCLUDE (product_name, community_id),
              CAST(e.community_id AS INTEGER) AS community_id,
              p.base_price
            FROM read_parquet('{embeds_path}') e
            JOIN ids USING (product_id)
            LEFT JOIN p USING (product_id)
        """, [cand_prod_ids]).df().to_dict(orient="records")       
       
    return product_ctx

def get_ctx_per_store(store_ids, order_ts):
    ts_hour = order_ts.replace(minute=0, second=0, microsecond=0)
    with duckdb.connect() as con:       
       
        shared_ctx = con.execute(f"""
            WITH
            store_list AS (
              SELECT store_id FROM UNNEST(?) AS t(store_id)
            ),
            w AS (
              SELECT { _sql_list(['store_id','ts_hour'] + WEATHER_COLS) if WEATHER_COLS else 'store_id, ts_hour' }
              FROM read_parquet('{weather_path}')
              WHERE store_id IN (SELECT store_id FROM store_list) AND ts_hour = ?
              QUALIFY row_number() OVER (PARTITION BY store_id, ts_hour ORDER BY ts_hour) = 1
            ),
            ev AS (
              SELECT { _sql_list(['store_id','ts_hour'] + EVENT_COLS) if EVENT_COLS else 'store_id, ts_hour' }
              FROM read_parquet('{events_path}')
              WHERE store_id IN (SELECT store_id FROM store_list) AND ts_hour = ?
              QUALIFY row_number() OVER (PARTITION BY store_id, ts_hour ORDER BY ts_hour) = 1
            ),
            st AS (
              SELECT { _sql_list(['store_id'] + STORE_CTX_COLS) if STORE_CTX_COLS else 'store_id' }
              FROM read_parquet('{stores_path}')
              WHERE store_id IN (SELECT store_id FROM store_list)
            )
            SELECT
              w.store_id,
              EXTRACT('hour' FROM w.ts_hour) AS hod,
              EXTRACT('dow' FROM w.ts_hour) AS dow,
              EXTRACT('month' FROM w.ts_hour) AS month,
              {", ".join([f"w.{c}" for c in WEATHER_COLS]) if WEATHER_COLS else "NULL AS _no_weather"},
              {", ".join([f"ev.{c}" for c in EVENT_COLS]) if EVENT_COLS else "NULL AS _no_events"},
              {", ".join([f"st.{c}" for c in STORE_CTX_COLS]) if STORE_CTX_COLS else "NULL AS _no_store_ctx"}
            FROM w
            LEFT JOIN ev USING (store_id, ts_hour)
            LEFT JOIN st USING (store_id)
        """, [store_ids, ts_hour, ts_hour]).df().to_dict(orient="records")
        
    return shared_ctx

# ############### Example use ##############################################
cand_prod_ids = [33722, 11215]
store_ids = [1,4]
order_ts = datetime(2026, 4, 21, 10, 30)
    
shared_ctx = get_ctx_per_store(store_ids, order_ts)
product_ctx = get_ctx_per_product(cand_prod_ids)
context = [{**shared_ctx[0], **ctx} for ctx in product_ctx]
