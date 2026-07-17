#!/usr/bin/env python

import duckdb

# -------------------- Inputs --------------------
ORDERS   = r"../data/parquets/orders_multistore_2026.parquet"         # order-level: order_id, store_id, order_timestamp
OP_QTY   = r"../data/parquets/order_products__prior_qty.parquet"      # line-level: order_id, product_id, quantity (+ maybe aisle_id/department_id)
PRICES   = r"../data/parquets/product_base_prices.parquet"            # product_id, base_price


OUT_PATH = r"../data/parquets/orders_prod_multistore_pos.parquet"                      # output parquet

# -------------------- Resource controls --------------------
TEMP_DIR     = r"C:/Users/NH61FL/ML_Data/duckdb_spill"  # adjust
THREADS      = 8
MEMORY_LIMIT = "20GB"                                   # tune (6GB–20GB)

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
        CREATE OR REPLACE TEMP VIEW prices_dedup AS
        SELECT product_id, base_price
        FROM read_parquet('{PRICES}')
        QUALIFY row_number() OVER (PARTITION BY product_id ORDER BY product_id) = 1;
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
                p.base_price,
                ROUND(p.base_price * l.quantity, 2) AS line_amount

            FROM orders_base ob
            INNER JOIN lines l USING (order_id)
            LEFT JOIN prices_dedup p ON p.product_id = l.product_id
            ORDER BY ob.order_timestamp, ob.store_id, ob.order_id, l.product_id

        ) TO '{OUT_PATH}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000);
    """)

    print("Wrote", OUT_PATH)
    con.close()

if __name__ == "__main__":
    main()
