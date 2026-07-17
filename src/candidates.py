import duckdb
from typing import Dict, List
from datetime import datetime
from ann_query import load_ann, query_similar
import pandas as pd

POS_DATA = r"../data/parquets/orders_prod_multistore_pos.parquet"
ANN_DIR = r"../data/ann"
W_HOURS = 3
B_HOURS = 12

def get_trending_items(
    t_now: datetime,
    top_trend: int = 10,
) -> pd.DataFrame:
    """
    For each store:
      - take top_trend trending products as seeds
      - collect ann_k neighbors per seed via ANN
      - candidates = seeds + neighbors (dedup, preserve order)
      - if no ANN neighbors found at all for the store -> fallback to top fallback_k trending

    Returns: {store_id: [candidate_product_ids...]}
    """
    sql_str = f"""
    WITH
    params AS (
      SELECT
        CAST(? AS TIMESTAMP) AS t_now,   
        INTERVAL {W_HOURS} HOUR AS w,
        INTERVAL {B_HOURS} HOUR AS b
    ),
    lh AS (
      SELECT
        p.store_id,
        p.product_id,
        SUM(p.quantity) AS qty_w,
        SUM(p.quantity * p.base_price) AS rev_w
      FROM read_parquet('{POS_DATA}') p
      CROSS JOIN params
      WHERE p.order_timestamp >= (t_now - w)
        AND p.order_timestamp <  t_now
      GROUP BY p.store_id, p.product_id
    ),
    base AS (
      SELECT
        p.store_id,
        p.product_id,
        SUM(p.quantity) / {B_HOURS}::DOUBLE AS qty_per_h_base,
        SUM(p.quantity * p.base_price) / {B_HOURS}::DOUBLE AS rev_per_h_base
      FROM read_parquet('{POS_DATA}') p
      CROSS JOIN params
      WHERE p.order_timestamp >= (t_now - (b + w))
        AND p.order_timestamp <  (t_now - w)
      GROUP BY p.store_id, p.product_id
    )
    SELECT
      lh.store_id,
      lh.product_id,
      lh.qty_w,
      lh.rev_w,
      COALESCE(base.qty_per_h_base, 0.0) AS qty_per_h_base,
      COALESCE(base.rev_per_h_base, 0.0) AS rev_per_h_base,
      (lh.qty_w + 1.0) / (COALESCE(base.qty_per_h_base, 0.0) * {W_HOURS} + 1.0) AS trend_qty_ratio,
      (lh.rev_w + 1.0) / (COALESCE(base.rev_per_h_base, 0.0) * {W_HOURS} + 1.0) AS trend_rev_ratio,
      LN(1.0 + lh.rev_w) * LN(
        (lh.rev_w + 1.0) / (COALESCE(base.rev_per_h_base, 0.0) * {W_HOURS} + 1.0)
      ) AS trend_score
    FROM lh
    LEFT JOIN base USING (store_id, product_id)
    WHERE lh.qty_w >= 3
    QUALIFY ROW_NUMBER() OVER (PARTITION BY store_id ORDER BY trend_score DESC) <= {top_trend}
    ORDER BY store_id, trend_score DESC
    """

    with duckdb.connect() as con:
        df = con.execute(sql_str, [t_now.replace(tzinfo=None)]).fetchdf()
    
    return df[["store_id","product_id","trend_score"]]


def enhance_with_ann(trending_df: pd.DataFrame,  
                     seed_k: int = 10, ann_k: int = 4) -> Dict[int, List[int]]:

    # Ensure stable order: assume trending_df already sorted, but keep input order per store
    # If not sorted, sort by your score column before calling this function.    
    
    df = (trending_df
           .sort_values(["store_id", "trend_score"], ascending=[True, False])
           .groupby("store_id", as_index=False)
           .head(seed_k)
          )

    store_candidates = {}    
   
    ann = load_ann(ANN_DIR)
    
    seeds = df["product_id"].unique()
    
    neighbors_all = {}
    for pid in seeds:
        try:
            # query_similar returns [(pid2, sim), ...]
            neigh = query_similar(ann, pid, topk=ann_k)
            
            neigh_ids = [pid2 for pid2, _ in neigh]       # keep only pid2
            neighbors_all[pid] = [pid] + neigh_ids

        except KeyError:
            # product_id not in ANN map; just skip this seed
            neighbors_all[pid] = [pid]    

    store_candidates = (
        df.groupby("store_id")["product_id"]
          .apply(lambda s: list(dict.fromkeys(
              [pid for seed in s.tolist()
                   for pid in neighbors_all.get(seed, [seed])]
          )))
          .to_dict()
    )

    return store_candidates

def build_store_candidates(t_now: datetime,
                           top_trend: int = 10,  
                           seed_k: int = 10, 
                           ann_k: int = 4) -> Dict[int, List[int]]:
    
    df = get_trending_items(t_now)
    x = enhance_with_ann(df)
    return x

################# Example run ###################    
# df = get_trending_items(datetime.now())
# x = enhance_with_ann(df)
# print(x[1])
