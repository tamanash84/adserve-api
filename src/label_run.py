# label_run.py
import os
import duckdb
from datetime import datetime
from config import Paths

_TS_FMT = "%Y-%m-%dT%H:%M:%S"  # adjust if you have milliseconds

def label_finalized_to_parquet(
    *,
    start_ts: datetime,
    end_ts: datetime,
    overwrite: bool = True,
) -> str:
    """
    Build a *delta* labeled parquet for the slice (start_ts, end_ts]:
      impressions with expires_at in (start_ts, end_ts],
    reward = 1 if a matching purchase of pid_shown exists within [ts_impr, expires_at].

    Output:
      out_dir/date=YYYY-MM-DD/hour=HH/train.parquet
    """
    impr_wal_dir = Paths.BANDIT_WAL
    pos_wal_dir = Paths.POS_WAL
    out_dir = Paths.BANDIT_TRAIN
    
    if end_ts <= start_ts:
        raise ValueError("end_ts must be after start_ts")

    day_str = end_ts.date().isoformat()      # hour bucket belongs to end_ts day
    hh = f"{end_ts.hour:02d}"
    out_path_dir = out_dir / f"{out_dir}/date={day_str}/hour={hh}"
    os.makedirs(out_path_dir, exist_ok=True)
    out_path = out_path_dir / "train.parquet"

    if (not overwrite) and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    impr_path = impr_wal_dir / f"impressions_{day_str}.jsonl"
    pos_path  = pos_wal_dir / f"purchases_{day_str}.jsonl"

    # If the WAL file for that day doesn't exist yet, just create nothing.
    if not os.path.exists(impr_path) or os.path.getsize(impr_path) == 0:
        return out_path  # caller can treat missing/empty file as "learn 0"

    con = duckdb.connect()
    con.execute(f"""
        COPY (
            WITH 
            impr_raw AS (
            SELECT
                * REPLACE (
                    CAST(impression_id AS VARCHAR) AS impression_id,
                    CAST(chosen_index AS INTEGER) AS chosen_index,
                    CAST(store_id AS INTEGER) AS store_id,
                    CAST(pid_shown AS BIGINT) AS pid_shown,
                    CAST(comment AS VARCHAR) AS comment
                ),
                STRPTIME(timestamp, '{_TS_FMT}') AS ts_impr,
                STRPTIME(expires_at, '{_TS_FMT}') AS expires_at_ts
                FROM read_json_auto('{impr_path.as_posix()}')
            ),
            impr AS (
                SELECT *
                FROM impr_raw
                WHERE expires_at_ts >  CAST('{start_ts}' AS TIMESTAMP)
                  AND expires_at_ts <= CAST('{end_ts}'   AS TIMESTAMP)
            ),
            pos_raw AS (
                SELECT
                    CAST(store_id AS INTEGER) AS store_id,
                    CAST(product_id AS BIGINT) AS product_id,
                    CAST(STRPTIME(timestamp, '{_TS_FMT}') AS TIMESTAMP) AS ts_pos
                FROM read_json_auto('{pos_path.as_posix()}')
            ),
            pos AS (
                SELECT *
                FROM pos_raw
                WHERE ts_pos <= CAST('{end_ts}' AS TIMESTAMP)
            )
            SELECT
                i.*,
                CASE WHEN EXISTS (
                    SELECT 1
                    FROM pos p
                    WHERE p.store_id = i.store_id
                      AND p.product_id = i.pid_shown
                      AND p.ts_pos >= i.ts_impr
                      AND p.ts_pos <= i.expires_at_ts
                )
                THEN 1 ELSE 0 END AS reward
            FROM impr i
        )
        TO '{out_path.as_posix()}' (FORMAT PARQUET)
    """)
    con.close()

    return out_path