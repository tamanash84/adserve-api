import time
import shutil
import logging
import sys
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from datetime import date
from xgb_utils import score_ensemble, build_model, concat_hourlies_to_day, load_ensemble
from label_run import label_finalized_to_parquet
from bandit_learn import vw_learn_from_delta_parquet
from candidates import build_store_candidates 
from context import get_ctx_per_store, get_ctx_per_product 
from config import Paths
import pyarrow.parquet as pq

# Adjust import path to your project
from bandit_engine import BanditEngine
from bandit_policy import VwAdfXGBPolicy

# -----------------------
# Config
# -----------------------
POS_PATH = "../data/parquets/orders_prod_multistore_pos.parquet"
REWARD_MATCH_WINDOW = 3600

# -----------------------
# Simulation calendar
# -----------------------
SIM_START_DATE = "2026-01-10"   # inclusive
SIM_END_DATE   = "2026-01-17"   # exclusive (recommended)

ACTIVE_START_HOUR = 8         # 06:00
ACTIVE_END_HOUR   = 14        # 22:00 (end boundary)
NIGHTLY_TRAIN_HOUR = 1         # 01:00

TRAIN_LOOKBACK_DAYS = 7      # if you want rolling training window

# If you simulate impressions per store, list them here.
# If your existing script already discovers stores elsewhere, keep that logic instead.
STORE_IDS = [1,2]  # e.g. [1,2,3] or None to infer opportunistically

IMPRESSION_EVERY_SEC = 300        # 5 minutes
#DELTA_SECONDS = 3600              # expires window (for log field)
#USE_XGB = False                   # as requested

SLEEP_PER_POS_ROW = 0.0           # slow down if desired
BATCH_SIZE = 4096
SKIP_ROWS = 10_000                 # pyarrow batch size
MAX_POS_ROWS: Optional[int] = 50_000  # cap purchases

def _parse_date(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()

def _dt(d: date, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    # naive local datetime (no UTC conversion)
    return datetime(d.year, d.month, d.day, hh, mm, ss)

def _in_active_window(ts: datetime) -> bool:
    return (ACTIVE_START_HOUR <= ts.hour < ACTIVE_END_HOUR)

def _safe_int_list(seq):
    # DuckDB / logging safety: convert any numpy ints to python ints
    return [int(x) for x in list(seq)]

def _floor_to_hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)

def hourly_policy_update(policy, T: datetime):
    start = T - timedelta(hours=1)
    end = T
    delta_path = label_finalized_to_parquet(start_ts=start, end_ts=end, overwrite=True)
    learned = vw_learn_from_delta_parquet(policy, delta_path)
    return learned

def nightly_train_and_load(train_ts):
    # concatenate hourly reward matched training files
    try:
        #concat_hourlies_to_day(ts=train_ts)    
        model_paths = build_model(train_ts.date(), TRAIN_LOOKBACK_DAYS)
        return load_ensemble(model_paths)
    except Exception as e:
        print(f"Could not build or load model: {e}")
        #return None
    
def iter_pos_rows(path: str, batch_size: int = 4096, skip_rows: int = 0):
    """
    Stream parquet rows sequentially as they appear in the file.
    Assumes file is already in chronological order.
    Skips the first `skip_rows` rows.
    """
    pf = pq.ParquetFile(path)
    skipped = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        for row in df.itertuples(index=False):
            if skipped < skip_rows:
                skipped += 1
                continue
            yield row
            
def setup_logging(level=logging.INFO):
    logger = logging.getLogger("bandit-sim")
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Avoid duplicate handlers if reloaded
    if not logger.handlers:
        logger.addHandler(handler)

    logger.propagate = False
    return logger
           

def cleanup_wal_files(
    dirs,
    patterns=("*.jsonl", "*.parquet"),
    archive=False,
    archive_root="data/_archive",
    dry_run=False,
    remove_empty_dirs=True,
):
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    moved = 0
    deleted = 0

    for d in dirs:
        root = Path(d).resolve()
        if not root.exists():
            print(f"[cleanup] skip missing dir: {root}")
            continue

        matched_files = []
        for pat in patterns:
            matched_files.extend(root.rglob(pat))   # ✅ RECURSIVE

        if not matched_files:
            print(f"[cleanup] no files in {root}")
            continue

        if archive:
            sub = root.name or "wal"
            dest_dir = Path(archive_root, ts_tag, sub)
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)

            for f in matched_files:
                dest = dest_dir / f.name
                if dry_run:
                    print(f"[cleanup][dry] MOVE {f} -> {dest}")
                else:
                    shutil.move(str(f), dest)
                moved += 1

        else:
            for f in matched_files:
                if dry_run:
                    print(f"[cleanup][dry] DELETE {f}")
                else:
                    f.unlink(missing_ok=True)
                deleted += 1

        # Remove empty dirs bottom‑up
        if remove_empty_dirs:
            for p in sorted(root.rglob("*"), reverse=True):
                if p.is_dir() and not any(p.iterdir()):
                    if dry_run:
                        print(f"[cleanup][dry] RMDIR {p}")
                    else:
                        p.rmdir()

    print(f"[cleanup] done. moved={moved}, deleted={deleted}, archive={archive}, dry_run={dry_run}")


def run_simulation():
    log = logging.getLogger("bandit-sim")

    start_d = _parse_date(SIM_START_DATE)
    end_d   = _parse_date(SIM_END_DATE)

    # Create engine once; refresh model nightly
    policy = VwAdfXGBPolicy()
    engine = BanditEngine(policy)

    # Stream POS rows sequentially
    pos_iter = iter_pos_rows(POS_PATH, batch_size=BATCH_SIZE, skip_rows=SKIP_ROWS)

    # Pull first row
    next_pos = next(pos_iter, None)

    # Fast-forward POS iterator to the first timestamp >= simulation start at 06:00
    sim_start_ts = _dt(start_d, ACTIVE_START_HOUR)
    while next_pos is not None:
        ts = getattr(next_pos, "order_timestamp", None)
        if ts is None:
            raise ValueError("POS row missing 'order_timestamp' field")
        # Ensure ts is python datetime if it comes as numpy/pandas
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts >= sim_start_ts:
            break
        next_pos = next(pos_iter, None)
    
    xgb_ensemble = None

    # Main daily loop
    d = start_d
    while d < end_d:
        train_ts = _dt(d, NIGHTLY_TRAIN_HOUR) - timedelta(days=1)    # 01:00 of day d
        day_start = _dt(d, ACTIVE_START_HOUR)     # 06:00
        day_end   = _dt(d, ACTIVE_END_HOUR)       # 22:00
        next_hour_tick = _floor_to_hour(day_start) + timedelta(hours=1)

        log.info(f"=== Day {d.isoformat()} === train@{train_ts.time()} ops[{day_start.time()}-{day_end.time()}]")
        # 1) Nightly train (01:00) and load model for TODAY
        
        xgb_ensemble = nightly_train_and_load(train_ts)
        if xgb_ensemble:
            log.info(f"[nightly] xgboost training COMPLETED at {train_ts.isoformat()}")
            

        # 2) Run operations window: interleave impressions + purchases in time order
        next_impr_ts = day_start

        # If you simulate impressions per store, decide store list
        # - If STORE_IDS is set, use it.
        # - Otherwise, we will generate impressions only for stores seen in POS rows that day (lazy).
        stores_today = STORE_IDS[:] if STORE_IDS else None
        stores_seen_today = set()

        # Advance POS rows until we reach today’s window or beyond
        while next_pos is not None:
            ts = getattr(next_pos, "order_timestamp")
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()

            if ts >= day_start:
                break
            next_pos = next(pos_iter, None)

        # Event loop for this day
        while True:
            # Determine next POS timestamp (if it is within today’s active window)
            next_pos_ts = None
            next_pos_store = None
            if next_pos is not None:
                ts = getattr(next_pos, "order_timestamp")
                pid = getattr(next_pos, "product_id")
                qty = getattr(next_pos, "quantity")
                amount = getattr(next_pos, "line_amount")
                order_id = getattr(next_pos, "order_id")
                if hasattr(ts, "to_pydatetime"):
                    ts = ts.to_pydatetime()
                if day_start <= ts < day_end and _in_active_window(ts):
                    next_pos_ts = ts
                    next_pos_store = int(getattr(next_pos, "store_id", 0) or 0)

            # Determine next impression timestamp
            next_impr = next_impr_ts if next_impr_ts < day_end else None

            # Stop if no more events in this day
            if next_impr is None and next_pos_ts is None:
                break            
            
            # ---- figure out the next event time ----
            if next_pos_ts is None:
                next_event_ts = next_impr
            elif next_impr is None:
                next_event_ts = next_pos_ts
            else:
                next_event_ts = next_pos_ts if next_pos_ts <= next_impr else next_impr                
                
            # ---- hourly update of bandit policy with incremental learning ----
            while next_hour_tick <= day_end and next_event_ts >= next_hour_tick:
                learned = hourly_policy_update(policy, next_hour_tick)
                engine.reload_policy(policy)
                log.info(f"[hourly] bandit policy update COMPLETED at {next_hour_tick.isoformat()}, learned={learned}")
                next_hour_tick = next_hour_tick + timedelta(hours=1)

            # Pick earliest event
            if next_pos_ts is not None and (next_impr is None or next_pos_ts <= next_impr):
                # --- Process purchase event ---
                stores_seen_today.add(next_pos_store)

                # Feed purchase into engine (adjust to your engine API)
                # e.g. engine.on_purchase(store_id=next_pos_store, ts=next_pos_ts, row=next_pos)
                # If your engine expects the raw tuple row, pass next_pos directly.
                # log POS purchase
                record = {
                            "purchase_id": order_id,
                            "timestamp": next_pos_ts.isoformat(),
                            "store_id": next_pos_store,
                            "product_id": pid,
                            "quantity": qty,
                            "amount": amount
                         }
                
                BanditEngine.log_wal(Paths.POS_WAL, record, "purchases")

                # advance pos iterator
                next_pos = next(pos_iter, None)

                if SLEEP_PER_POS_ROW:
                    time.sleep(SLEEP_PER_POS_ROW)

            else:
                # --- Process impression event(s) ---
                # Determine which stores get an impression tick
                if stores_today is None:
                    # Lazy mode: if we haven't seen any store yet today, skip impressions until first store appears,
                    # OR you can default to last known store set if you have one.
                    active_stores = sorted(stores_seen_today) if stores_seen_today else []
                else:
                    active_stores = stores_today
                    
                cand_all_stores = build_store_candidates(next_impr_ts)
                shared_ctx_dicts = get_ctx_per_store(active_stores, next_impr_ts)

                for sid, shared_ctx_dict in zip(active_stores, shared_ctx_dicts):
                    # Recommend for this store at this timestamp (adjust to your engine API)
                   
                    # Items prior
                    candidates = cand_all_stores.get(sid)
                    if not candidates:
                        continue
                    
                    product_ctx_dicts = get_ctx_per_product(candidates)                             
                    xgb_context = [{**shared_ctx_dict, **ctx} for ctx in product_ctx_dicts]      
                    
                    _, probs_mean, probs_var = score_ensemble(xgb_ensemble, xgb_context)        
                    
                    K = len(candidates)
                    order = np.argsort(-probs_mean)  # descending XGB score
                    
                    # add rank (based on descending xgb score) as context
                    for rank, idx in enumerate(order):
                        product_ctx_dicts[idx]["rank_norm"] = rank / max(K - 1, 1)

                    for idx in range(0, K):      
                        product_ctx_dicts[idx]["xgb_prob_mean"] = probs_mean[idx]
                        product_ctx_dicts[idx]["xgb_prob_var"] = probs_var[idx]
                    
                    # featues to be used in bandit         
                    keep_action_feats = ["rank_norm", "xgb_prob_mean", "xgb_prob_var"]
                    keep_shared_feats = ["store_id", "hod", "dow", "month"]
                    
                    action_feats_bandit = pd.DataFrame(product_ctx_dicts)[keep_action_feats].to_dict(orient="records")
                    shared_feats_bandit = {k:shared_ctx_dict[k]for k in keep_shared_feats}
                    
                    impr_id = str(uuid.uuid4())
                    out = engine.recommend(shared_feats_bandit, action_feats_bandit, candidates)
                    
                    served_idx = int(out["chosen_index"])

                    record = {
                                "impression_id": impr_id,
                                "store_id": sid,
                                "timestamp": next_impr_ts.isoformat(),
                                "expires_at": (next_impr_ts + timedelta(seconds=REWARD_MATCH_WINDOW)).isoformat(),
                                "policy_name": out["policy_name"],
                                "chosen_index": served_idx,
                                "propensity": out["propensity"],
                                "pid_shown": int(out["pid_shown"]),
                                "expected_rewards": [float(i) for i in probs_mean],
                                "xgb_prob_mean": float(probs_mean[served_idx]),
                                "xgb_prob_var": float(probs_var[served_idx]),
                                "comment": out["comment"],    
                                #"reward": None
                            } 
                    
                    record = {**record, **xgb_context[served_idx]}                    
                   
                    BanditEngine.log_wal(Paths.BANDIT_WAL, record, "impressions")

                # Next impression tick
                next_impr_ts = next_impr_ts + timedelta(seconds=IMPRESSION_EVERY_SEC)

        # ------- End-of-day bookkeeping --------- 
        concat_hourlies_to_day(ts=day_end)
        policy_path = Paths.VW_POLICY / f"vw_policy_{day_end.date().isoformat()}.bin"
        policy.vw.save(policy_path)               
        log.info(f"End day {d.isoformat()} (stores_seen={len(stores_seen_today)})")

        d = d + timedelta(days=1)

    log.info("Simulation complete.")


if __name__ == "__main__":
    log = setup_logging(logging.INFO)

    log.info("Cleaning up old WAL files (archive mode)")
    cleanup_wal_files(
                        dirs=["../data/bandit/wal",
                              "../data/bandit/training",
                              "../data/pos/wal",
                              "../data/pos/purchases",
                              "../model/xgboost",
                              "../model/vw"]
                     )   
    
    run_simulation()