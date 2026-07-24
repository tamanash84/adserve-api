import time
import shutil
import logging
import uuid
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Optional, List
from dataclasses import dataclass

from xgb_utils import score_ensemble, build_model, concat_hourlies_to_day, load_ensemble
from label_run import label_finalized_to_parquet
from bandit_learn import vw_learn_from_delta_parquet
from candidates import build_store_candidates
from context import get_ctx_per_store, get_ctx_per_product
from config import Paths

from bandit_engine import BanditEngine
from bandit_policy import VwAdfXGBPolicy


@dataclass
class SimulationConfig:
    pos_path: str = "../data/parquets/orders_prod_multistore_pos.parquet"
    reward_match_window: int = 3600

    sim_start_date: str = "2026-01-10"
    sim_end_date: str = "2026-01-17"

    active_start_hour: int = 8
    active_end_hour: int = 14
    nightly_train_hour: int = 1

    train_lookback_days: int = 7
    store_ids: Optional[List[int]] = None

    impression_every_sec: int = 300
    sleep_per_pos_row: float = 0.0
    batch_size: int = 4096
    skip_rows: int = 0
    max_pos_rows: Optional[int] = 50_000

    cleanup_before_run: bool = True
    archive_cleanup: bool = False
    dry_run_cleanup: bool = False


def _parse_date(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def _dt(d: date, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, ss)


def _in_active_window(ts: datetime, cfg: SimulationConfig) -> bool:
    return cfg.active_start_hour <= ts.hour < cfg.active_end_hour


def _floor_to_hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


def iter_pos_rows(path: str, batch_size: int = 4096, skip_rows: int = 0, max_rows: Optional[int] = None):
    """
    Stream parquet rows sequentially as they appear in the file.
    Avoids loading the entire POS parquet into memory.
    """
    pf = pq.ParquetFile(path)
    skipped = 0
    yielded = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()

        for row in df.itertuples(index=False):
            if skipped < skip_rows:
                skipped += 1
                continue

            if max_rows is not None and yielded >= max_rows:
                return

            yielded += 1
            yield row


def hourly_policy_update(policy, T: datetime, log: Optional[logging.Logger] = None):
    log = log or logging.getLogger("bandit-sim")

    start = T - timedelta(hours=1)
    end = T

    try:
        delta_path = label_finalized_to_parquet(
            start_ts=start,
            end_ts=end,
            overwrite=True,
        )
    except Exception as e:
        log.exception(f"[hourly] labeling failed for {start} to {end}: {e}")
        return False

    if delta_path is None:
        log.warning(f"[hourly] no delta parquet created for {start} to {end}")
        return False

    delta_path = Path(delta_path)

    if not delta_path.exists():
        log.warning(f"[hourly] delta parquet missing: {delta_path}")
        return False

    learned = vw_learn_from_delta_parquet(policy, delta_path)
    return learned


def nightly_train_and_load(train_ts: datetime, cfg: SimulationConfig, log: logging.Logger):
    try:
        model_paths = build_model(train_ts.date(), cfg.train_lookback_days)
        return load_ensemble(model_paths)

    except FileNotFoundError as e:
        log.warning(f"[nightly] Cold start: {e}")
        return None

    except Exception as e:
        log.exception(f"Could not build or load model: {e}")
        return None


def cleanup_wal_files(
    dirs,
    patterns=("*.jsonl", "*.parquet", "*.bin"),
    archive=False,
    archive_root="../data/_archive",
    dry_run=False,
    remove_empty_dirs=True,
    log: Optional[logging.Logger] = None,
):
    log = log or logging.getLogger("bandit-sim")

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    moved = 0
    deleted = 0

    for d in dirs:
        root = Path(d).resolve()

        if not root.exists():
            log.info(f"[cleanup] skip missing dir: {root}")
            continue

        matched_files = []
        for pat in patterns:
            matched_files.extend(root.rglob(pat))

        if not matched_files:
            log.info(f"[cleanup] no files in {root}")
            continue

        if archive:
            sub = root.name or "wal"
            dest_dir = Path(archive_root, ts_tag, sub)

            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)

            for f in matched_files:
                dest = dest_dir / f.name
                if dry_run:
                    log.info(f"[cleanup][dry] MOVE {f} -> {dest}")
                else:
                    shutil.move(str(f), dest)
                moved += 1

        else:
            for f in matched_files:
                if dry_run:
                    log.info(f"[cleanup][dry] DELETE {f}")
                else:
                    f.unlink(missing_ok=True)
                deleted += 1

        if remove_empty_dirs:
            for p in sorted(root.rglob("*"), reverse=True):
                if p.is_dir() and not any(p.iterdir()):
                    if dry_run:
                        log.info(f"[cleanup][dry] RMDIR {p}")
                    else:
                        p.rmdir()

    log.info(
        f"[cleanup] done. moved={moved}, deleted={deleted}, "
        f"archive={archive}, dry_run={dry_run}"
    )


def run_simulation(cfg: SimulationConfig, log: Optional[logging.Logger] = None):
    log = log or logging.getLogger("bandit-sim")

    start_d = _parse_date(cfg.sim_start_date)
    end_d = _parse_date(cfg.sim_end_date)

    policy = VwAdfXGBPolicy()
    engine = BanditEngine(policy)

    pos_iter = iter_pos_rows(
        cfg.pos_path,
        batch_size=cfg.batch_size,
        skip_rows=cfg.skip_rows,
        max_rows=cfg.max_pos_rows,
    )

    next_pos = next(pos_iter, None)

    sim_start_ts = _dt(start_d, cfg.active_start_hour)
    

    while next_pos is not None:
        ts = getattr(next_pos, "order_timestamp", None)

        if ts is None:
            raise ValueError("POS row missing 'order_timestamp' field")

        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        if ts >= sim_start_ts:
            break

        next_pos = next(pos_iter, None)

    d = start_d

    while d < end_d:
        train_ts = _dt(d, cfg.nightly_train_hour) - timedelta(days=1)
        day_start = _dt(d, cfg.active_start_hour)
        day_end = _dt(d, cfg.active_end_hour)

        next_hour_tick = _floor_to_hour(day_start) + timedelta(hours=1)

        log.info(
            f"=== Day {d.isoformat()} === "
            f"train@{train_ts.time()} "
            f"ops[{day_start.time()}-{day_end.time()}]"
        )

        xgb_ensemble = nightly_train_and_load(train_ts, cfg, log)

        if xgb_ensemble:
            log.info(f"[nightly] xgboost training COMPLETED at {train_ts.isoformat()}")
        else:
            log.warning("[nightly] xgboost ensemble unavailable. Simulation may fail if scoring needs it.")

        next_impr_ts = day_start

        stores_today = cfg.store_ids[:] if cfg.store_ids else None
        stores_seen_today = set()

        while next_pos is not None:
            ts = getattr(next_pos, "order_timestamp")

            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()

            if ts >= day_start:
                break

            next_pos = next(pos_iter, None)

        while True:
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
   
                if day_start <= ts < day_end and _in_active_window(ts, cfg):
                    next_pos_ts = ts
                    next_pos_store = int(getattr(next_pos, "store_id", 0) or 0)

            next_impr = next_impr_ts if next_impr_ts < day_end else None

            if next_impr is None and next_pos_ts is None:
                break

            if next_pos_ts is None:
                next_event_ts = next_impr
            elif next_impr is None:
                next_event_ts = next_pos_ts
            else:
                next_event_ts = next_pos_ts if next_pos_ts <= next_impr else next_impr

            while next_hour_tick <= day_end and next_event_ts >= next_hour_tick:
                learned = hourly_policy_update(policy, next_hour_tick, log)
                engine.reload_policy(policy)

                log.info(
                    f"[hourly] bandit policy update COMPLETED at "
                    f"{next_hour_tick.isoformat()}, learned={learned}"
                )

                next_hour_tick = next_hour_tick + timedelta(hours=1)

            if next_pos_ts is not None and (next_impr is None or next_pos_ts <= next_impr):
                stores_seen_today.add(next_pos_store)

                record = {
                    "purchase_id": order_id,
                    "timestamp": next_pos_ts.isoformat(),
                    "store_id": next_pos_store,
                    "product_id": pid,
                    "quantity": qty,
                    "amount": amount,
                }

                BanditEngine.log_wal(Paths.POS_WAL, record, "purchases")
                next_pos = next(pos_iter, None)

                if cfg.sleep_per_pos_row:
                    time.sleep(cfg.sleep_per_pos_row)

            else:
                if stores_today is None:
                    active_stores = sorted(stores_seen_today) if stores_seen_today else []
                else:
                    active_stores = stores_today

                cand_all_stores = build_store_candidates(next_impr_ts)
                shared_ctx_dicts = get_ctx_per_store(active_stores, next_impr_ts)

                for sid, shared_ctx_dict in zip(active_stores, shared_ctx_dicts):
                    candidates = cand_all_stores.get(sid)

                    if not candidates:
                        continue

                    product_ctx_dicts = get_ctx_per_product(candidates)
                    xgb_context = [
                        {**shared_ctx_dict, **ctx}
                        for ctx in product_ctx_dicts
                    ]

                    if xgb_ensemble is None:
                        K = len(candidates)
                        probs_mean = np.full(K, 0.01, dtype=float)
                        probs_var = np.zeros(K, dtype=float)
                    else:
                        _, probs_mean, probs_var = score_ensemble(xgb_ensemble, xgb_context)

                    K = len(candidates)
                    order = np.argsort(-probs_mean)

                    for rank, idx in enumerate(order):
                        product_ctx_dicts[idx]["rank_norm"] = rank / max(K - 1, 1)

                    for idx in range(K):
                        product_ctx_dicts[idx]["xgb_prob_mean"] = probs_mean[idx]
                        product_ctx_dicts[idx]["xgb_prob_var"] = probs_var[idx]

                    keep_action_feats = [
                        "rank_norm",
                        "xgb_prob_mean",
                        "xgb_prob_var",
                    ]

                    keep_shared_feats = [
                        "store_id",
                        "hod",
                        "dow",
                        "month",
                    ]

                    action_feats_bandit = (
                        pd.DataFrame(product_ctx_dicts)[keep_action_feats]
                        .to_dict(orient="records")
                    )

                    shared_feats_bandit = {
                        k: shared_ctx_dict[k]
                        for k in keep_shared_feats
                    }

                    impr_id = str(uuid.uuid4())

                    out = engine.recommend(
                        shared_feats_bandit,
                        action_feats_bandit,
                        candidates,
                    )

                    served_idx = int(out["chosen_index"])

                    record = {
                        "impression_id": impr_id,
                        "store_id": sid,
                        "timestamp": next_impr_ts.isoformat(),
                        "expires_at": (
                            next_impr_ts + timedelta(seconds=cfg.reward_match_window)
                        ).isoformat(),
                        "policy_name": out["policy_name"],
                        "chosen_index": served_idx,
                        "propensity": out["propensity"],
                        "pid_shown": int(out["pid_shown"]),
                        "expected_rewards": [float(i) for i in probs_mean],
                        "xgb_prob_mean": float(probs_mean[served_idx]),
                        "xgb_prob_var": float(probs_var[served_idx]),
                        "comment": out["comment"],
                    }

                    record = {**record, **xgb_context[served_idx]}

                    BanditEngine.log_wal(Paths.BANDIT_WAL, record, "impressions")

                next_impr_ts = next_impr_ts + timedelta(seconds=cfg.impression_every_sec)

        concat_hourlies_to_day(ts=day_end)

        policy_path = Paths.VW_POLICY / f"vw_policy_{day_end.date().isoformat()}.bin"
        policy.vw.save(str(policy_path))

        log.info(f"[eod] saved VW policy: {policy_path}")
        log.info(f"End day {d.isoformat()} stores_seen={len(stores_seen_today)}")

        d = d + timedelta(days=1)

    log.info("Simulation complete.")
    
    

# # run_simulation_script.py

# import logging


# def setup_console_logger() -> logging.Logger:
#     logger = logging.getLogger("bandit-sim")
#     logger.setLevel(logging.INFO)
#     logger.handlers.clear()
#     logger.propagate = False

#     handler = logging.StreamHandler()
#     handler.setLevel(logging.INFO)

#     formatter = logging.Formatter(
#         fmt="%(asctime)s | %(levelname)-7s | %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )

#     handler.setFormatter(formatter)
#     logger.addHandler(handler)

#     return logger


# def main() -> None:
#     log = setup_console_logger()

#     # ------------------------------------------------------------
#     # Hardcoded test configuration
#     # ------------------------------------------------------------

#     pos_path = Path(
#         r"C:\Users\NH61FL\ML_Data\adserve_api\data\parquets\orders_prod_multistore_pos.parquet"
#     ).resolve()

#     if not pos_path.exists():
#         raise FileNotFoundError(f"POS parquet not found: {pos_path}")

#     cfg = SimulationConfig(
#         pos_path=str(pos_path),

#         sim_start_date="2026-01-10",
#         sim_end_date="2026-01-11",

#         active_start_hour=8,
#         active_end_hour=14,
#         nightly_train_hour=1,

#         train_lookback_days=7,
#         store_ids=[1, 2],

#         impression_every_sec=300,
#         reward_match_window=3600,

#         batch_size=4096,

#         # For debugging, avoid skipping relevant POS rows
#         skip_rows=0,

#         # None means no cap
#         max_pos_rows=None,

#         # For first debugging run, I suggest False
#         cleanup_before_run=True,
#         archive_cleanup=False,
#         dry_run_cleanup=False,
#     )

#     # ------------------------------------------------------------
#     # Logging config summary
#     # ------------------------------------------------------------

#     log.info("Starting standalone simulation script")
#     log.info(f"POS path: {cfg.pos_path}")
#     log.info(f"POS exists: {Path(cfg.pos_path).exists()}")
#     log.info(f"Simulation dates: {cfg.sim_start_date} to {cfg.sim_end_date}")
#     log.info(f"Active window: {cfg.active_start_hour}:00 to {cfg.active_end_hour}:00")
#     log.info(f"Nightly train hour: {cfg.nightly_train_hour}:00")
#     log.info(f"Store IDs: {cfg.store_ids}")
#     log.info(f"Skip rows: {cfg.skip_rows}")
#     log.info(f"Max POS rows: {cfg.max_pos_rows}")
#     log.info(f"Cleanup before run: {cfg.cleanup_before_run}")

#     # ------------------------------------------------------------
#     # Optional cleanup
#     # ------------------------------------------------------------

#     if cfg.cleanup_before_run:
#         log.info("Running cleanup before simulation")

#         cleanup_wal_files(
#             dirs=[
#                 "../data/bandit/wal",
#                 "../data/bandit/training",
#                 "../data/pos/wal",
#                 "../data/pos/purchases",
#                 "../model/xgboost",
#                 "../model/vw",
#             ],
#             archive=cfg.archive_cleanup,
#             dry_run=cfg.dry_run_cleanup,
#             log=log,
#         )

#     # ------------------------------------------------------------
#     # Run simulation directly
#     # ------------------------------------------------------------

#     run_simulation(cfg, log)

#     log.info("Standalone simulation script complete")


# if __name__ == "__main__":
#     main()