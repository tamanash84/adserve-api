import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ---- import your engine ----
# adjust import path to match your project layout
from bandit_engine import RealTimeEngine


# ---------- Parquet streaming (sequential) ----------
def iter_parquet_rows(path: str, batch_size: int = 1024):
    """
    Stream parquet row-by-row without loading the full file into memory.
    Uses pyarrow if available; falls back to pandas read_parquet (loads full file).
    """
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            for row in df.itertuples(index=False):
                yield row
    except Exception:
        # fallback (not ideal for very large files)
        df = pd.read_parquet(path)
        for row in df.itertuples(index=False):
            yield row


# ---------- Simulation state ----------
@dataclass
class SimState:
    running: bool = False
    impressions: int = 0
    purchases: int = 0
    last_impression_ts: Optional[str] = None
    last_purchase_ts: Optional[str] = None
    tasks: list = field(default_factory=list)


STATE = SimState()
ENGINE: Optional[RealTimeEngine] = None


# ---------- FastAPI ----------
app = FastAPI(title="Bandit Test Harness")


class StartRequest(BaseModel):
    store_id: int = 1
    pos_parquet_path: str = "orders_prod_multistore_pos.parquet"
    impression_every_sec: int = 300          # 5 minutes
    purchase_sleep_sec: float = 0.25         # how fast to replay rows
    delta_seconds: int = 3600                # reward window
    use_xgb: bool = False                    # you said xgboost=False for now


class RecommendRequest(BaseModel):
    context: Dict[str, Any] = {}  # optional override context


@app.get("/status")
def status():
    return {
        "running": STATE.running,
        "impressions": STATE.impressions,
        "purchases": STATE.purchases,
        "last_impression_ts": STATE.last_impression_ts,
        "last_purchase_ts": STATE.last_purchase_ts,
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    global ENGINE
    if ENGINE is None:
        return {"error": "Engine not started. Call /start first."}

    # update impression timestamp each call
    ENGINE.ts_impr = datetime.now(timezone.utc)

    out = ENGINE.recommend(req.context)
    STATE.impressions += 1
    STATE.last_impression_ts = ENGINE.ts_impr.isoformat()
    return out


async def impressions_loop(every_sec: int):
    global ENGINE
    while STATE.running:
        ENGINE.ts_impr = datetime.now(timezone.utc)
        ENGINE.recommend(context={})
        STATE.impressions += 1
        STATE.last_impression_ts = ENGINE.ts_impr.isoformat()
        await asyncio.sleep(every_sec)


async def pos_loop(pos_path: str, sleep_sec: float):
    global ENGINE
    # Expected columns in parquet:
    # store_id, product_id, order_timestamp, quantity, base_price (or amount)
    for row in iter_parquet_rows(pos_path, batch_size=2048):
        if not STATE.running:
            break

        # adjust these field names to your parquet schema
        store_id = int(getattr(row, "store_id"))
        product_id = int(getattr(row, "product_id"))
        ts_val = getattr(row, "order_timestamp", None) or getattr(row, "ts_utc", None) or getattr(row, "timestamp", None)
        qty = int(getattr(row, "quantity", 1))
        amount = getattr(row, "amount", None)
        order_id = getattr(row, "order_id", None)

        # parse ts
        if isinstance(ts_val, str):
            ts = datetime.fromisoformat(ts_val)
        elif isinstance(ts_val, datetime):
            ts = ts_val
        else:
            ts = datetime.now(timezone.utc)

        ENGINE.reward(
            store_id=store_id,
            purchased_product_id=product_id,
            ts=ts,
            quantity=qty,
            amount=float(amount) if amount is not None else None,
            order_id=str(order_id) if order_id is not None else None,
        )

        STATE.purchases += 1
        STATE.last_purchase_ts = ts.isoformat()
        await asyncio.sleep(sleep_sec)


@app.post("/start")
async def start(req: StartRequest):
    global ENGINE

    if STATE.running:
        return {"status": "already running"}

    # Create engine (xgb disabled as you requested)
    ENGINE = RealTimeEngine(
        store_id=req.store_id,
        ts_impr=datetime.now(timezone.utc),
        window_sec=req.delta_seconds,
        use_xgbranker=req.use_xgb,
    )

    STATE.running = True
    STATE.impressions = 0
    STATE.purchases = 0
    STATE.last_impression_ts = None
    STATE.last_purchase_ts = None

    # launch background tasks
    t1 = asyncio.create_task(impressions_loop(req.impression_every_sec))
    t2 = asyncio.create_task(pos_loop(req.pos_parquet_path, req.purchase_sleep_sec))
    STATE.tasks = [t1, t2]

    return {"status": "started"}


@app.post("/stop")
async def stop():
    if not STATE.running:
        return {"status": "not running"}

    STATE.running = False
    # cancel tasks
    for t in STATE.tasks:
        t.cancel()
    STATE.tasks = []

    return {"status": "stopped"}