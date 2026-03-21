import os
import asyncio
import contextlib
from contextlib import asynccontextmanager
from math import isfinite
from typing import List

from fastapi import FastAPI, HTTPException
from .bandit_engine import RealTimeEngine
from .schemas import RecommendRequest, RecommendResponse, RewardRequest, HealthResponse, ConfigState, ConfigUpdate, LineItem, PurchaseEvent

SWEEP_INTERVAL_SEC = 60  # adjust to traffic & window size


# ---------- App & lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Default policy and knobs can be set via env (optional)
    default_policy = os.getenv("SERVED_POLICY", "P1")
    topK = int(os.getenv("TOPK", "10"))
    randomK = int(os.getenv("RANDOMK", "5"))

    engine = RealTimeEngine(
        served_policy=default_policy,
        topK=topK,
        randomK=randomK,
    )
    app.state.engine = engine

    async def _sweeper():
        # lightweight loop; stops when task is cancelled on shutdown
        while True:
            try:
                
                # learn 0 on expired, unmatched impressions
                engine._expire_and_learn_zero()     # <-- NEW

                # Assuming your AttributionStore implements this method
                engine.attr.sweep_expired()
            except Exception as e:
                # optional: log, but never crash the loop
                print(f"[sweeper] error: {e}")
            await asyncio.sleep(SWEEP_INTERVAL_SEC)

    sweeper_task = asyncio.create_task(_sweeper())
    try:
        yield
    finally:
        # graceful shutdown
        sweeper_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweeper_task


app = FastAPI(lifespan=lifespan)


# ---------- Endpoints ----------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@app.get("/config", response_model=ConfigState)
def get_config():
    eng: RealTimeEngine = app.state.engine
    return ConfigState(
        served_policy=eng.served_policy,
        topK=eng.topK,
        randomK=eng.randomK,
    )


@app.post("/config", response_model=ConfigState)
def update_config(cfg: ConfigUpdate):
    eng: RealTimeEngine = app.state.engine
    if cfg.served_policy:
        eng.served_policy = cfg.served_policy.upper()
    if cfg.topK is not None:
        if cfg.topK <= 0:
            raise HTTPException(status_code=400, detail="topK must be > 0")
        eng.topK = int(cfg.topK)
    if cfg.randomK is not None:
        if cfg.randomK <= 0:
            raise HTTPException(status_code=400, detail="randomK must be > 0")
        eng.randomK = int(cfg.randomK)
    return ConfigState(
        served_policy=eng.served_policy,
        topK=eng.topK,
        randomK=eng.randomK,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest):
    eng: RealTimeEngine = app.state.engine

    # Per-request temporary overrides (do not persist on engine)
    old_policy, old_topK, old_randomK = eng.served_policy, eng.topK, eng.randomK
    try:
        if payload.served_policy:
            eng.served_policy = payload.served_policy.upper()
        if payload.topK is not None:
            if payload.topK <= 0:
                raise HTTPException(status_code=400, detail="topK must be > 0")
            eng.topK = int(payload.topK)
        if payload.randomK is not None:
            if payload.randomK <= 0:
                raise HTTPException(status_code=400, detail="randomK must be > 0")
            eng.randomK = int(payload.randomK)

        out = eng.recommend(payload.context)
        # out is already in the expected shape from RealTimeEngine
        return RecommendResponse(**out)

    finally:
        # Restore engine defaults after the call
        eng.served_policy, eng.topK, eng.randomK = old_policy, old_topK, old_randomK


@app.post("/reward")
def reward(payload: RewardRequest):
    eng: RealTimeEngine = app.state.engine
    return eng.reward(payload.request_id, payload.purchased_item, payload.revenue)


def _compute_revenue_for_item(lines: List[LineItem], sku: str) -> float:
    rev = 0.0
    for li in lines:
        if li.sku == sku:
            rev += (li.unit_price - li.discount) * li.qty
    return float(rev)


@app.post("/event/purchase")
def event_purchase(evt: PurchaseEvent):
    """
    POS webhook: converts a checkout into a bandit reward.

    Matching priority:
      1) request_id present  -> attribute to that exact impression
      2) no request_id       -> attempt item-based matching per line (best-effort)
    """
    eng: RealTimeEngine = app.state.engine  # your RealTimeEngine (has .attr store)  
    store = eng.attr        # AttributionStore with by_request, match, match_by_item  

    # ------------------------------
    # Path A: Deterministic by request_id (recommended)
    # ------------------------------
    if evt.request_id:
        imp = store.by_request.get(evt.request_id)  # impression dict  
        if not imp:
            raise HTTPException(status_code=404, detail="unknown request_id")

        target_item = evt.purchased_item or imp["item"]
        # Use provided revenue if POS sent it and it's finite/positive; else compute from lines
        if evt.revenue_override is not None and isfinite(evt.revenue_override) and evt.revenue_override >= 0:
            revenue = float(evt.revenue_override)
        else:
            revenue = _compute_revenue_for_item(evt.lines, target_item)

        # Perform attribution (will also fail if expired, mismatch, etc.)  
        result = eng.reward(evt.request_id, target_item, revenue)
        return {
            "matched": bool(result.get("matched")),
            "policy": result.get("policy"),
            "item": result.get("item"),
            "reward": result.get("reward", 0.0),
            "reason": result.get("reason"),
        }

    # ------------------------------
    # Path B: Fallback (no request_id) — match by purchased item(s) within window
    #         This may match multiple items if basket contains several served SKUs recently.
    # ------------------------------
    matched_results = []
    for li in evt.lines:
        revenue = float((li.unit_price - li.discount) * li.qty)
        # Try best-effort item-based attribution 
        res = store.match_by_item(li.sku, revenue)
        if res.get("matched"):
            matched_results.append({
                "policy": res.get("policy"),
                "item": res.get("item"),
                "reward": res.get("reward", 0.0),
            })

    return {
        "matched": len(matched_results) > 0,
        "results": matched_results,
    }

