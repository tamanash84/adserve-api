import asyncio
import contextlib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .bandit_engine import RealTimeEngine

SWEEP_INTERVAL_SEC = 60  # adjust to traffic & window size

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = RealTimeEngine()
    app.state.engine = engine

    async def _sweeper():
        # lightweight loop; stops when task is cancelled on shutdown
        while True:
            try:
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

@app.post("/recommend")
def recommend(payload: dict):
    return app.state.engine.recommend(payload["context"])

@app.post("/reward")
def reward(payload: dict):
    return app.state.engine.reward(
        payload["request_id"], payload["purchased_item"], payload["revenue"]
    )

@app.get("/health")
def health():
    return {"status": "ok"}
