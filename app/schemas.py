from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime

# ---------- Request/Response models ----------

class RecommendRequest(BaseModel):
    context: Dict[str, Any]
    # Optional per-request overrides (do not mutate server defaults)
    served_policy: Optional[str] = Field(default=None, pattern="^(P1|P2|P3|P4)$")
    topK: Optional[int] = None
    randomK: Optional[int] = None


class RecommendResponse(BaseModel):
    request_id: str
    policy: str
    item: Optional[str]
    prob: float
    debug: Dict[str, Any]


class RewardRequest(BaseModel):
    request_id: str
    purchased_item: str
    revenue: float


class HealthResponse(BaseModel):
    status: str = "ok"


class ConfigState(BaseModel):
    served_policy: str
    topK: int
    randomK: int


class ConfigUpdate(BaseModel):
    served_policy: Optional[str] = Field(default=None, pattern="^(P1|P2|P3|P4)$")
    topK: Optional[int] = None
    randomK: Optional[int] = None
    
    
class LineItem(BaseModel):
    sku: str
    qty: float = Field(ge=0)
    unit_price: float = Field(ge=0)
    discount: float = Field(default=0.0)

class PurchaseEvent(BaseModel):
    # Preferred: request_id set when you render the product URL / QR / POS link with ?rid=...
    request_id: Optional[str] = None

    # Optional fallback identity (screen / device / session); not used directly below
    session_id: Optional[str] = None

    # If the POS can send the exact purchased item that should be attributed
    purchased_item: Optional[str] = None

    # POS basket
    lines: List[LineItem] = []

    # When the purchase happened
    occurred_at: datetime

    # Optional: when POS already computed revenue; if provided and positive, we trust it
    revenue_override: Optional[float] = Field(default=None)
