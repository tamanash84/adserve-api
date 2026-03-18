from pydantic import BaseModel
from typing import Dict, Optional

class RecommendRequest(BaseModel):
    context: Dict

class RecommendResponse(BaseModel):
    request_id: str
    item: str
    prob: float
    policy: str
    debug: Dict

class RewardRequest(BaseModel):
    request_id: str
    purchased_item: str
    revenue: float