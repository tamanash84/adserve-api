from collections import defaultdict, deque
from datetime import datetime, timedelta

class AttributionStore:
    def __init__(self, window_sec=6*3600):
        self.window_sec = window_sec
        
        # Primary index: request_id -> impression dict
        self.by_request = {}  # { str: {request_id, ts, expires_at, policy, item, prob, ...} }

        
        # Secondary index: item -> deque of recent impressions (for optional fallback)
        # Each deque holds impressions for that item, ordered by arrival time.
        self.by_item = defaultdict(deque)  # { item: deque([impression_dict, ...]) }


    def add(self, impression):

        """
        impression = {
            "request_id": "...",     # UUID
            "policy": "P1",
            "item": "Milk 1L",
            "prob": 0.23,
            "context": {...},
            "ts": datetime.utcnow(),
            # expires_at may be derived here or in caller:
            # "expires_at": impression["ts"] + timedelta(seconds=self.window_sec)
        }
        """

        rid = impression["request_id"]
        item = impression["item"]

        self.by_request[rid] = impression
        self.by_item[item].append(impression)

    
    def match(self, request_id: str, purchased_item: str, revenue: float):
        imp = self.by_request.get(request_id)
        if not imp:
            # Fall back to item-based (optional); see below.
            return {"matched": False}
    
        # Check expiry
        if datetime.utcnow() > imp["expires_at"]:
            return {"matched": False, "reason": "expired"}
    
        # Optional sanity: ensure purchased item matches served item (can be relaxed)
        if purchased_item != imp["item"]:
            # You can decide to reject, or accept if your policy allows.
            # Here we accept only if item matches:
            return {"matched": False, "reason": "item_mismatch"}
    
        # Mark the impression as matched and attach reward
        imp["matched"] = True
        imp["reward"]  = float(revenue)
        imp["ts_reward"] = datetime.utcnow()
    
        # Optional: you may remove it from by_item’s deque here (O(n) worst-case for that deque),
        # or leave it for the sweeper to prune.
        return {
            "matched": True,
            "policy": imp["policy"],
            "item": imp["item"],
            "reward": imp["reward"],
            }
    
    # Optional fallback: match by item (only if you cannot use request_id)
    def match_by_item(self, purchased_item: str, revenue: float):
        dq = self.by_item.get(purchased_item)
        if not dq:
            return {"matched": False}
    
        now = datetime.utcnow()
    
        # Walk from the right (newest → oldest) to find first open, non-expired impression
        for i in range(len(dq) - 1, -1, -1):
            imp = dq[i]
            if imp.get("matched"):
                continue
            if now > imp["expires_at"]:
                continue  # let sweeper prune later
            # Found a match:
            imp["matched"] = True
            imp["reward"]  = float(revenue)
            imp["ts_reward"] = now
            return {
                "matched": True,
                "policy": imp["policy"],
                "item": imp["item"],
                "reward": imp["reward"],
            }
    
        return {"matched": False}
    
    def sweep_expired(self):
        now = datetime.utcnow()
        for item, dq in list(self.by_item.items()):
            while dq and (dq[0].get("matched") or dq[0]["expires_at"] < now):
                dq.popleft()
            if not dq:
                del self.by_item[item]