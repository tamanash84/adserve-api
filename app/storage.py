from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, Optional, Set


def _to_dt(val) -> datetime:
    """Convert ISO string or datetime to naive-UTC datetime for internal comparisons."""
    if isinstance(val, datetime):
        return val  # assume already UTC-naive; keep consistent across codebase
    if isinstance(val, str):
        # Accept 'Z' or '+00:00' and normalize
        s = val.replace("Z", "+00:00")
        # fromisoformat with offset returns aware; make naive-UTC
        dt = datetime.fromisoformat(s)
        # If aware, convert to naive-UTC by dropping tzinfo after UTC normalization
        if dt.tzinfo:
            dt = (dt.astimezone(tz=None).replace(tzinfo=None))
        return dt
    raise TypeError(f"Unsupported datetime value: {type(val)}")


class AttributionStore:
    """
    In-memory store for impression → reward attribution.

    Indexes:
      - by_request: request_id -> impression dict
      - by_item:    item -> deque[impression] (time-ordered)
      - by_session: session_id -> deque[impression] (time-ordered)

    Idempotency:
      - seen_events: set of event_id to drop duplicates

    Learning hook:
      - on_match: Optional[Callable[[dict], None]]  # called with impression dict upon match
    """

    def __init__(self, window_sec: int = 6 * 3600, on_match: Optional[Callable[[dict], None]] = None):
        self.window_sec = int(window_sec)
        self.on_match = on_match

        self.by_request: Dict[str, Dict[str, Any]] = {}
        self.by_item: Dict[str, Deque[Dict[str, Any]]] = defaultdict(deque)
        self.by_session: Dict[str, Deque[Dict[str, Any]]] = defaultdict(deque)
        self.seen_events: Set[str] = set()

    # -----------------------------
    # Impressions
    # -----------------------------
    def add(self, impression: Dict[str, Any]) -> None:
        """
        Expected minimum fields:
          {
            "request_id": str,
            "policy": "P1|P2|P3|P4",
            "item": str,
            "prob": float,
            "context": dict,
            "ts": datetime,                # naive UTC
            "expires_at": datetime,        # naive UTC
            # optional:
            "session_id": str,             # device/screen/session ID (recommended)
          }
        """
        rid = impression["request_id"]
        itm = impression["item"]

        # Normalize datetimes in case caller sent strings
        impression["ts"] = _to_dt(impression["ts"])
        impression["expires_at"] = _to_dt(impression["expires_at"])

        self.by_request[rid] = impression
        self.by_item[itm].append(impression)

        sess = impression.get("session_id")
        if sess:
            self.by_session[sess].append(impression)

    # -----------------------------
    # Reward matching (request_id preferred)
    # -----------------------------
    def match(self, request_id: str, purchased_item: str, revenue: float, event_id: Optional[str] = None) -> Dict[str, Any]:
        # Dedupe
        if event_id:
            if event_id in self.seen_events:
                return {"matched": False, "reason": "duplicate_event"}
            self.seen_events.add(event_id)

        imp = self.by_request.get(request_id)
        if not imp:
            return {"matched": False, "reason": "unknown_request_id:" + request_id}

        # Expiry
        if datetime.utcnow() > imp["expires_at"]:
            return {"matched": False, "reason": "expired"}

        # Item must match (strict policy; adjust if you allow basket/brand-level credit)
        if purchased_item != imp["item"]:
            return {"matched": False, "reason": "item_mismatch"}

        # Mark matched and attach reward
        imp["matched"] = True
        imp["reward"] = float(revenue)
        imp["ts_reward"] = datetime.utcnow()

        # Optional learner hook
        if self.on_match:
            try:
                self.on_match(imp)
            except Exception:
                # Never let learning failures break attribution
                pass

        return {"matched": True, "policy": imp["policy"], "item": imp["item"], "reward": imp["reward"]}

    # -----------------------------
    # Fallback: match by item (no request_id)
    # -----------------------------
    def match_by_item(self, purchased_item: str, revenue: float, event_id: Optional[str] = None) -> Dict[str, Any]:
        # Dedupe
        if event_id:
            if event_id in self.seen_events:
                return {"matched": False, "reason": "duplicate_event"}
            self.seen_events.add(event_id)

        dq = self.by_item.get(purchased_item)
        if not dq:
            return {"matched": False}

        now = datetime.utcnow()
        # Newest → oldest to find first open, non-expired
        for i in range(len(dq) - 1, -1, -1):
            imp = dq[i]
            if imp.get("matched"):
                continue
            if now > imp["expires_at"]:
                continue
            # Found a viable impression
            imp["matched"] = True
            imp["reward"] = float(revenue)
            imp["ts_reward"] = now

            if self.on_match:
                try:
                    self.on_match(imp)
                except Exception:
                    pass

            return {"matched": True, "policy": imp["policy"], "item": imp["item"], "reward": imp["reward"]}

        return {"matched": False}

    # -----------------------------
    # Fallback: match by session + item (best-effort in-store scenario)
    # -----------------------------
    def match_by_session(self, session_id: str, purchased_item: str, revenue: float, event_id: Optional[str] = None) -> Dict[str, Any]:
        # Dedupe
        if event_id:
            if event_id in self.seen_events:
                return {"matched": False, "reason": "duplicate_event"}
            self.seen_events.add(event_id)

        dq = self.by_session.get(session_id)
        if not dq:
            return {"matched": False, "reason": "unknown_session"}

        now = datetime.utcnow()
        for i in range(len(dq) - 1, -1, -1):
            imp = dq[i]
            if imp.get("matched"):
                continue
            if now > imp["expires_at"]:
                continue
            if imp["item"] != purchased_item:
                continue
            # Match
            imp["matched"] = True
            imp["reward"] = float(revenue)
            imp["ts_reward"] = now

            if self.on_match:
                try:
                    self.on_match(imp)
                except Exception:
                    pass

            return {"matched": True, "policy": imp["policy"], "item": imp["item"], "reward": imp["reward"]}

        return {"matched": False}

    # -----------------------------
    # Maintenance
    # -----------------------------
    def sweep_expired(self) -> None:
        """Remove expired or matched impressions from deques and light cleanup of empty keys."""
        now = datetime.utcnow()

        # by_item
        for item, dq in list(self.by_item.items()):
            while dq and (dq[0].get("matched") or dq[0]["expires_at"] < now):
                dq.popleft()
            if not dq:
                del self.by_item[item]

        # by_session
        for sess, dq in list(self.by_session.items()):
            while dq and (dq[0].get("matched") or dq[0]["expires_at"] < now):
                dq.popleft()
            if not dq:
                del self.by_session[sess]