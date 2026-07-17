import requests
import urllib.parse as urlparse
from datetime import datetime, timedelta, UTC, timezone
from typing import Any, Dict, List, Optional
import html
from urllib.parse import parse_qs, urlencode, urlunparse
import math
from collections import defaultdict
from math import inf


# -----------------------------
# Configuration
# -----------------------------

DISTANCE_BUCKETS_KM = [
    (0, 1),
    (1, 3)
]

TIME_BUCKETS_HOURS = [
    (0, 3),
    (3, 12)
]

GENRE_MAP = {
    "music": ["music"],
    "theatre": ["theatre", "arts"],
    "nightlife": ["dance", "electronic"],
    "exhibition": ["exhibition", "museum"],
    "family": ["family", "children"],
}

class EventsFetcher:
    BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"

    def __init__(self, api_key: str, src_lat_lon: tuple, timeout: float = 15.0):
        self.api_key = api_key
        self.src_lat_lon = src_lat_lon
        self.timeout = timeout
        self.session = requests.Session()

    # ---------- helpers ----------

    @staticmethod
    def distance(origin: tuple, destination: tuple) -> float:
        lat1, lon1 = origin
        lat2, lon2 = destination
        radius = 6371 # km

        dlat = math.radians(lat2-lat1)
        dlon = math.radians(lon2-lon1)
        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = radius * c
        return d

    @staticmethod
    def _safe_get(obj: Any, path: List[Any], default: Optional[Any] = None) -> Any:
        cur = obj
        for key in path:
            try:
                if isinstance(key, int):
                    if not isinstance(cur, list) or key >= len(cur):
                        return default
                    cur = cur[key]
                else:
                    if not isinstance(cur, dict) or key not in cur:
                        return default
                    cur = cur[key]
            except Exception:
                return default
        return cur

    @staticmethod
    def _to_float_or_none(x: Any) -> Optional[float]:
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _utc_now_iso() -> str:
        # timezone-aware UTC (no deprecation warning)
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _normalize_name(name: Optional[str]) -> str:
        if not name:
            return ""
        s = name.lower().strip()
        # remove common suffix variants that cause duplicates
        cut_suffixes = [
            " | vip packages",
            " | comfort seats",
            " | vip",
            " | packages",
        ]
        for suf in cut_suffixes:
            if s.endswith(suf):
                s = s[: -len(suf)]
        # light punctuation trim
        return s.replace("–", "-").strip()

    @staticmethod
    def _unwrap_affiliate_url(u: Optional[str]) -> Optional[str]:
        """
        Ticketmaster affiliate links often look like:
        https://ticketmaster.evyy.net/c/...?...&u=<encoded real url>
        We extract and return the underlying `u` param if present.
        """
        if not u:
            return None
        try:
            parsed = urlparse.urlparse(u)
            q = parse_qs(parsed.query)
            if "u" in q and q["u"]:
                return urlparse.unquote(q["u"][0])
            return u
        except Exception:
            return u

    @staticmethod
    def _better_of(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """
        When two records collide on a dedupe key, prefer the one with more useful info:
        - Has venue name
        - Has price range (min/max/currency)
        - Longer name
        """
        score_a = 0
        score_b = 0

        if a.get("venue_name"): score_a += 1
        if b.get("venue_name"): score_b += 1

        if a.get("price_min") is not None or a.get("price_max") is not None: score_a += 1
        if b.get("price_min") is not None or b.get("price_max") is not None: score_b += 1

        score_a += len(a.get("name") or "")
        score_b += len(b.get("name") or "")

        return a if score_a >= score_b else b

    # ---------- extraction ----------


    def normalize_ticketmaster_url(self, u: str) -> str:
        """
        - HTML-unescape (handles &amp;)
        - If affiliate wrapper (ticketmaster.evyy.net) is present, extract the real 'u' param
        - URL-decode once
        - Rebuild a clean URL with proper query separators
        """
        if not u:
            return u

        # Step 1: HTML-unescape (&amp; -> &)
        u = html.unescape(u)

        # Step 2: unwrap affiliate "u" parameter if present
        try:
            parsed = urlparse.urlparse(u)
            if parsed.netloc.endswith("ticketmaster.evyy.net"):
                q = parse_qs(parsed.query)
                real = q.get("u", [None])[0]
                if real:
                    u = html.unescape(urlparse.unquote(real))  # decode underlying target
                    parsed = urlparse.urlparse(u)              # re-parse target

            # Step 3: normalize query (optional but nice)
            q2 = parse_qs(parsed.query, keep_blank_values=True)
            # If you don’t want UTMs in your final URL, you can drop them:
            # for k in list(q2.keys()):
            #     if k.lower().startswith("utm_") or k in {"ref"}:
            #         q2.pop(k, None)

            clean_query = urlencode({k: v[0] if len(v)==1 else v for k, v in q2.items()}, doseq=True)
            clean = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", clean_query, ""))
            return clean
        except Exception:
            return u  # fallback: return best-effort

    def _extract(self, event: Dict[str, Any]) -> Dict[str, Any]:
        # Core
        event_id = event.get("id")
        name = event.get("name")
        url = event.get("url")
        status = self._safe_get(event, ["dates", "status", "code"])

        # Time
        date_local   = self._safe_get(event, ["dates", "start", "localDate"])
        time_local   = self._safe_get(event, ["dates", "start", "localTime"])
        dateTime_utc = self._safe_get(event, ["dates", "start", "dateTime"])

        # Venue
        venue_name = self._safe_get(event, ["_embedded", "venues", 0, "name"])
        venue_id   = self._safe_get(event, ["_embedded", "venues", 0, "id"])
        venue_city = self._safe_get(event, ["_embedded", "venues", 0, "city", "name"])
        venue_state = (
            self._safe_get(event, ["_embedded", "venues", 0, "state", "name"]) or
            self._safe_get(event, ["_embedded", "venues", 0, "state", "stateCode"])
        )
        venue_country = (
            self._safe_get(event, ["_embedded", "venues", 0, "country", "name"]) or
            self._safe_get(event, ["_embedded", "venues", 0, "country", "countryCode"])
        )
        venue_postal = self._safe_get(event, ["_embedded", "venues", 0, "postalCode"])
        venue_addr1  = self._safe_get(event, ["_embedded", "venues", 0, "address", "line1"])

        venue_lat = self._to_float_or_none(self._safe_get(event, ["_embedded", "venues", 0, "location", "latitude"]))
        venue_lon = self._to_float_or_none(self._safe_get(event, ["_embedded", "venues", 0, "location", "longitude"]))

        venue_distance_km = self.distance((venue_lat, venue_lon), self.src_lat_lon)

        # Classifications
        segment  = self._safe_get(event, ["classifications", 0, "segment", "name"])
        genre    = self._safe_get(event, ["classifications", 0, "genre", "name"])
        subgenre = self._safe_get(event, ["classifications", 0, "subGenre", "name"])

        # Commercial
        promoter_name = (
            self._safe_get(event, ["promoter", "name"]) or
            self._safe_get(event, ["promoters", 0, "name"])
        )

        price_min      = self._safe_get(event, ["priceRanges", 0, "min"])
        price_max      = self._safe_get(event, ["priceRanges", 0, "max"])
        price_currency = self._safe_get(event, ["priceRanges", 0, "currency"])

        seatmap_url        = self._safe_get(event, ["seatmap", "staticUrl"])
        accessibility_info = (
            self._safe_get(event, ["accessibility", "info"]) or
            self._safe_get(event, ["accessibility", "ticketLimit"]) or
            self._safe_get(event, ["pleaseNote"])
        )

        # Construct venue_address safely, handling potential None values
        address_parts = [p for p in [venue_addr1, venue_postal, venue_city] if p is not None and p != ""]
        venue_address = ", ".join(address_parts)

        return {
            "name": name,
            "url": self.normalize_ticketmaster_url(url),
            "date_local": date_local,
            "time_local": time_local,
            "dateTime_utc": dateTime_utc,
            "venue_name": venue_name,
            "venue_address": venue_address,
            "venue_distance_km": venue_distance_km,
            "segment/genre": f"{segment or ''}/{genre or ''}".strip('/'),
            "promoter_name": promoter_name
        }

    # ---------- public API ----------

    def get_events(
        self,
        city: str = "Amsterdam",
        days_ahead: int = 30,
        limit: int = 50,
        paginate: bool = False,
        suppress_variants: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Deduplicated, relevant events.
        - suppress_variants=True removes VIP/Packages/Comfort Seats variants.
        """
        start_date = self._utc_now_iso()
        end_date = (datetime.now(UTC) + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "apikey": self.api_key,
            "city": city,
            "startDateTime": start_date,
            "endDateTime": end_date,
            "sort": "date,asc",
            "size": max(1, min(limit, 200)),
        }

        # Store candidates before dedupe
        candidates: List[Dict[str, Any]] = []

        page = 0
        total_pages = 1

        while True:
            params["page"] = page
            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                break

            raw_events = self._safe_get(data, ["_embedded", "events"], default=[]) or []
            for ev in raw_events:
                extracted_event = self._extract(ev)
                if extracted_event: # Only add if extraction was successful (not None)
                    candidates.append(extracted_event)

            if not paginate:
                break

            total_pages = self._safe_get(data, ["page", "totalPages"], default=1) or 1
            page += 1
            if page >= int(total_pages):
                break

        # ---------- optional: suppress variants like VIP/Packages ----------
        if suppress_variants:
            filtered = []
            for e in candidates:
                nm = (e.get("name") or "").lower()
                if any(tag in nm for tag in ["| vip packages", "| comfort seats", " vip packages", " comfort seats"]):
                    continue
                filtered.append(e)
            candidates = filtered

        # ---------- build keys for dedupe ----------
        dedup_id: Dict[str, Dict[str, Any]] = {}
        dedup_url: Dict[str, Dict[str, Any]] = {}
        dedup_semantic: Dict[str, Dict[str, Any]] = {}

        results: List[Dict[str, Any]] = []

        for e in candidates:
            if e.get('dateTime_utc') is None:
                continue
            # 1) by event ID
            eid = e.get("id")
            if eid:
                if eid not in dedup_id:
                    dedup_id[eid] = e
                    results.append(e)
                else:
                    # prefer better record
                    better = self._better_of(dedup_id[eid], e)
                    if better is not dedup_id[eid]:
                        # replace in results
                        idx = results.index(dedup_id[eid])
                        results[idx] = better
                        dedup_id[eid] = better
                continue

            # 2) by canonical URL (unwrap affiliate)
            raw_url = e.get("url")
            canon = self._unwrap_affiliate_url(raw_url)
            if canon:
                if canon not in dedup_url:
                    dedup_url[canon] = e
                    results.append(e)
                else:
                    better = self._better_of(dedup_url[canon], e)
                    if better is not dedup_url[canon]:
                        idx = results.index(dedup_url[canon])
                        results[idx] = better
                        dedup_url[canon] = better
                continue

            # 3) by semantic fingerprint
            key = "|".join([
                self._normalize_name(e.get("name")),
                (e.get("venue_name") or "").lower().strip(),
                e.get("date_local") or "",
                e.get("time_local") or "",
            ])
            if key not in dedup_semantic:
                dedup_semantic[key] = e
                results.append(e)
            else:
                better = self._better_of(dedup_semantic[key], e)
                if better is not dedup_semantic[key]:
                    idx = results.index(dedup_semantic[key])
                    results[idx] = better
                    dedup_semantic[key] = better

        return results

# -----------------------------
# Helpers
# -----------------------------

def parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def bucket_label(prefix, low, high, unit):
    return f"{prefix}_b{low}_{high}{unit}"

def get_genre_bucket(segment: str) -> str:
    if not segment:
        return "other"
    seg = segment.lower()
    for bucket, keywords in GENRE_MAP.items():
        if any(k in seg for k in keywords):
            return bucket
    return "other"

# -----------------------------
# Feature extraction
# -----------------------------
def _init_all_bucket_features(features, genres=None):
    """
    Pre-populate ALL distance x time x genre bucket combinations.
    If a bucket has no events:
      - count = 0
      - min_dist = -1
    """
    if genres is None:
        # include "other" even if not in map, because get_genre_bucket can return it
        genres = list(GENRE_MAP.keys()) + ["other"]

    for (d0, d1) in DISTANCE_BUCKETS_KM:
        dlab = bucket_label("b", d0, d1, "km")
        for (t0, t1) in TIME_BUCKETS_HOURS:
            tlab = bucket_label("t", t0, t1, "h")
            for g in genres:
                # counts default to 0
                features[f"events_{g}_count_{dlab}_{tlab}"] = 0.0
                # min distance default to -1 (explicit “no event”)
                #features[f"events_{g}_min_dist_km_{dlab}_{tlab}"] = 100.0

    # Optional: also keep totals across genres (often useful)
    for (d0, d1) in DISTANCE_BUCKETS_KM:
        dlab = bucket_label("b", d0, d1, "km")
        for (t0, t1) in TIME_BUCKETS_HOURS:
            tlab = bucket_label("t", t0, t1, "h")
            features[f"events_count_{dlab}_{tlab}"] = 0.0
            features[f"events_min_dist_km_{dlab}_{tlab}"] = -1.0
    
    features["events_weighted_sum_t0_24h"] = 0.0
    features["unique_venues_t0_24h"] = 0.0
    features["max_events_same_venue_t0_24h"] = 0.0
    

def extract_event_features(events, now_utc: datetime):
    features = defaultdict(float)
        
    # pre-create all bucket keys with defaults
    _init_all_bucket_features(features)

    venue_counter = defaultdict(int)

    for e in events:
        # ---- basic fields ----
        try:
            event_time = parse_utc(e["dateTime_utc"])
        except Exception:
            continue

        distance = e.get("venue_distance_km", inf)
        venue = e.get("venue_name", "unknown")
        genre_bucket = get_genre_bucket(e.get("segment/genre"))

        # ---- time to event (hours) ----
        delta_hours = (event_time - now_utc).total_seconds() / 3600

        if delta_hours < 0 or delta_hours > 24:
            continue

        venue_counter[venue] += 1

        # ---- A + C: count & temporal buckets ----
        for d_lo, d_hi in DISTANCE_BUCKETS_KM:
            if not (d_lo <= distance < d_hi):
                continue

            for t_lo, t_hi in TIME_BUCKETS_HOURS:
                if not (t_lo <= delta_hours < t_hi):
                    continue

                key = (
                    f"events_count_b{d_lo}_{d_hi}km_t{t_lo}_{t_hi}h"
                )
                features[key] += 1

                # ---- D: genre split ----
                genre_key = (
                    f"events_{genre_bucket}_b{d_lo}_{d_hi}km_t{t_lo}_{t_hi}h"
                )
                features[genre_key] += 1

        # ---- B: distance‑weighted intensity ----
        if distance > 0:
            features["events_weighted_sum_t0_24h"] += 1 / distance

    # ---- E: venue repetition ----
    if venue_counter:
        features["unique_venues_t0_24h"] = len(venue_counter)
        features["max_events_same_venue_t0_24h"] = max(
            venue_counter.values()
        )

    return dict(features)

def get_event_features(now_utc: datetime, 
                       lat: float, 
                       lon: float) -> Dict[str, float]:
    fetcher = EventsFetcher(api_key='ouTfABGLAuJcyXUuABaKHNdAQxZgQ41k', src_lat_lon=(lat, lon))    
    # Single page (fast)
    events = fetcher.get_events(city="Amsterdam", days_ahead=1, limit=250, paginate=True, suppress_variants=True)
    features = extract_event_features(events, now_utc)
    return features
