import math
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import requests

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

URL = "http://www.7timer.info/bin/api.pl"

# -----------------------------
# Bucketing helpers
# -----------------------------

def _cloudcover_to_percent(cloudcover_1_to_9: int) -> float:
    """
    7Timer cloudcover is 1..9. We map it to an approximate percent (0..100).
    Uses a mid-point mapping of 9 bins.
    """
    if cloudcover_1_to_9 is None:
        return None
    c = int(cloudcover_1_to_9)
    c = max(1, min(9, c))
    # 9 bins of ~11.11% each; take midpoint
    return (c - 0.5) * (100.0 / 9.0)


def _cloud_bucket_from_percent(pct: float) -> str:
    """
    Bucket into clear/partly/mostly/cloudy using thresholds
    consistent with earlier discussion:
    <20% clear; 20-60 partly; 60-80 mostly; >80 cloudy
    """
    if pct is None:
        return "unknown"
    if pct < 20:
        return "clear"
    elif pct < 60:
        return "partly"
    elif pct < 80:
        return "mostly"
    else:
        return "cloudy"


def _precip_intensity_from_amount(prec_amount: int) -> str:
    """
    7Timer prec_amount is typically 0..9 (ordinal scale).
    We map to none/light/moderate/heavy.
    """
    if prec_amount is None:
        return "unknown"
    a = int(prec_amount)
    if a <= 0:
        return "none"
    elif a <= 2:
        return "light"
    elif a <= 5:
        return "moderate"
    else:
        return "heavy"


def _precip_type_norm(prec_type: str) -> str:
    """
    Normalize precipitation type. 7Timer often uses:
    'none', 'rain', 'snow', 'frzr' (freezing rain), 'icep' (ice pellets)
    """
    if not prec_type:
        return "unknown"
    t = str(prec_type).lower()
    if t in ("none", "0"):
        return "none"
    if "rain" in t and "freez" in t:
        return "mixed"
    if t in ("frzr", "icep"):
        return "mixed"
    if "snow" in t:
        return "snow"
    if "rain" in t:
        return "rain"
    return "other"


def _thunderstorm_possible(lifted_index: int) -> int:
    """
    Rule-of-thumb: LI < -5 => thunderstorm possible.
    """
    if lifted_index is None:
        return 0
    try:
        return 1 if int(lifted_index) < -5 else 0
    except Exception:
        return 0


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def _safe_max(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return max(xs) if xs else None


# -----------------------------
# Core feature extraction
# -----------------------------

def get_weather_features(
    lat: float,
    lon: float,
    now_utc: datetime = None,
    tz_name: str = "Europe/Amsterdam",
    time_windows_hours=((0, 3),),
    include_onehots: bool = True
) -> dict:
    """
    Convert 7Timer (product=civil) JSON into XGBoost/bandit-friendly features.

    Output: flat dict of numeric features (floats/ints).
    """
    
    params = {"lat": lat, "lon": lon, "product": "civil", "output": "json"}
    resp = requests.get(URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    tz = ZoneInfo(tz_name) if ZoneInfo else None

    # 7Timer "init" looks like "2026040106" sometimes; civil dataseries uses timepoint hours
    init = data.get("init")
    init_dt_utc = None
    if init:
        # Try parsing typical "YYYYMMDDHH"
        try:
            init_dt_utc = datetime.strptime(init, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        except Exception:
            init_dt_utc = None

    dataseries = data.get("dataseries", []) or []

    # Helper: compute each forecast step's timestamp in UTC
    def step_time_utc(d):
        tp = d.get("timepoint")  # hours from init
        if init_dt_utc is not None and tp is not None:
            return init_dt_utc + timedelta(hours=int(tp))
        # fallback: approximate using now_utc as reference
        if tp is not None:
            return now_utc + timedelta(hours=int(tp))
        return None

    # Collect per window
    feats = defaultdict(float)

    for (h_lo, h_hi) in time_windows_hours:
        window_key = f"t{h_lo}_{h_hi}h"

        # filter points in horizon
        points = []
        for d in dataseries:
            t_utc = step_time_utc(d)
            if t_utc is None:
                continue
            delta_h = (t_utc - now_utc).total_seconds() / 3600.0
            if h_lo <= delta_h < h_hi:
                points.append((t_utc, d))

        # If empty window, keep zeros for counts and NaNs for means (or omit)
        if not points:
            # counts as 0
            if include_onehots:
                for cb in ("clear", "partly", "mostly", "cloudy", "unknown"):
                    feats[f"weather_cloud_{cb}_{window_key}"] += 0
                for pt in ("none", "rain", "snow", "mixed", "other", "unknown"):
                    feats[f"weather_precip_{pt}_{window_key}"] += 0
                for pi in ("none", "light", "moderate", "heavy", "unknown"):
                    feats[f"weather_precip_int_{pi}_{window_key}"] += 0
            feats[f"weather_thunderstorm_possible_{window_key}"] += 0
            feats[f"weather_high_humidity_{window_key}"] += 0
            feats[f"weather_is_day_{window_key}"] += 0
            feats[f"weather_n_points_{window_key}"] += 0
            continue

        cloud_buckets = []
        precip_types = []
        precip_ints = []
        ts_flags = []
        humidity_flags = []
        is_day_flags = []
        temps = []
        wind_speeds = []
        prec_amounts = []

        for t_utc, d in points:
            cloud_pct = _cloudcover_to_percent(d.get("cloudcover"))
            cloud_buckets.append(_cloud_bucket_from_percent(cloud_pct))

            pt = _precip_type_norm(d.get("prec_type"))
            precip_types.append(pt)

            pi = _precip_intensity_from_amount(d.get("prec_amount"))
            precip_ints.append(pi)

            ts_flags.append(_thunderstorm_possible(d.get("lifted_index")))

            rh = d.get("rh2m")  # may be present in some 7Timer products; if missing -> None
            if rh is None:
                humidity_flags.append(None)
            else:
                # rh2m often categorical strings like '90%' or numbers; be robust
                try:
                    rh_val = int(str(rh).replace("%", ""))
                    humidity_flags.append(1 if rh_val >= 90 else 0)
                except Exception:
                    humidity_flags.append(None)

            # day/night from local hour
            if tz:
                local = t_utc.astimezone(tz)
                # naive rule: day if 07:00-19:00
                is_day_flags.append(1 if 7 <= local.hour < 19 else 0)
            else:
                # fallback: assume UTC approximates day; not ideal
                is_day_flags.append(1 if 7 <= t_utc.hour < 19 else 0)

            # numeric fields
            temps.append(d.get("temp2m"))
            # wind10m might be dict {"direction":"SE","speed":3}
            w = d.get("wind10m", {})
            if isinstance(w, dict):
                wind_speeds.append(w.get("speed"))
            else:
                wind_speeds.append(None)

            prec_amounts.append(d.get("prec_amount"))

        # --- Aggregations ---
        n = len(points)
        feats[f"weather_n_points_{window_key}"] = n

        # Dominant categories (mode)
        cloud_mode = Counter(cloud_buckets).most_common(1)[0][0]
        precip_type_mode = Counter(precip_types).most_common(1)[0][0]
        precip_int_mode = Counter(precip_ints).most_common(1)[0][0]

        # One-hot for modes (sparse & stable)
        if include_onehots:
            for cb in ("clear", "partly", "mostly", "cloudy", "unknown"):
                feats[f"weather_cloud_{cb}_{window_key}"] = 1.0 if cb == cloud_mode else 0.0

            for pt in ("none", "rain", "snow", "mixed", "other", "unknown"):
                feats[f"weather_precip_{pt}_{window_key}"] = 1.0 if pt == precip_type_mode else 0.0

            for pi in ("none", "light", "moderate", "heavy", "unknown"):
                feats[f"weather_precip_int_{pi}_{window_key}"] = 1.0 if pi == precip_int_mode else 0.0

        # Counts/proportions (often better than one-hot for XGB)
        feats[f"weather_prop_cloudy_{window_key}"] = sum(1 for x in cloud_buckets if x == "cloudy") / n
        feats[f"weather_prop_precip_{window_key}"] = sum(1 for x in precip_types if x != "none") / n
        feats[f"weather_prop_heavy_precip_{window_key}"] = sum(1 for x in precip_ints if x in ("heavy",)) / n

        # Thunderstorm/humidity/day flags (mean = fraction of points)
        feats[f"weather_thunderstorm_possible_{window_key}"] = sum(ts_flags) / n
        # humidity might be missing; average only non-missing
        hum_mean = _safe_mean([h for h in humidity_flags if h is not None])
        feats[f"weather_high_humidity_{window_key}"] = float(hum_mean) if hum_mean is not None else 0.0
        feats[f"weather_is_day_{window_key}"] = sum(is_day_flags) / n

        # Numeric summaries
        temp_mean = _safe_mean(temps)
        temp_max = _safe_max(temps)
        wind_max = _safe_max(wind_speeds)
        prec_amount_mean = _safe_mean(prec_amounts)

        # Use 0 if missing; alternatively omit keys
        feats[f"weather_temp2m_mean_{window_key}"] = float(temp_mean) if temp_mean is not None else 0.0
        feats[f"weather_temp2m_max_{window_key}"] = float(temp_max) if temp_max is not None else 0.0
        feats[f"weather_wind10m_speed_max_{window_key}"] = float(wind_max) if wind_max is not None else 0.0
        feats[f"weather_prec_amount_mean_{window_key}"] = float(prec_amount_mean) if prec_amount_mean is not None else 0.0

    return dict(feats)