import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Stable seeding helpers (reproducible across runs / machines)
# ------------------------------------------------------------
def stable_u32(*parts) -> int:
    """Stable 32-bit seed from arbitrary parts."""
    s = "||".join(map(str, parts)).encode("utf-8")
    h = hashlib.md5(s).hexdigest()  # stable
    return int(h[:8], 16)

def rng_from(*parts):
    return np.random.default_rng(stable_u32(*parts))

# ------------------------------------------------------------
# 7Timer-like helpers (same logic as in weather_7timer_api.py)
# ------------------------------------------------------------
def cloudcover_to_percent(cloudcover_1_to_9: int) -> float:
    if cloudcover_1_to_9 is None:
        return None
    c = int(cloudcover_1_to_9)
    c = max(1, min(9, c))
    return (c - 0.5) * (100.0 / 9.0)

def cloud_bucket_from_percent(pct: float) -> str:
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

def precip_intensity_from_amount(prec_amount: int) -> str:
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

def precip_type_norm(prec_type: str) -> str:
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

def thunderstorm_possible(lifted_index: int) -> int:
    # Rule in your existing code: LI < -5 => possible
    if lifted_index is None:
        return 0
    try:
        return 1 if int(lifted_index) < -5 else 0
    except Exception:
        return 0

def safe_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None

def safe_max(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return max(xs) if xs else None


# ------------------------------------------------------------
# Climate model knobs (tuned to your requirements)
# ------------------------------------------------------------
# Rain more in summer+autumn (per your spec), not necessarily climatology-accurate.
MONTH_RAIN_PROB = {
    1: 0.22, 2: 0.20, 3: 0.20, 4: 0.22,
    5: 0.28, 6: 0.40, 7: 0.48, 8: 0.48,
    9: 0.55, 10: 0.55, 11: 0.38, 12: 0.30
}

# Stormy winds early summer + early winter (your spec)
MONTH_STORM_PROB = {
    1: 0.06, 2: 0.03, 3: 0.02, 4: 0.02,
    5: 0.03, 6: 0.07, 7: 0.04, 8: 0.03,
    9: 0.03, 10: 0.04, 11: 0.05, 12: 0.07
}

WIND_DIRS = ["N","NE","E","SE","S","SW","W","NW"]

# ------------------------------------------------------------
# Synthetic weather generator (7Timer-like JSON dataseries)
# ------------------------------------------------------------
def seasonal_temp_c(day_of_year: int, hour: int, lat: float, store_week_anom: float, store_day_anom: float):
    """
    Sine seasonality + daily cycle + anomalies.
    Tuned so that extremes occasionally reach [-2, 38].
    """
    # seasonal: peak ~ late July (day ~ 210), trough ~ Jan/Feb
    # mean and amplitude chosen for NL-ish distribution but allow higher extremes
    mean = 11.0
    amp = 9.0
    phase = 210  # peak around end of July
    seasonal = mean + amp * math.sin(2 * math.pi * (day_of_year - phase) / 365.25)

    # daily cycle: warmer mid-afternoon
    daily = 2.8 * math.sin(2 * math.pi * (hour - 15) / 24.0)

    # latitude effect: tiny
    lat_adj = -0.35 * (lat - 52.0)

    temp = seasonal + daily + lat_adj + store_week_anom + store_day_anom
    return float(np.clip(temp, -2.0, 38.0))

def sample_precip(dt_local: datetime, temp_c: float, wetness: float, rng):
    """
    wetness ~ [0,1] gives persistence; higher -> more rain and heavier.
    """
    m = dt_local.month
    base_p = MONTH_RAIN_PROB[m]

    # More rain if "wet day" (persistent regime)
    p_rain = np.clip(base_p + 0.35 * (wetness - 0.5), 0.02, 0.90)
    has_precip = rng.random() < p_rain

    if not has_precip:
        return "none", 0

    # type depends on temperature
    if temp_c <= -0.5:
        p_snow = 0.80
        p_mixed = 0.15
    elif temp_c <= 1.5:
        p_snow = 0.30
        p_mixed = 0.40
    elif temp_c <= 3.0:
        p_snow = 0.05
        p_mixed = 0.20
    else:
        p_snow = 0.0
        p_mixed = 0.02

    u = rng.random()
    if u < p_snow:
        ptype = "snow"
    elif u < p_snow + p_mixed:
        ptype = "mixed"
    else:
        ptype = "rain"

    # intensity: heavier more likely in summer/autumn + wetness
    heavy_boost = 0.10 if m in (6,7,8,9,10) else 0.05
    heavy_p = np.clip(0.10 + heavy_boost + 0.35*(wetness-0.5), 0.05, 0.55)

    r = rng.random()
    if r < heavy_p:
        amount = int(rng.integers(6, 10))  # heavy: 6..9
    else:
        # mostly light/moderate
        amount = int(rng.choice([1,2,3,4,5], p=[0.35,0.25,0.18,0.13,0.09]))

    return ptype, amount

def sample_wind(dt_local: datetime, storminess: float, rng):
    m = dt_local.month
    p_storm = MONTH_STORM_PROB[m]
    # persistent storm regime nudges probability
    p_storm = np.clip(p_storm + 0.20*(storminess-0.5), 0.01, 0.35)

    is_storm = rng.random() < p_storm
    direction = rng.choice(WIND_DIRS)

    if is_storm:
        # storm winds (m/s) ~ 12..28
        speed = float(rng.uniform(12.0, 28.0))
    else:
        # typical winds (m/s) ~ 1..12, skewed low
        speed = float(np.clip(rng.gamma(shape=2.0, scale=2.0), 0.5, 12.0))

    return direction, speed, int(is_storm)

def sample_cloudcover(prec_amount: int, rng):
    # If raining, mostly cloudy to overcast
    if prec_amount and prec_amount > 0:
        return int(rng.choice([7,8,9], p=[0.25,0.40,0.35]))
    # else a mix
    return int(rng.choice([2,3,4,5,6,7], p=[0.10,0.18,0.22,0.20,0.16,0.14]))

def sample_lifted_index(temp_c: float, prec_amount: int, is_storm: int, dt_local: datetime, rng):
    # Thunder risk mostly in warmer months + storms/heavy precip
    m = dt_local.month
    summer = m in (6,7,8,9)
    if summer and (is_storm or prec_amount >= 6) and temp_c >= 18:
        return int(rng.integers(-9, -2))
    if summer and prec_amount > 0 and temp_c >= 16:
        return int(rng.integers(-7, 2))
    # mostly stable
    return int(np.clip(rng.normal(loc=2.0, scale=4.0), -10, 10))

def sample_rh2m(prec_amount: int, rng):
    if prec_amount and prec_amount > 0:
        rh = int(np.clip(rng.normal(93, 4), 75, 100))
    else:
        rh = int(np.clip(rng.normal(78, 10), 45, 98))
    return f"{rh}%"

def simulate_7timer_json_for_hour(
    store_id: int,
    lat: float,
    lon: float,
    now_utc: datetime,
    horizon_hours: int = 3,
    step_hours: int = 1,
) -> dict:
    """
    Build a 7Timer-like JSON response with init + dataseries (timepoint hours).
    We generate *future points* for [0, horizon_hours) so the feature extractor
    can aggregate over (0,3)h like your existing code does.
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    # Use Amsterdam local time for seasonality
    # We avoid zoneinfo dependency by using a fixed +1/+2 approximation? Better: use pandas for conversion outside.
    # Here we assume now_utc already corresponds to local-ish ordering; user will pass Europe/Amsterdam timestamps converted to UTC.
    # We'll still compute day_of_year/hour on UTC; acceptable for synthetic.
    init_str = now_utc.strftime("%Y%m%d%H")

    dataseries = []
    # persistent anomalies
    iso = now_utc.isocalendar()
    year, week = int(iso.year), int(iso.week)

    week_rng = rng_from("week", store_id, year, week)
    day_rng  = rng_from("day", store_id, now_utc.date().isoformat())

    store_week_anom = float(np.clip(week_rng.normal(0.0, 2.0), -5.0, 5.0))   # +/- a few degrees
    store_day_anom  = float(np.clip(day_rng.normal(0.0, 1.5), -4.0, 4.0))
    wetness         = float(day_rng.beta(2.0, 2.0))                           # 0..1
    storminess      = float(day_rng.beta(2.0, 3.0))                           # 0..1

    for h in range(0, horizon_hours, step_hours):
        t = now_utc + timedelta(hours=h)
        hour_rng = rng_from("hour", store_id, t.strftime("%Y-%m-%d %H"))

        day_of_year = int(t.timetuple().tm_yday)
        hour = int(t.hour)

        temp_c = seasonal_temp_c(day_of_year, hour, lat, store_week_anom, store_day_anom)

        ptype, pamount = sample_precip(t, temp_c, wetness, hour_rng)
        wdir, wspeed, is_storm = sample_wind(t, storminess, hour_rng)

        cloudcover = sample_cloudcover(pamount, hour_rng)
        lifted_index = sample_lifted_index(temp_c, pamount, is_storm, t, hour_rng)
        rh2m = sample_rh2m(pamount, hour_rng)

        dataseries.append({
            "timepoint": h,
            "cloudcover": int(cloudcover),     # 1..9
            "prec_type": ptype,                # none/rain/snow/mixed
            "prec_amount": int(pamount),       # 0..9 ordinal
            "lifted_index": int(lifted_index), # -10..10
            "rh2m": rh2m,                      # "92%"
            "temp2m": float(temp_c),           # °C
            "wind10m": {"direction": wdir, "speed": float(wspeed)},  # m/s
        })

    return {"init": init_str, "dataseries": dataseries}


# ------------------------------------------------------------
# Feature extraction from 7Timer-like JSON (matches your schema)
# ------------------------------------------------------------
def get_weather_features_from_json(
    data: dict,
    now_utc: datetime,
    tz_name: str = "Europe/Amsterdam",
    time_windows_hours=((0, 3),),
    include_onehots: bool = True
) -> dict:

    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    init = data.get("init")
    init_dt_utc = None
    if init:
        try:
            init_dt_utc = datetime.strptime(init, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        except Exception:
            init_dt_utc = None

    dataseries = data.get("dataseries", []) or []

    def step_time_utc(d):
        tp = d.get("timepoint")
        if init_dt_utc is not None and tp is not None:
            return init_dt_utc + timedelta(hours=int(tp))
        if tp is not None:
            return now_utc + timedelta(hours=int(tp))
        return None

    feats = {}

    for (h_lo, h_hi) in time_windows_hours:
        window_key = f"t{h_lo}_{h_hi}h"

        points = []
        for d in dataseries:
            t_utc = step_time_utc(d)
            if t_utc is None:
                continue
            delta_h = (t_utc - now_utc).total_seconds() / 3600.0
            if h_lo <= delta_h < h_hi:
                points.append((t_utc, d))

        # empty -> zeros
        if not points:
            if include_onehots:
                for cb in ("clear", "partly", "mostly", "cloudy", "unknown"):
                    feats[f"weather_cloud_{cb}_{window_key}"] = 0.0
                for pt in ("none", "rain", "snow", "mixed", "other", "unknown"):
                    feats[f"weather_precip_{pt}_{window_key}"] = 0.0
                for pi in ("none", "light", "moderate", "heavy", "unknown"):
                    feats[f"weather_precip_int_{pi}_{window_key}"] = 0.0

            feats[f"weather_thunderstorm_possible_{window_key}"] = 0.0
            feats[f"weather_high_humidity_{window_key}"] = 0.0
            feats[f"weather_is_day_{window_key}"] = 0.0
            feats[f"weather_n_points_{window_key}"] = 0.0
            feats[f"weather_prop_cloudy_{window_key}"] = 0.0
            feats[f"weather_prop_precip_{window_key}"] = 0.0
            feats[f"weather_prop_heavy_precip_{window_key}"] = 0.0
            feats[f"weather_temp2m_mean_{window_key}"] = 0.0
            feats[f"weather_temp2m_max_{window_key}"] = 0.0
            feats[f"weather_wind10m_speed_max_{window_key}"] = 0.0
            feats[f"weather_prec_amount_mean_{window_key}"] = 0.0
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
            cloud_pct = cloudcover_to_percent(d.get("cloudcover"))
            cloud_buckets.append(cloud_bucket_from_percent(cloud_pct))

            pt = precip_type_norm(d.get("prec_type"))
            precip_types.append(pt)

            pi = precip_intensity_from_amount(d.get("prec_amount"))
            precip_ints.append(pi)

            ts_flags.append(thunderstorm_possible(d.get("lifted_index")))

            rh = d.get("rh2m")
            if rh is None:
                humidity_flags.append(None)
            else:
                try:
                    rh_val = int(str(rh).replace("%", ""))
                    humidity_flags.append(1 if rh_val >= 90 else 0)
                except Exception:
                    humidity_flags.append(None)

            # day/night rule: 07-19 (approx). We avoid full tz conversion for speed.
            is_day_flags.append(1 if 7 <= t_utc.hour < 19 else 0)

            temps.append(d.get("temp2m"))
            w = d.get("wind10m", {})
            wind_speeds.append(w.get("speed") if isinstance(w, dict) else None)

            prec_amounts.append(d.get("prec_amount"))

        n = len(points)
        feats[f"weather_n_points_{window_key}"] = float(n)

        # modes
        cloud_mode = pd.Series(cloud_buckets).mode().iloc[0]
        precip_type_mode = pd.Series(precip_types).mode().iloc[0]
        precip_int_mode = pd.Series(precip_ints).mode().iloc[0]

        if include_onehots:
            for cb in ("clear", "partly", "mostly", "cloudy", "unknown"):
                feats[f"weather_cloud_{cb}_{window_key}"] = 1.0 if cb == cloud_mode else 0.0
            for pt in ("none", "rain", "snow", "mixed", "other", "unknown"):
                feats[f"weather_precip_{pt}_{window_key}"] = 1.0 if pt == precip_type_mode else 0.0
            for pi in ("none", "light", "moderate", "heavy", "unknown"):
                feats[f"weather_precip_int_{pi}_{window_key}"] = 1.0 if pi == precip_int_mode else 0.0

        feats[f"weather_prop_cloudy_{window_key}"] = float(sum(x == "cloudy" for x in cloud_buckets) / n)
        feats[f"weather_prop_precip_{window_key}"] = float(sum(x != "none" for x in precip_types) / n)
        feats[f"weather_prop_heavy_precip_{window_key}"] = float(sum(x == "heavy" for x in precip_ints) / n)

        feats[f"weather_thunderstorm_possible_{window_key}"] = float(sum(ts_flags) / n)
        hum_mean = safe_mean([h for h in humidity_flags if h is not None])
        feats[f"weather_high_humidity_{window_key}"] = float(hum_mean) if hum_mean is not None else 0.0
        feats[f"weather_is_day_{window_key}"] = float(sum(is_day_flags) / n)

        temp_mean = safe_mean(temps)
        temp_max = safe_max(temps)
        wind_max = safe_max(wind_speeds)
        prec_amount_mean = safe_mean(prec_amounts)

        feats[f"weather_temp2m_mean_{window_key}"] = float(temp_mean) if temp_mean is not None else 0.0
        feats[f"weather_temp2m_max_{window_key}"] = float(temp_max) if temp_max is not None else 0.0
        feats[f"weather_wind10m_speed_max_{window_key}"] = float(wind_max) if wind_max is not None else 0.0
        feats[f"weather_prec_amount_mean_{window_key}"] = float(prec_amount_mean) if prec_amount_mean is not None else 0.0

    return feats


# ------------------------------------------------------------
# End-to-end: generate hourly weather and merge onto orders
# ------------------------------------------------------------
def random_nl_store_locations(store_ids: List[int], seed: int = 123) -> Dict[int, Tuple[float, float]]:
    """
    If you don't have real store lat/lon, sample deterministic random points in NL bbox.
    """
    rng = np.random.default_rng(seed)
    # Rough NL bounding box
    lat_min, lat_max = 50.75, 53.55
    lon_min, lon_max = 3.35, 7.22
    locs = {}
    for sid in store_ids:
        locs[sid] = (float(rng.uniform(lat_min, lat_max)), float(rng.uniform(lon_min, lon_max)))
    return locs


def enforce_weather_dtypes(weather_hourly: pd.DataFrame) -> pd.DataFrame:
    # identify columns
    npoint_cols = [c for c in weather_hourly.columns if c.startswith("weather_n_points_")]
    weather_cols = [c for c in weather_hourly.columns if c.startswith("weather_") and c not in npoint_cols]

    # cast
    for c in weather_cols:
        weather_hourly[c] = weather_hourly[c].astype("float32")
    for c in npoint_cols:
        # fill missing then int
        weather_hourly[c] = weather_hourly[c].fillna(0).astype("int16")

    return weather_hourly

def gen_synthetic_weather(
    keys: List[Tuple[int, int]],
    store_locations: Optional[Dict[int, Tuple[float, float]]] = None,
    time_windows_hours=((0,3),),
    include_onehots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      orders_enriched, weather_hourly_table
    """

    records = []

    for (sid, ts_hour) in keys:
        lat, lon = store_locations[int(sid)]
        now_utc = ts_hour.to_pydatetime().replace(tzinfo=timezone.utc)

        # synthetic 7timer-like response for that hour
        json_data = simulate_7timer_json_for_hour(
            store_id=int(sid),
            lat=lat, lon=lon,
            now_utc=now_utc,
            horizon_hours=max(h for _, h in time_windows_hours),  # cover biggest window
            step_hours=1
        )

        feats = get_weather_features_from_json(
            json_data, now_utc=now_utc,
            time_windows_hours=time_windows_hours,
            include_onehots=include_onehots
        )

        rec = {"store_id": int(sid), "ts_hour": ts_hour}
        rec.update(feats)
        records.append(rec)

    weather_hourly = pd.DataFrame.from_records(records)
    

        
    weather_hourly = enforce_weather_dtypes(weather_hourly)

    return weather_hourly


if __name__ == "__main__":
    """
    Script-only execution:
    - Loads orders from a fixed path
    - Generates synthetic NL weather (flat columns)
    - Writes enriched orders + hourly weather tables
    """

    import pandas as pd

    # ------------------------------------------------------------------
    # CONFIG — edit these paths once
    # ------------------------------------------------------------------
    ORDERS_PATH = "../data/parquets/orders_multistore_2026.parquet"     # or .csv
    ORDERS_IS_CSV = False                              # set True if CSV

    OUT_WEATHER_PATH = "../data/parquets/weather_hourly.parquet"

    # ------------------------------------------------------------------
    # Load orders
    # ------------------------------------------------------------------
    if ORDERS_IS_CSV:
        orders = pd.read_csv(ORDERS_PATH)
    else:
        orders = pd.read_parquet(ORDERS_PATH)

    # ------------------------------------------------------------------
    # Generate synthetic weather (flat columns only)
    # ------------------------------------------------------------------
    
    orders["order_timestamp"] = pd.to_datetime(orders["order_timestamp"])
    # bucket to hour (saves huge compute)
    orders["ts_hour"] = orders["order_timestamp"].dt.floor("H")
    
    keys = list(orders[["store_id", "ts_hour"]]
            .drop_duplicates()
            .itertuples(index=False, name=None))
    
    df = pd.read_csv("stores.csv")    
   
    store_locations = (
        df.set_index("store_id")[["latitude", "longitude"]]
          .apply(tuple, axis=1)
          .to_dict()
    )
    
    weather_hourly = gen_synthetic_weather(
        keys,
        store_locations=store_locations,
        time_windows_hours=((0, 3),),
        include_onehots=True
    )

    # ------------------------------------------------------------------
    # Persist outputs
    # ------------------------------------------------------------------
    weather_hourly.to_parquet(OUT_WEATHER_PATH, index=False)

    # ------------------------------------------------------------------
    # Sanity output
    # ------------------------------------------------------------------
    print("Weather generation completed")
    print("Orders input        :", len(orders))
    print("Hourly weather rows :", len(weather_hourly))