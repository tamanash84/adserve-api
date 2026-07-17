import requests
import json
import time
import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

"""overpass_poi_api2_retail.py

Retail-ad-serving oriented OSM/Overpass feature extractor.

What it does
- One Overpass POST for all requested tag-keys/values within max radius
- Aggregates into a ONE-LEVEL (flat) dictionary with stable keys per radius:
    poi_<group>_count_r{r}
    poi_<group>_any_r{r}
    poi_<group>_min_dist_m_r{r}
    poi_<group>_mean_top3_dist_m_r{r}
  (+ optional density and open_now counts)

Notes
- Designed for offline caching; do NOT call Overpass per impression.
"""

# -----------------------------
# Retail-relevant POI groups
# -----------------------------

# Tag key -> group_name -> set(values)
# Use None instead of set(values) to mean "any value" (key existence).
POI_GROUPS_BY_KEY = {
  "shop": {
    "shop_grocery_supermarket": {"supermarket"},
    "shop_grocery_convenience": {"convenience"},
    "shop_food_nonsupermarket": {"bakery","butcher","greengrocer","deli","delicatessen","organic","wine"},
    "shop_nonfood_nonelectronics": {"clothes","shoes","cosmetics"},
    "shop_nonfood_electronics": {"electronics"},
    "shop_tourism_souvenir": {"souvenir"},
  },

  "amenity": {
    "amenity_marketplace": {"marketplace"},
    "amenity_health_nonhospital": {"pharmacy","clinic"},
    "amenity_health_hospital": {"hospital"},
    "amenity_money": {"atm","bank"},
    "amenity_food_fastfood": {"fast_food"},
    "amenity_food_nonfastfood": {"restaurant","cafe"},
    "amenity_nightlife": {"bar","pub","nightclub"},
    "amenity_entertainment": {"cinema","theatre"},
    "amenity_stadium": {"stadium"},
    "amenity_mobility_parking": {"parking"},
    "amenity_mobility_car_rental": {"car_rental"},
    "amenity_lifestyle": {"gym","spa"},
    "amenity_education": {"school","college","university"},

    # unique group names (no overwriting)
    "pt_access_bus_station": {"bus_station"},
    "pt_access_ferry": {"ferry_terminal"},
  },

  # public transport and railway stay, but keep unique group names
  "public_transport": {
    "pt_access_generic": {"station", "platform", "stop_position"},
  },
  "railway": {
    "pt_access_rail": {"station", "subway_entrance", "tram_stop"},
  },

  # footway/crossing should be via highway=*
  "highway": {
    "pt_access_bus_stop": {"bus_stop"},
    "walkability_links": {"footway", "pedestrian"},
    "walkability_crossing": {"crossing"},   # highway=crossing
  },

  "leisure": {
    "leisure_park": {"park"},
  },
  "tourism": {
    "tourism_attraction": {"attraction", "museum"},
    "tourism_accommodation": {"hotel", "hostel"},
  },
  "landuse": {
    "landuse_residential": {"residential"},
    "landuse_industrial": {"industrial"},
  },
  "building": {
    "building_apartments": {"apartments"},
    "building_house": {"house"},
  },
  "office": {
    "office_any": None,
  },

  # drive_through is a tag key; treat it as “exists” or explicit yes
  "drive_through": {
    "drive_through_yes": {"yes"},
  },

  # add (optional) crossing tag existence (secondary tag), if you want:
  "crossing": {"crossing_any": None},
}

# Values for which "open now" is meaningful.
TIME_SENSITIVE_AMENITY = {
    "restaurant", "fast_food", "cafe", "pub", "bar", "nightclub",
    "cinema", "theatre"
}
TIME_SENSITIVE_SHOP = {
    "supermarket", "convenience", "bakery", "butcher", "greengrocer",
}


OPEN_NOW_GROUP_WHITELIST = {
    "amenity_food_fastfood",
    "amenity_food_nonfastfood",
    "amenity_nightlife",
    "amenity_entertainment",
    "amenity_health_nonhospital",
    "shop_grocery_supermarket",
    "shop_grocery_convenience",
    "shop_food_nonsupermarket",
}


# -----------------------------
# Overpass helper
# -----------------------------
def _band_for_distance(dist, bands):
    """
    Return index of the band (low, high) such that:
      - first band includes dist in [0, high]
      - other bands include dist in (low, high]
    """
    for i, (low, high) in enumerate(bands):
        if i == 0:
            if 0.0 <= dist <= high:
                return i
        else:
            if low < dist <= high:
                return i
    return None

def safe_overpass_post(query, max_tries=3, timeout=60):
    url = "https://overpass-api.de/api/interpreter"
    for _ in range(max_tries):
        try:
            r = requests.post(url, data=query.encode("utf-8"), timeout=timeout)
            if r.status_code != 200:
                print(f"[Overpass] HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(1)
                continue
            try:
                return r.json()
            except json.JSONDecodeError:
                print("[Overpass] Invalid JSON response:")
                print(r.text[:300])
                time.sleep(1)
        except Exception as e:
            print("Error:", e)
            time.sleep(1)
    return None

# -----------------------------
# Geometry helpers
# -----------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def osm_point(el):
    """node -> (lat,lon), way/relation -> (center.lat, center.lon) (requires `out center`)"""
    if el.get("type") == "node":
        return el.get("lat"), el.get("lon")
    c = el.get("center") or {}
    return c.get("lat"), c.get("lon")


def _regex_from_values(values):
    """Create Overpass regex like ^(a|b|c)$ from a set of values."""
    values = sorted(set(values))
    if not values:
        return None
    esc = [re.escape(v) for v in values]
    return "^(" + "|".join(esc) + ")$"

# -----------------------------
# opening_hours parser (practical subset)
# -----------------------------

DAY_MAP = {"Mo": 0, "Tu": 1, "We": 2, "Th": 3, "Fr": 4, "Sa": 5, "Su": 6}


def _expand_days(day_expr: str):
    day_expr = (day_expr or "").strip()
    if not day_expr:
        return set()
    parts = [p.strip() for p in day_expr.split(",") if p.strip()]
    days = set()
    for p in parts:
        if "-" in p and p[:2] in DAY_MAP and p[-2:] in DAY_MAP:
            start = DAY_MAP[p[:2]]
            end = DAY_MAP[p[-2:]]
            if start <= end:
                days.update(range(start, end + 1))
            else:
                days.update(list(range(start, 7)) + list(range(0, end + 1)))
        else:
            if p[:2] in DAY_MAP:
                days.add(DAY_MAP[p[:2]])
    return days


_time_range_re = re.compile(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})")


def _parse_hhmm(s):
    hh, mm = s.split(":")
    return int(hh), int(mm)


def is_open_now(opening_hours: str, now: datetime):
    """Return 1 if open, 0 if closed, None if unknown/unparseable."""
    if not opening_hours or not isinstance(opening_hours, str):
        return None
    oh = opening_hours.strip()
    if not oh:
        return None
    if oh in {"24/7", "24x7", "24-7"}:
        return 1
    oh = oh.replace("PH off", "").replace("PH", "").strip()

    weekday = now.weekday()
    minutes_now = now.hour * 60 + now.minute

    rules = [r.strip() for r in oh.split(";") if r.strip()]
    if not rules:
        return None

    any_parsed = False
    for rule in rules:
        m = _time_range_re.search(rule)
        if not m:
            continue
        any_parsed = True

        idx = m.start()
        day_expr = rule[:idx].strip()
        if not day_expr:
            days = set(range(7))
        else:
            tokens = day_expr.split()
            days = _expand_days(tokens[0]) if tokens else set()
        if weekday not in days:
            continue

        ranges = _time_range_re.findall(rule)
        for start_s, end_s in ranges:
            sh, sm = _parse_hhmm(start_s)
            eh, em = _parse_hhmm(end_s)
            start = sh * 60 + sm
            end = eh * 60 + em

            if start <= end:
                if start <= minutes_now <= end:
                    if "off" not in rule.lower():
                        return 1
            else:
                # overnight window
                if minutes_now >= start or minutes_now <= end:
                    if "off" not in rule.lower():
                        return 1

    if any_parsed:
        return 0
    return None

# -----------------------------
# Overpass query builder
# -----------------------------

def _collect_key_specs(POI_GROUPS_BY_KEY):
    """Return (key_to_regex, key_exists) based on POI_GROUPS_BY_KEY."""
    key_to_regex = {}
    key_exists = set()
    for k, gmap in (POI_GROUPS_BY_KEY or {}).items():
        for _, vals in (gmap or {}).items():
            if vals is None:
                key_exists.add(k)
            else:
                key_to_regex.setdefault(k, set()).update(vals)
    return key_to_regex, sorted(key_exists)


def fetch_elements(lat, lon, max_radius, POI_GROUPS_BY_KEY):
    key_to_vals, key_exists = _collect_key_specs(POI_GROUPS_BY_KEY)

    blocks = []
    for k, vals in key_to_vals.items():
        rx = _regex_from_values(vals)
        if not rx:
            continue
        blocks.append(
            f"""
            node(around:{max_radius},{lat},{lon})["{k}"~"{rx}"];
            way(around:{max_radius},{lat},{lon})["{k}"~"{rx}"];
            relation(around:{max_radius},{lat},{lon})["{k}"~"{rx}"];
            """
        )

    for k in key_exists:
        blocks.append(
            f"""
            node(around:{max_radius},{lat},{lon})["{k}"];
            way(around:{max_radius},{lat},{lon})["{k}"];
            relation(around:{max_radius},{lat},{lon})["{k}"];
            """
        )

    query = f"""
    [out:json][timeout:60];
    (
      {''.join(blocks)}
    );
    out center tags;
    """

    result = safe_overpass_post(query)
    return (result or {}).get("elements", [])

# -----------------------------
# Feature builder (flat dict)
# -----------------------------

def get_poi_features(
    lat,
    lon,
    bands=((0, 50), (50, 200), (200, 800)),
    missing_value=-1.0,
    add_any=True,
    add_density=False,
    add_open_now=True,
    tz="Europe/Amsterdam",
    now=None,
    self_exclude_keywords=("albert heijn", "ah"),
):
    """
    Return one-level dict of aggregated POI signals for non-overlapping distance bands.

    bands: iterable of (low, high) in meters, e.g. ((0,50),(50,200),(200,800))
           Convention:
             - first band includes [0, high]
             - subsequent include (low, high]
    """

    # sanitize/sort bands
    bands = [(float(lo), float(hi)) for lo, hi in bands]
    bands = sorted(bands, key=lambda x: x[1])
    if not bands or any(hi <= 0 for _, hi in bands):
        raise ValueError("bands must contain positive upper bounds")

    # ensure monotonic and non-negative lows
    for i, (lo, hi) in enumerate(bands):
        if lo < 0 or hi <= lo:
            raise ValueError(f"Invalid band {bands[i]}: require 0 <= low < high")
        if i > 0 and lo != bands[i-1][1]:
            # not strictly required, but usually what you want for contiguous bands
            # you can relax this if you want gaps/overlaps intentionally
            pass

    max_r = int(max(hi for _, hi in bands))

    # Build group lookups (same as original)
    group_names = []
    value_to_group = {}
    any_value_group = {}

    for k, gmap in POI_GROUPS_BY_KEY.items():
        value_to_group[k] = {}
        for g, vals in gmap.items():
            group_names.append(g)
            if vals is None:
                any_value_group[k] = g
            else:
                for v in vals:
                    if v not in value_to_group[k]:
                        value_to_group[k][v] = g

    group_names = sorted(set(group_names))

    # One Overpass call for max radius
    elements = fetch_elements(lat, lon, max_r, POI_GROUPS_BY_KEY)

    # Init output feature dict
    feats = {}
    for (lo, hi) in bands:
        suffix = f"b{int(lo)}_{int(hi)}"
        for g in group_names:
            feats[f"poi_{g}_count_{suffix}"] = 0.0
            if add_any:
                feats[f"poi_{g}_any_{suffix}"] = 0.0
            feats[f"poi_{g}_min_dist_m_{suffix}"] = missing_value
            feats[f"poi_{g}_mean_top3_dist_m_{suffix}"] = missing_value
            if add_density:
                feats[f"poi_{g}_density_{suffix}"] = 0.0
            if add_open_now and g in OPEN_NOW_GROUP_WHITELIST:
                feats[f"poi_{g}_open_now_count_{suffix}"] = 0.0
                if add_any and g in OPEN_NOW_GROUP_WHITELIST:
                    feats[f"poi_{g}_open_now_any_{suffix}"] = 0.0

    # distance buckets per band for min/mean stats
    dists = {i: {g: [] for g in group_names} for i in range(len(bands))}

    # timezone for opening_hours
    if add_open_now:
        if now is None:
            now = datetime.now(ZoneInfo(tz))
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ZoneInfo(tz))

    # Process elements once, assign each to a single band
    for el in elements:
        tags = el.get("tags", {}) or {}
        p_lat, p_lon = osm_point(el)
        if p_lat is None or p_lon is None:
            continue

        dist = haversine_m(lat, lon, p_lat, p_lon)
        if dist > max_r:
            continue

        band_idx = _band_for_distance(dist, bands)
        if band_idx is None:
            continue

        matched_groups = set()
        for k in POI_GROUPS_BY_KEY.keys():
            if k not in tags:
                continue
            val = tags.get(k)
            if val is None:
                continue

            # optional self-exclusion for shops
            if k == "shop" and self_exclude_keywords:
                name = tags.get("name") or ""
                brand = tags.get("brand") or ""
                operator = tags.get("operator") or ""
                txt = f"{name} {brand} {operator}".lower()
                if any(kw.lower() in txt for kw in self_exclude_keywords):
                    continue

            if k in any_value_group:
                matched_groups.add(any_value_group[k])

            g = value_to_group.get(k, {}).get(val)
            if g:
                matched_groups.add(g)

        if not matched_groups:
            continue

        open_flag = None
        if add_open_now:
            oh = tags.get("opening_hours")
            amen = tags.get("amenity")
            shp = tags.get("shop")
            is_ts = (amen in TIME_SENSITIVE_AMENITY) or (shp in TIME_SENSITIVE_SHOP)
            if is_ts and oh:
                open_flag = is_open_now(oh, now)

        lo, hi = bands[band_idx]
        suffix = f"b{int(lo)}_{int(hi)}"

        for g in matched_groups:
            feats[f"poi_{g}_count_{suffix}"] += 1.0
            if add_any:
                feats[f"poi_{g}_any_{suffix}"] = 1.0
            dists[band_idx][g].append(dist)

            if add_open_now and g in OPEN_NOW_GROUP_WHITELIST and open_flag == 1:
                feats[f"poi_{g}_open_now_count_{suffix}"] += 1.0
                if add_any and g in OPEN_NOW_GROUP_WHITELIST:
                    feats[f"poi_{g}_open_now_any_{suffix}"] = 1.0

    # finalize min/mean_top3 per band
    for i, (lo, hi) in enumerate(bands):
        suffix = f"b{int(lo)}_{int(hi)}"
        for g in group_names:
            arr = sorted(dists[i][g])
            if arr:
                feats[f"poi_{g}_min_dist_m_{suffix}"] = float(arr[0])
                top3 = arr[:3]
                feats[f"poi_{g}_mean_top3_dist_m_{suffix}"] = float(sum(top3) / len(top3))

        if add_density:
            # density per annulus area (π(hi^2 - lo^2))
            area = math.pi * (hi**2 - lo**2)
            if area > 0:
                for g in group_names:
                    feats[f"poi_{g}_density_{suffix}"] = feats[f"poi_{g}_count_{suffix}"] / area

    return feats