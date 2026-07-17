import pandas as pd
import requests
import re

BASE_API = "https://opendata.cbs.nl/ODataApi/OData/"
DATASET_COO = "85640NED" 
DATASET_AGE = "83502NED"

# Dimension mapping (stays on ODataApi fine)
dim = requests.get(f"{BASE_API}{DATASET_COO}/Herkomstland", headers={"Accept": "application/json"}).json()["value"]
key_title_map = {row["Key"]: row["Title"] for row in dim}

url = f"{BASE_API}{DATASET_COO}/TypedDataSet"

# NOTE: put all filters in the API call

def get_age_group(postcode:int, year:int) -> dict:
    
    postcode_str = f"PC{int(postcode):04d}"
    period_str = f"{int(year):04d}JJ00"

    params = {
        "$filter": f"(Postcode  eq '{postcode_str}  ') and (Perioden eq '{period_str}')",
        "$format": "json",
    }
  
    data = requests.get(url, params=params, headers={"Accept": "application/json"}).json()
    df = pd.DataFrame(data.get("value", []))
    
    df["Herkomstland_name"] = df["Herkomstland"].map(key_title_map)
    
    agg = (
        df.groupby("Herkomstland_name", as_index=False)
          .agg({"Bevolking_1": "sum"})
          .sort_values("Bevolking_1", ascending=False)
          .reset_index(drop=True)
    )
    
    # --- total population ---
    total_pop = agg.loc[agg["Herkomstland_name"].eq("Totaal"), "Bevolking_1"].iloc[0]
    
    # detect whether non-total values are shares (typical: <= 1)
    non_total = agg.loc[~agg["Herkomstland_name"].eq("Totaal"), "Bevolking_1"]
    looks_like_share = non_total.dropna().le(1.0).all()
    
    # build a lookup series
    s = agg.set_index("Herkomstland_name")["Bevolking_1"]
    
    # helper: sum over labels, missing labels contribute 0
    def sum_labels(labels):
        return float(s.reindex(labels).fillna(0.0).sum())
    
    # --- define your buckets (by labels in your agg) ---
    buckets = {
        "total_population": ["Totaal"],
    
        "pct_pop_dutch": ["Nederland"],
    
        "pct_pop_surinam": ["Suriname"],
    
        "pct_pop_asia": ["Azië (exclusief Indonesië en Turkije)"],
    
        "pct_pop_africa": ["Afrika (exclusief Marokko)"],
    
        "pct_pop_islamic": ["Marokko", "Turkije"], 
        
        # Prefer the already-aggregated CBS row if it exists:
        "pct_pop_europe": ["Europa (exclusief Nederland)"],
    }
    
    out = {}
    for cat, labels in buckets.items():
        val = sum_labels(labels)
    
        if cat == "total_population":
            # Total is a count already
            val = total_pop
        else:
            val = val / total_pop if total_pop else float("nan")
    
        out[cat] = val
    
    return out


def _parse_age_bounds(title: str):
    """
    Parse CBS Leeftijd_title patterns such as:
      - '0 tot 5 jaar'  -> (0, 5)
      - '95 jaar of ouder' -> (95, None)
      - 'Totaal leeftijd' -> (None, None)
    Returns (lower_inclusive, upper_exclusive_or_None)
    """
    if not isinstance(title, str):
        return (None, None)

    t = title.strip().lower()

    # total bucket row
    if "totaal" in t:
        return (None, None)

    # 'X tot Y jaar'
    m = re.search(r"(\d+)\s+tot\s+(\d+)\s+jaar", t)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2))
        return (lo, hi)

    # 'X jaar of ouder'
    m = re.search(r"(\d+)\s+jaar\s+of\s+ouder", t)
    if m:
        lo = int(m.group(1))
        return (lo, None)

    # Sometimes: 'X tot Y jaar' variants with hyphen or en-dash
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s+jaar", t)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2))
        return (lo, hi)

    return (None, None)


def get_country_origin(postcode: int, year: int) -> dict:
    """
    Returns a dict with:
      - total_population
      - pct_age_0_25
      - pct_age_25_65
      - pct_age_gt_65

    Uses CBS ODataApi dataset 83502NED:
      population by sex, age, 4-digit postcode, at 1 January. [1](https://www.cbs.nl/nl-nl/cijfers/detail/83502NED)[2](https://dataportal.cbs.nl/detail/CBS/83502NED)
    """
    postcode_str = f"PC{int(postcode):04d}"
    period_str = f"{int(year):04d}JJ00"

    # --- Build dimension maps (Leeftijd + Geslacht) ---
    leeftijd_dim = requests.get(
        f"{BASE_API}{DATASET_AGE}/Leeftijd",
        headers={"Accept": "application/json"}
    ).json()["value"]
    leeftijd_map = {row["Key"]: row["Title"] for row in leeftijd_dim}

    geslacht_dim = requests.get(
        f"{BASE_API}{DATASET_AGE}/Geslacht",
        headers={"Accept": "application/json"}
    ).json()["value"]
    geslacht_map = {row["Title"].strip(): row["Key"] for row in geslacht_dim}

    # Use "Totaal mannen en vrouwen"
    geslacht_key = geslacht_map.get("Totaal mannen en vrouwen")
    if geslacht_key is None:
        # Fallback: pick the first title containing 'Totaal' to be defensive
        for title, key in geslacht_map.items():
            if "Totaal" in title:
                geslacht_key = key
                break

    if geslacht_key is None:
        raise RuntimeError("Could not resolve Geslacht key for 'Totaal mannen en vrouwen'.")

    # --- Pull typed dataset rows for this postcode/year/sex ---
    url = f"{BASE_API}{DATASET_AGE}/TypedDataSet"

    # CBS postcode fields sometimes carry trailing spaces; we try both.
    filter_variants = [
        f"(Postcode eq '{postcode_str}  ') and (Perioden eq '{period_str}') and (Geslacht eq '{geslacht_key}')",
        f"(Postcode eq '{postcode_str}  ') and (Perioden eq '{period_str}') and (Geslacht eq '{geslacht_key}')",
    ]

    df = None
    last_payload = None
    for flt in filter_variants:
        params = {"$filter": flt, "$format": "json"}
        payload = requests.get(url, params=params, headers={"Accept": "application/json"}).json()
        last_payload = payload
        cand = pd.DataFrame(payload.get("value", []))
        if not cand.empty:
            df = cand
            break

    if df is None or df.empty:
        # helpful debugging info, without being too verbose
        raise ValueError(
            f"No data returned for postcode={postcode_str}, year={year}. "
            f"Last response keys: {list((last_payload or {}).keys())}"
        )

    # --- Identify the population column (usually 'Bevolking_1' in TypedDataSet) ---
    pop_cols = [c for c in df.columns if c.lower().startswith("bevolking")]
    if not pop_cols:
        raise RuntimeError(f"Could not find a Bevolking column in response. Columns: {df.columns.tolist()}")
    pop_col = pop_cols[0]

    # --- Add Leeftijd_title and parse bounds ---
    df["Leeftijd_title"] = df["Leeftijd"].map(leeftijd_map)
    df[["age_lo", "age_hi"]] = df["Leeftijd_title"].apply(
        lambda s: pd.Series(_parse_age_bounds(s))
    )

    # Total population: prefer explicit 'Totaal leeftijd' row if present
    total_rows = df[df["Leeftijd_title"].str.contains("Totaal", case=False, na=False)]
    if not total_rows.empty:
        total_pop = float(total_rows[pop_col].sum())
    else:
        # fallback: sum all age-band rows that have a parsed lower bound
        total_pop = float(df[df["age_lo"].notna()][pop_col].sum())

    if total_pop == 0:
        return {
            "pct_age_0_25": float("nan"),
            "pct_age_25_65": float("nan"),
            "pct_age_gt_65": float("nan"),
        }

    # --- Aggregate into buckets ---
    # For band [lo, hi): allocate to bucket by band midpoint.
    # (CBS age bands are non-overlapping and narrow enough that midpoint allocation is safe.)
    bands = df[df["age_lo"].notna()].copy()
    bands["mid"] = bands.apply(
        lambda r: (r["age_lo"] + (r["age_hi"] if pd.notna(r["age_hi"]) else r["age_lo"] + 5)) / 2.0,
        axis=1
    )

    pop_0_25 = float(bands[(bands["mid"] < 25)][pop_col].sum())
    pop_25_65 = float(bands[(bands["mid"] >= 25) & (bands["mid"] < 65)][pop_col].sum())

    # >65 includes open-ended groups like '95 jaar of ouder'
    # For those, mid will be >= lo, so they'll fall in this bucket.
    pop_gt_65 = float(bands[(bands["mid"] >= 65)][pop_col].sum())

    return {
        "pct_age_0_25": pop_0_25 / total_pop,
        "pct_age_25_65": pop_25_65 / total_pop,
        "pct_age_gt_65": pop_gt_65 / total_pop,
    }