from __future__ import annotations

from typing import Dict, Iterable, Tuple, Optional
import numpy as np
import pandas as pd


def gen_synthetic_events(
    keys: Iterable[Tuple[int, pd.Timestamp]],
    store_locations: Dict[int, Tuple[float, float]],
    seed: int = 42,
    base_lam_24h: float = 3.0,     # baseline expected events in next 24h
    tau_hours: float = 24.0,       # smoothness (12–48 typical)
    innov_sigma: float = 0.12,     # hourly innovation scale (stationary)
    store_bias_sigma: float = 0.25,
    dist_decay_km: float = 3.0,
) -> pd.DataFrame:
    """
    Generate synthetic event features for (store_id, ts_hour) keys.

    Output columns (flat):
      events_count_b_b0_1km_t_b0_3h
      events_count_b_b0_1km_t_b3_24h
      events_weighted_sum_t0_24h
      unique_venues_t0_24h
      max_events_same_venue_t0_24h
      events_count_b_b1_3km_t_b0_3h
      events_min_dist_km_b_b0_1km_t_b0_3h
      events_music_count_b_b0_1km_t_b0_3h
      events_nightlife_count_b_b0_1km_t_b0_3h
      events_family_count_b_b0_1km_t_b0_3h
    """
    # --- build df from keys ---
    df = pd.DataFrame(list(keys), columns=["store_id", "ts_hour"])
    df["ts_hour"] = pd.to_datetime(df["ts_hour"])
    # if df["ts_hour"].dt.tz is None:
    #     df["ts_hour"] = df["ts_hour"].dt.tz_localize("UTC")
    # else:
    #     df["ts_hour"] = df["ts_hour"].dt.tz_convert("UTC")
    df["ts_hour"] = df["ts_hour"].dt.floor("h")

    # Drop duplicates just in case; sort for deterministic sequential evolution
    df = df.drop_duplicates(["store_id", "ts_hour"]).sort_values(["store_id", "ts_hour"], kind="mergesort")
    df = df.reset_index(drop=True)

    # --- attach lat/lon ---
    # vectorized map; assumes all store_ids exist in dict
    df["lat"] = df["store_id"].map(lambda sid: store_locations[sid][0]).astype(float)
    df["lon"] = df["store_id"].map(lambda sid: store_locations[sid][1]).astype(float)

    # --- time regime multipliers (vectorized) ---
    dow = df["ts_hour"].dt.weekday.to_numpy()
    hour = df["ts_hour"].dt.hour.to_numpy()
    month = df["ts_hour"].dt.month.to_numpy()

    is_weekend = (dow >= 5)
    is_evening = (hour >= 17) & (hour <= 23)
    holiday_season = (month == 11) | (month == 12)
    summer = (month >= 6) & (month <= 8)

    time_mult = np.ones(len(df), dtype=np.float64)
    time_mult *= np.where(is_weekend, 1.35, 1.00)
    time_mult *= np.where(is_evening, 1.20, 1.00)
    time_mult *= np.where(holiday_season, 1.10, 1.00)
    time_mult *= np.where(summer, 1.05, 1.00)

    # “Soon” share (0–3h) varies with evening/weekend
    p_0_3h = 0.20 + 0.05 * is_evening.astype(float) + 0.03 * is_weekend.astype(float)
    p_0_3h = np.clip(p_0_3h, 0.12, 0.35)
    p_3_24h = 1.0 - p_0_3h

    # --- store-level stable factors ---
    # loc_factor in ~[0.85, 1.15]
    loc_factor = 0.85 + 0.30 * (0.5 + 0.5 * np.sin((df["lat"].to_numpy() + df["lon"].to_numpy()) * 3.0))

    # distance bucket shares: store-specific (compute per row but based on loc_factor)
    p_0_1km = np.clip(0.30 + 0.10 * (loc_factor - 1.0) / 0.15, 0.20, 0.45)
    p_1_3km = np.clip(0.45 - 0.08 * (loc_factor - 1.0) / 0.15, 0.30, 0.55)

    # expected decay factor for weighted sum (approx)
    # If d ~ Exp(scale=s), E[exp(-d/L)] = 1/(1+s/L)
    s_km = 1.8
    edecay = 1.0 / (1.0 + s_km / dist_decay_km)

    # --- outputs ---
    out_cols = [
        "events_count_b_b0_1km_t_b0_3h",
        "events_count_b_b0_1km_t_b3_24h",
        "events_weighted_sum_t0_24h",
        "unique_venues_t0_24h",
        "max_events_same_venue_t0_24h",
        "events_count_b_b1_3km_t_b0_3h",
        "events_min_dist_km_b_b0_1km_t_b0_3h",
        "events_music_count_b_b0_1km_t_b0_3h",
        "events_nightlife_count_b_b0_1km_t_b0_3h",
        "events_family_count_b_b0_1km_t_b0_3h",
    ]
    for c in out_cols:
        df[c] = 0.0

    # --- continuous-time AR(1)/OU parameters ---
    # per-hour reversion phi = exp(-1/tau)
    phi_1h = float(np.exp(-1.0 / max(tau_hours, 1e-9)))

    # --- simulate per store (fast + correct handling of missing hours) ---
    # group indices for each store
    rng_master = np.random.default_rng(seed)

    # persistent store bias: fixed across time; deterministic given seed and store_id ordering
    store_ids_unique = df["store_id"].unique()
    # stable mapping store_id -> bias using a seeded RNG and sorting store ids
    store_ids_sorted = np.sort(store_ids_unique)
    bias_rng = np.random.default_rng(seed + 12345)
    store_bias_map = {sid: bias_rng.normal(0.0, store_bias_sigma) for sid in store_ids_sorted}

    for sid, g in df.groupby("store_id", sort=False):
        idx = g.index.to_numpy()
        # store-specific rng (deterministic)
        rng = np.random.default_rng(seed + int(sid) * 1000003)

        # time deltas in hours (including gaps)
        t = g["ts_hour"].to_numpy()
        # convert to int64 ns and compute dt in hours
        t_ns = t.astype("datetime64[ns]").astype(np.int64)
        dt_hours = np.empty(len(idx), dtype=np.float64)
        dt_hours[0] = 1.0
        if len(idx) > 1:
            dt_hours[1:] = (t_ns[1:] - t_ns[:-1]) / (3600 * 1e9)
            dt_hours[1:] = np.clip(dt_hours[1:], 1.0, 24.0 * 14)  # cap huge gaps for stability

        # latent state evolution (stationary OU discretization)
        # state_t = phi^dt * state_prev + sigma * sqrt(1 - phi^(2dt)) * eps
        state = rng.normal(0.0, innov_sigma)  # initial
        sbias = store_bias_map[sid]

        # pull row-wise arrays for this store
        tm = time_mult[idx]
        p03 = p_0_3h[idx]
        p324 = p_3_24h[idx]
        p01 = p_0_1km[idx]
        p13 = p_1_3km[idx]
        lf = loc_factor[idx]

        # outputs arrays
        c0_3_01 = np.zeros(len(idx), dtype=np.int32)
        c3_24_01 = np.zeros(len(idx), dtype=np.int32)
        c0_3_13 = np.zeros(len(idx), dtype=np.int32)
        n24 = np.zeros(len(idx), dtype=np.int32)
        wsum = np.zeros(len(idx), dtype=np.float64)
        uniq = np.zeros(len(idx), dtype=np.float64)
        mxsv = np.zeros(len(idx), dtype=np.float64)
        mind = np.full(len(idx), -1.0, dtype=np.float64)
        music = np.zeros(len(idx), dtype=np.int32)
        night = np.zeros(len(idx), dtype=np.int32)
        fam = np.zeros(len(idx), dtype=np.int32)

        for j in range(len(idx)):
            dt = dt_hours[j]
            phi_dt = phi_1h ** dt
            # stationary innovation scaling
            innov_scale = innov_sigma * np.sqrt(max(0.0, 1.0 - (phi_dt ** 2)))
            state = phi_dt * state + innov_scale * rng.normal(0.0, 1.0)

            lam24 = base_lam_24h * lf[j] * tm[j] * np.exp(sbias + state)
            lam24 = float(np.clip(lam24, 0.01, 50.0))

            # total events in 0-24h
            n = int(rng.poisson(lam24))
            n24[j] = n

            # bucket counts via Poisson thinning
            c0_3_01[j] = int(rng.poisson(lam24 * p03[j] * p01[j]))
            c3_24_01[j] = int(rng.poisson(lam24 * p324[j] * p01[j]))
            c0_3_13[j] = int(rng.poisson(lam24 * p03[j] * p13[j]))

            # min dist within 0-1km & 0-3h
            k = c0_3_01[j]
            if k > 0:
                scale = 0.35 / k
                md = rng.exponential(scale=scale)
                md = float(np.clip(md + rng.normal(0.0, 0.02), 0.05, 1.0))
                mind[j] = md
            else:
                mind[j] = -1.0

            # category splits in immediate close bucket (0-1km, 0-3h)
            # probabilities vary with time:
            pmusic = 0.18 + (0.12 if is_weekend[idx[j]] else 0.0) + (0.10 if is_evening[idx[j]] else 0.0)
            pnight = 0.08 + (0.16 if is_evening[idx[j]] else 0.0) + (0.05 if is_weekend[idx[j]] else 0.0)
            pfam = 0.12 + (0.07 if (hour[idx[j]] <= 18) else 0.0) + (0.04 if is_weekend[idx[j]] else 0.0)

            pmusic = float(np.clip(pmusic, 0.05, 0.65))
            pnight = float(np.clip(pnight, 0.03, 0.65))
            pfam = float(np.clip(pfam, 0.03, 0.65))

            m = rng.binomial(k, pmusic)
            r1 = k - m
            pnight2 = pnight / max(1e-9, (1.0 - pmusic))
            pnight2 = float(np.clip(pnight2, 0.0, 1.0))
            nli = rng.binomial(r1, pnight2)
            r2 = r1 - nli
            pfam2 = pfam / max(1e-9, (1.0 - pmusic - pnight))
            pfam2 = float(np.clip(pfam2, 0.0, 1.0))
            f = rng.binomial(r2, pfam2)

            music[j], night[j], fam[j] = int(m), int(nli), int(f)

            # weighted sum: approx expected (lam * edecay) with mild noise
            # (fast; avoids simulating each event distance)
            val = lam24 * edecay * (0.90 + 0.20 * rng.random())
            wsum[j] = val

            # venues: approximate unique venues & max same venue
            if n == 0:
                uniq[j] = 0.0
                mxsv[j] = 0.0
            else:
                vcount = int(np.clip(np.rint(np.sqrt(n) + 1.0), 1, max(1, n)))
                unique = vcount * (1.0 - np.exp(-n / max(vcount, 1)))
                unique = float(min(unique, n))
                uniq[j] = unique
                base = np.ceil(n / max(unique, 1.0))
                bump = 1.0 if (rng.random() < 0.25 and n >= 3) else 0.0
                mxsv[j] = float(base + bump)

        # write back
        df.loc[idx, "events_count_b_b0_1km_t_b0_3h"] = c0_3_01.astype(np.float32)
        df.loc[idx, "events_count_b_b0_1km_t_b3_24h"] = c3_24_01.astype(np.float32)
        df.loc[idx, "events_count_b_b1_3km_t_b0_3h"] = c0_3_13.astype(np.float32)

        df.loc[idx, "events_min_dist_km_b_b0_1km_t_b0_3h"] = mind.astype(np.float32)
        df.loc[idx, "events_music_count_b_b0_1km_t_b0_3h"] = music.astype(np.float32)
        df.loc[idx, "events_nightlife_count_b_b0_1km_t_b0_3h"] = night.astype(np.float32)
        df.loc[idx, "events_family_count_b_b0_1km_t_b0_3h"] = fam.astype(np.float32)

        df.loc[idx, "events_weighted_sum_t0_24h"] = wsum.astype(np.float32)
        df.loc[idx, "unique_venues_t0_24h"] = uniq.astype(np.float32)
        df.loc[idx, "max_events_same_venue_t0_24h"] = mxsv.astype(np.float32)

    # keep only required columns
    keep = ["store_id", "ts_hour"] + out_cols
    return df[keep]


# ------------------------------------------------------------------
ORDERS_PATH = "../data/parquets/orders_multistore_2026.parquet"
ORDERS_IS_CSV = False

OUT_EVENTS_PATH = "../data/parquets/events_hourly.parquet"
# ------------------------------------------------------------------

# Load orders
orders = pd.read_csv(ORDERS_PATH) if ORDERS_IS_CSV else pd.read_parquet(ORDERS_PATH)

orders["order_timestamp"] = pd.to_datetime(orders["order_timestamp"])
orders["ts_hour"] = orders["order_timestamp"].dt.floor("h")

keys = list(
    orders[["store_id", "ts_hour"]]
    .drop_duplicates()
    .itertuples(index=False, name=None)
)

# Load stores and build store_locations dict
stores_df = pd.read_csv("stores.csv")
store_locations = (
    stores_df.set_index("store_id")[["latitude", "longitude"]]
    .apply(tuple, axis=1)
    .to_dict()
)

events_hourly = gen_synthetic_events(
    keys,
    store_locations=store_locations,
    seed=7,
    base_lam_24h=3.0,
    tau_hours=30.0,      # slow changes ~30h
    innov_sigma=0.12,
)

events_hourly.to_parquet(OUT_EVENTS_PATH, index=False)
print("Wrote:", OUT_EVENTS_PATH, "rows:", len(events_hourly))