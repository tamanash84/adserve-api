import pandas as pd
import numpy as np
from datetime import datetime

def make_multistore_pos(
    orders_csv_path: str,
    out_path: str = "orders_multistore_2026.parquet",
    n_stores: int = 5,
    seed: int = 42,
    year: int = 2026,
    store_col: str = "store_id",
    timestamp_col: str = "order_timestamp",
):
    """
    Reads Instacart orders.csv and:
      1) assigns store_id in roughly equal parts
      2) assigns a timestamp within [01-01-year, 12-31-year] such that:
         - weekday matches order_dow (0..6)
         - hour matches order_hour_of_day (0..23)

    Output is saved to Parquet by default (fast & compact). Use .csv if you prefer.
    """
    rng = np.random.default_rng(seed)

    df = pd.read_csv(orders_csv_path)

    required = {"order_dow", "order_hour_of_day"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in orders.csv: {missing}")

    # --- 1) assign stores in roughly equal parts (balanced random) ---
    n = len(df)
    store_ids = np.tile(np.arange(1, n_stores + 1, dtype=np.int16), int(np.ceil(n / n_stores)))[:n]
    rng.shuffle(store_ids)
    df[store_col] = store_ids

    # --- 2) build date pools for each weekday in the target year ---
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    all_days = pd.date_range(start, end, freq="D")

    # Pandas weekday: Monday=0 ... Sunday=6
    # Instacart order_dow is commonly 0..6 (often 0=Sunday in some versions),
    # but your requirement says "based on dow", so we map directly by default.
    #
    # IMPORTANT: If your Instacart order_dow uses 0=Sunday, you must remap.
    # See REMAP section below.
    weekday_to_dates = {dow: all_days[all_days.weekday == dow].to_numpy() for dow in range(7)}

    # --- 3) assign dates per order_dow evenly across the year ---
    ts = np.empty(n, dtype="datetime64[ns]")

    # Validate order_dow range
    if df["order_dow"].min() < 0 or df["order_dow"].max() > 6:
        raise ValueError("order_dow must be in [0..6].")

    for dow in range(7):
        idx = df.index[df["order_dow"].values == dow].to_numpy()
        if idx.size == 0:
            continue

        dates = weekday_to_dates[dow]
        if dates.size == 0:
            raise ValueError(f"No dates for dow={dow} in year {year} (unexpected).")

        # shuffle indices, then round-robin dates (even spread)
        rng.shuffle(idx)
        chosen_dates = dates[np.arange(idx.size) % dates.size]

        # hours from file
        hours = df.loc[idx, "order_hour_of_day"].to_numpy(dtype=np.int16)

        # random minute/second for realism
        minutes = rng.integers(0, 60, size=idx.size, dtype=np.int16)
        seconds = rng.integers(0, 60, size=idx.size, dtype=np.int16)

        # build timestamps: date + hour + minute + second
        # Convert chosen_dates to pandas Timestamp for vector ops
        base = pd.to_datetime(chosen_dates)
        built = base + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m") + pd.to_timedelta(seconds, unit="s")
        ts[idx] = built.to_numpy()

    df[timestamp_col] = pd.to_datetime(ts)

    # Optional: sort by timestamp (typical POS style)
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # Save
    if out_path.lower().endswith(".csv"):
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    return df


if __name__ == "__main__":
    # Example usage:
    df_out = make_multistore_pos(
        orders_csv_path="../data/Instacart/orders.csv",
        out_path="../data/parquets/orders_multistore_2026.parquet",
        n_stores=5,
        seed=7,
        year=2026
    )
    print(df_out[[ "order_id", "order_dow", "order_hour_of_day", "store_id", "order_timestamp"]].head())