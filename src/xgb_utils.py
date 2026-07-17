import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import json
import duckdb
import xgboost as xgb
from config import Paths
import glob


def score_ensemble(xgb_ensemble: List[xgb.Booster], 
                   xgb_context: List[Dict[str, Any]]):
    """
    Returns:
      mean:  (K,) mean across models
      var:   (K,) variance across models
    """
    K = len(xgb_context)
    if not xgb_ensemble:   # None or []
        preds = np.full((1, K), 1.0 / max(K, 1), dtype=np.float32)
        mean = preds[0]
        var = np.zeros(K, dtype=np.float32)
        return preds, mean, var

    M = len(xgb_ensemble)
    preds = np.zeros((M, K), dtype=np.float32)
    
    """
    Convert list-of-dicts to DMatrix with stable column order.
    Missing columns are filled with 0. Extra columns are ignored.
    """    
    with open("xgb_feature_names.json") as f:
        xgb_features = json.load(f)
        
    #X = [xgb_context[name] for name in xgb_features["feature_names"]]
    X = [[row.get(name, 0) for name in xgb_features["feature_names"]]
        for row in xgb_context]
    dmat = xgb.DMatrix(X, feature_names=xgb_features["feature_names"])
    
    for i, m in enumerate(xgb_ensemble):
        preds[i] = m.predict(dmat)

    return preds, preds.mean(axis=0), preds.var(axis=0)


def load_window_parquet(glob_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Loads partitioned parquet rows where date in [start_date, end_date).
    Expects a 'date' column (string 'YYYY-MM-DD') in the parquet.
    """
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT *
        FROM '{glob_path}'
        WHERE date > '{start_date}' AND date <= '{end_date}'
    """).df()
    con.close()
    return df


def compute_dr_target(df: pd.DataFrame,
                      min_prop: float = 0.01,
                      clip01: bool = True) -> pd.DataFrame:
    """    DR target:
      dr_target = xgb_prob_mean + (reward - xgb_prob_mean)/propensity
      dr_target_clipped = clip(dr_target, 0, 1)
      xgb_prob_mean: mean of M ensemble models
    """
    # required cols check
    required = ["reward", "propensity", "xgb_prob_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for DR target: {missing}")

    p = df["propensity"].astype(float).to_numpy()
    p = np.clip(p, min_prop, 1.0)

    r = df["reward"].astype(float).to_numpy()
    m = df["xgb_prob_mean"].astype(float).to_numpy()

    dr = m + (r - m) / p
    if clip01:
        dr = np.clip(dr, 0.0, 1.0)

    df = df.copy()
    df["dr_target"] = dr
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Builds X and y where y is DR target.
    categorical -> category codes; then drop non-features.
    """
    with open("xgb_feature_names.json") as f:
        xgb_features = json.load(f)

    # Encode categoricals (same approach as your training code)
    for col in xgb_features["categorical"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    if "dr_target" not in df.columns:
        raise ValueError("df must contain dr_target before building features.")

    y = df["dr_target"].astype(float).to_numpy()   
    X = df[xgb_features["feature_names"]]

    return X, y


def build_model(
    run_day: date | None = None,
    lookback_days: int = 1,
    M: int = 10,
    seed: int = 42,
    min_prop: float = 0.01,
    num_boost_round: int = 300,
    ) -> List[str]:
    """
    Nightly training job for a bootstrapped XGBoost ensemble.

    Side effects:
      - Loads training data from parquet
      - Computes DR targets
      - Trains M bootstrapped XGBoost models
      - Saves models + feature order to out_dir

    Returns:
      - None (raises on failure)
    """
  
    if run_day is None:
        run_day = date.today()

    start_day = (run_day - timedelta(days=lookback_days)).isoformat()
    end_day = run_day.isoformat()

    print(f"Training lookback window ={lookback_days} days, run_day={run_day.isoformat()}, M={M}")
    
    glob_path = Paths.TRAIN_DAILY_PATTERN.as_posix()

    df = load_window_parquet(glob_path, start_day, end_day)

    df["xgb_prob_mean"] =  df["xgb_prob_mean"].fillna(0.)
    if len(df) == 0:
        raise ValueError("No training data found for the given window")

    df = compute_dr_target(df, min_prop=min_prop, clip01=True)
    X, y = build_features(df)

    print(f"Feature matrix: {X.shape}, y mean={float(np.mean(y)):.4f}")

    """
    Train M bootstrapped XGB models and save them.
    """
    
    out_dir = Paths.XGB_MODEL
    os.makedirs(out_dir, exist_ok=True)
    # Uses reg:logistic (probability-like output).
    params = {
        "objective": "reg:logistic",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "tree_method": "hist",
        "verbosity": 0,
    }

    rng = np.random.default_rng(seed)
    n = len(X)

    # Ensure stable feature ordering
    feature_names = list(X.columns)
    
    out_paths = []
    for i in range(M):
        # Bootstrap sample rows with replacement
        idx = rng.integers(0, n, size=n)
        Xb = X.iloc[idx]
        yb = y[idx]

        dtrain = xgb.DMatrix(Xb, label=yb, feature_names=feature_names)
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        # (Optional but good) also set directly:
        model.feature_names = feature_names
        out_path = out_dir / f"xgb_ens_{end_day}_{i:02d}.json"
        model.save_model(out_path)
        out_paths.append(out_path)
        print(f"[{i+1:02d}/{M}] saved {out_path}")

    print(f"Done. Trained {M} models for {end_day} in {out_dir}")
    return out_paths


# def concat_hourlies_to_day(
#     ts: datetime,
#     ) -> str:
    
#     day = ts.date().isoformat()

#     out_path = Paths.TRAIN_DAILY_PATTERN.as_posix().replace("date=*", f"date={day}")
#     glob_path = Paths.TRAIN_HOURLY_PATTERN.as_posix().replace("date=*", f"date={day}")

#     con = duckdb.connect()
#     try:
#         # This will read all matching hour files and union them
#         con.execute(f"""
#             COPY (
#                 SELECT *
#                 FROM read_parquet('{glob_path}')
#                 ORDER BY ts_impr
#             )
#             TO '{out_path}' (FORMAT PARQUET);
#         """)        

#     finally:
#         con.close()

#     return out_path


def concat_hourlies_to_day(ts: datetime) -> str:
    day = ts.date().isoformat()

    out_path = (
        Paths.TRAIN_DAILY_PATTERN
        .as_posix()
        .replace("date=*", f"date={day}")
    )

    glob_path = (
        Paths.TRAIN_HOURLY_PATTERN
        .as_posix()
        .replace("date=*", f"date={day}")
    )

    files = [f.replace("\\", "/") for f in glob.glob(glob_path)]

    if not files:
        raise FileNotFoundError(
            f"No parquet files found matching: {glob_path}"
        )

    file_list_sql = ",".join(f"'{f}'" for f in files)

    con = duckdb.connect()
    try:
        con.execute(f"""
            COPY (
                SELECT *
                FROM read_parquet([{file_list_sql}])
                ORDER BY ts_impr
            )
            TO '{out_path}'
            (FORMAT PARQUET);
        """)
    finally:
        con.close()

    return out_path

def load_ensemble(model_paths: List[str]) -> List[xgb.Booster]:
    xgb_ensemble: List[xgb.Booster] = []
    for p in model_paths:
        bst = xgb.Booster()
        bst.load_model(p)
        xgb_ensemble.append(bst)            

    feature_names = xgb_ensemble[0].feature_names  # read from trained model
    if feature_names is None:
        raise ValueError("Model has no feature_names; ensure training set feature_names on DMatrix.")
    
    return xgb_ensemble

################### Example #########################
# x = build_model(date(2026, 1, 2), 30)