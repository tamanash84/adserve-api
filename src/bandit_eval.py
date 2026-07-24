import pandas as pd
from config import Paths
from bandit_policy import VwAdfXGBPolicy
import ast


dates = ["2026-01-10", "2026-01-11"]

out = []

for dt in dates:
    
    path = str(Paths.TRAIN_DAILY_PATTERN).replace("*", dt)

    df = pd.read_parquet(path)
    
    daily = (
        df.groupby([
            pd.to_datetime(df["ts_impr"]).dt.date,
            "policy_name"
        ])
        .agg(
            impressions=("reward", "size"),
            purchases=("reward", "sum"),
            cvr=("reward", "mean")
        )
        .reset_index()
    )
    
    pivot = daily.pivot(
        index="ts_impr",
        columns="policy_name",
        values="cvr"
    )
    
    pivot["lift"] = (
        pivot["VW_ADF"] / pivot["Random"] - 1
    )
    
    out.append(pivot)

res = pd.concat(out, ignore_index=True)
print(res)

import numpy as np

def ips_evaluate(
    logged_events,
    new_policy,
):
    ips_sum = 0.0
    n = 0

    for row in logged_events.itertuples(index=False):        
        
        adf = ast.literal_eval(row.comment)
        
        if len(adf) == 0:
            continue
        
        chosen_idx = row.chosen_index
        reward = row.reward
        p0 = row.propensity
        p1 = new_policy.probs_given_adf(adf)[chosen_idx]
        w = p1 / p0

        ips_sum += w * reward
        n += 1

    return ips_sum / n

def snips_evaluate(
    logged_events,
    new_policy,
):
    weighted_rewards = 0.0
    total_weight = 0.0

    for row in logged_events.itertuples(index=False):

        adf = ast.literal_eval(row.comment)
        
        if len(adf) == 0:
            continue
        
        chosen_idx = row.chosen_index
        reward = row.reward
        p0 = row.propensity
        p1 = new_policy.probs_given_adf(adf)[chosen_idx]
        w = p1 / p0

        weighted_rewards += w * reward
        total_weight += w

    return weighted_rewards / total_weight

def doubly_robust(
    logged_events,
    new_policy,
):

    dr_values = []

    for row in logged_events.itertuples(index=False):

        adf = ast.literal_eval(row.comment)
        er = row.expected_rewards
        
        if len(adf) == 0:
            continue
        
        chosen_idx = row.chosen_index
        reward = row.reward
        p0 = row.propensity

        p1_vec = new_policy.probs_given_adf(adf)

        p1 = p1_vec[chosen_idx]

        q_logged = er[chosen_idx]

        q_new = np.dot(p1_vec, er)

        dr = q_new + (p1 / p0) * (reward - q_logged)

        dr_values.append(dr)

    return np.mean(dr_values)

def safe_literal_eval(x, default=None):
    if default is None:
        default = []

    if x is None:
        return default

    if isinstance(x, (list, tuple, dict)):
        return x

    try:
        return ast.literal_eval(str(x))
    except Exception:
        return default


def doubly_robust2(logged_events: pd.DataFrame, new_policy) -> float:
    dr_values = []

    required = {"comment", "chosen_index", "reward", "propensity", "expected_rewards"}
    if not required.issubset(logged_events.columns):
        return 0.0

    for row in logged_events.itertuples(index=False):
        adf = safe_literal_eval(getattr(row, "comment", None), default=[])
        er = getattr(row, "expected_rewards", None)

        if not adf or not er.any():
            continue

        chosen_idx = int(getattr(row, "chosen_index"))
        reward = float(getattr(row, "reward") or 0.0)
        p0 = float(getattr(row, "propensity") or 0.0)

        if p0 <= 0:
            continue

        p1_vec = np.asarray(new_policy.probs_given_adf(adf), dtype=float)
        er = np.asarray(er, dtype=float)

        if chosen_idx >= len(p1_vec) or chosen_idx >= len(er):
            continue

        m = min(len(p1_vec), len(er))
        p1_vec = p1_vec[:m]
        er = er[:m]

        p1 = float(p1_vec[chosen_idx])
        q_logged = float(er[chosen_idx])
        q_new = float(np.dot(p1_vec, er))

        dr = q_new + (p1 / p0) * (reward - q_logged)
        dr_values.append(dr)

    return float(np.mean(dr_values)) if dr_values else 0.0

for i in range(1, len(dates)):
    
    prev_log_path = str(Paths.TRAIN_DAILY_PATTERN).replace("*", dates[i-1])
    prev_policy_path = Paths.VW_POLICY / f"vw_policy_{dates[i-1]}.bin"
    cur_policy_path = Paths.VW_POLICY / f"vw_policy_{dates[i]}.bin"    
    
    yesterday_logs = pd.read_parquet(prev_log_path)
    yesterday_policy = VwAdfXGBPolicy(model_path=prev_policy_path)
    today_policy = VwAdfXGBPolicy(model_path=cur_policy_path)

    ips_ctr = ips_evaluate(
        yesterday_logs,
        today_policy,
    )
    
    snips_ctr = snips_evaluate(
        yesterday_logs,
        today_policy,
    )
    
    dr_ctr = doubly_robust2(
        yesterday_logs,
        today_policy,
    )
    
    print(dates[i], ips_ctr, snips_ctr, dr_ctr)
