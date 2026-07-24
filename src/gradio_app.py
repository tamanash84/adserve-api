import queue
import logging
import threading
import traceback
from pathlib import Path
from datetime import datetime, timedelta, date
from config import Paths

import gradio as gr
import pandas as pd
import duckdb
import ast
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from simulate import SimulationConfig, run_simulation, cleanup_wal_files
from bandit_policy import VwAdfXGBPolicy

simulation_lock = threading.Lock()

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def hour_choices():
    return [f"{h:02d}:00" for h in range(24)]


def hour_to_int(hhmm: str) -> int:
    if not hhmm:
        raise ValueError("Hour value is empty")
    return int(hhmm.split(":")[0])


def parse_date_yyyy_mm_dd(x: str):
    return datetime.strptime(x.strip(), "%Y-%m-%d").date()


def parse_store_ids(text: str):
    text = (text or "").strip()

    if not text:
        return None

    out = []

    for token in text.split(","):
        token = token.strip()
        if token:
            out.append(int(token))

    return out or None


def adjust_end_date(start_date, end_date):

    if start_date is None:
        return end_date

    start_dt = datetime.fromtimestamp(start_date)

    if end_date is None:
        return start_dt + timedelta(days=1)

    end_dt = datetime.fromtimestamp(end_date)

    if end_dt <= start_dt:
        return start_dt + timedelta(days=1)

    return end_dt


def normalize_gradio_date(x) -> date:
    """
    Accepts Gradio DateTime values as:
    - float/int Unix timestamp
    - datetime.datetime
    - datetime.date
    - string like '2026-01-10'
    - string like '2026-01-10 00:00:00'
    - string like '2026-01-10T00:00:00'

    Returns python date.
    """

    if x is None:
        raise ValueError("Date value is empty")

    # Gradio may return Unix timestamp float, e.g. 1767999600.0
    if isinstance(x, (int, float)):
        return datetime.fromtimestamp(x).date()

    # datetime.datetime
    if isinstance(x, datetime):
        return x.date()

    # datetime.date
    if isinstance(x, date):
        return x

    # string
    s = str(x).strip()

    if not s:
        raise ValueError("Date value is empty")

    # keep first 10 chars: YYYY-MM-DD
    s = s[:10]

    return datetime.strptime(s, "%Y-%m-%d").date()

def show_selected_file(file_obj):
    if file_obj is None:
        return ""
    return file_obj.name

def build_daily_files(root_dir, start_date, end_date, prefix="purchases"):
    """
    Build daily jsonl file list.

    end_date is exclusive.

    Example filenames:
        purchases_2026-01-10.jsonl
        impressions_2026-01-10.jsonl
    """

    files = []

    d = normalize_gradio_date(start_date)
    end_d = normalize_gradio_date(end_date)

    if end_d <= d:
        raise ValueError(
            f"end_date must be after start_date. Got start={d}, end={end_d}"
        )

    root = Path(root_dir)

    while d < end_d:
        p = root / f"{prefix}_{d.isoformat()}.jsonl"

        if p.exists():
            files.append(str(p))

        d += timedelta(days=1)

    return files

def to_date_obj(x):
    """
    Handles Gradio DateTime values.

    Gradio DateTime may return:
      - float timestamp, e.g. 1767999600.0
      - string, e.g. '2026-01-10'
      - datetime/date
      - None
    """
    if x is None:
        return None

    if isinstance(x, (int, float)):
        return datetime.fromtimestamp(x).date()

    if isinstance(x, datetime):
        return x.date()

    if isinstance(x, date):
        return x

    return datetime.strptime(str(x)[:10], "%Y-%m-%d").date()


def build_hourly_train_files(start_date=None, end_date=None):
    """
    Builds list of hourly train parquet files.

    Expected structure:
        data/bandit/training/date=2026-01-16/hour=09/train.parquet

    end_date is exclusive.
    """

    start_d = to_date_obj(start_date)
    end_d = to_date_obj(end_date)
    
    if start_d is None or end_d is None:
        return []

    if end_d <= start_d:
        return []

    files = []

    d = start_d
    while d < end_d:
        day_dir = Paths.BANDIT_TRAIN / f"date={d.isoformat()}"

        if day_dir.exists():
            for hour_dir in sorted(day_dir.glob("hour=*")):
                p = hour_dir / "train.parquet"

                if p.exists():
                    files.append(str(p.resolve()))

        d += timedelta(days=1)

    return files


def date_range_list(start_date, end_date) -> list:
    """
    Returns dates in [start_date, end_date), as YYYY-MM-DD strings.
    """
    start_d = normalize_gradio_date(start_date)
    end_d = normalize_gradio_date(end_date)

    if end_d <= start_d:
        return []

    dates = []
    d = start_d

    while d < end_d:
        dates.append(d.isoformat())
        d += timedelta(days=1)

    return dates


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


def ips_evaluate(logged_events: pd.DataFrame, new_policy) -> float:
    ips_sum = 0.0
    n = 0

    required = {"comment", "chosen_index", "reward", "propensity"}
    if not required.issubset(logged_events.columns):
        return 0.0

    for row in logged_events.itertuples(index=False):
        adf = safe_literal_eval(getattr(row, "comment", None), default=[])

        if not adf:
            continue

        chosen_idx = int(getattr(row, "chosen_index"))
        reward = float(getattr(row, "reward") or 0.0)
        p0 = float(getattr(row, "propensity") or 0.0)

        if p0 <= 0:
            continue

        p1_vec = new_policy.probs_given_adf(adf)

        if chosen_idx >= len(p1_vec):
            continue

        p1 = float(p1_vec[chosen_idx])
        w = p1 / p0

        ips_sum += w * reward
        n += 1

    return ips_sum / n if n else 0.0


def snips_evaluate(logged_events: pd.DataFrame, new_policy) -> float:
    weighted_rewards = 0.0
    total_weight = 0.0

    required = {"comment", "chosen_index", "reward", "propensity"}
    if not required.issubset(logged_events.columns):
        return 0.0

    for row in logged_events.itertuples(index=False):
        adf = safe_literal_eval(getattr(row, "comment", None), default=[])

        if not adf:
            continue

        chosen_idx = int(getattr(row, "chosen_index"))
        reward = float(getattr(row, "reward") or 0.0)
        p0 = float(getattr(row, "propensity") or 0.0)

        if p0 <= 0:
            continue

        p1_vec = new_policy.probs_given_adf(adf)

        if chosen_idx >= len(p1_vec):
            continue

        p1 = float(p1_vec[chosen_idx])
        w = p1 / p0

        weighted_rewards += w * reward
        total_weight += w

    return weighted_rewards / total_weight if total_weight else 0.0


def doubly_robust(logged_events: pd.DataFrame, new_policy) -> float:
    dr_values = []

    required = {"comment", "chosen_index", "reward", "propensity", "expected_rewards"}
    if not required.issubset(logged_events.columns):
        return 0.0

    for row in logged_events.itertuples(index=False):
        adf = safe_literal_eval(getattr(row, "comment", None), default=[])
        er = getattr(row, "expected_rewards", None)

        if not adf or len(er) == 0:
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

def zero_kpi_cards():
    return (
        fmt_metric("Impressions", "0"),
        fmt_metric("Matched Purchases", "0"),
        fmt_metric("Purchase Rate %", "0.00"),
        fmt_metric("Attributed Revenue", "€0.00"),
        fmt_metric("Total Purchases", "0"),
        fmt_metric("Total Revenue", "€0.00"),
        fmt_metric("Revenue / Impression", "0.0000"),
        fmt_metric("Stores", "0"),
    )

def refresh_kpis(start_date=None, end_date=None):
    """
    Computes KPI boxes from hourly finalized training parquet.

    KPI definitions:
      impressions          = distinct impression_id if available, else row count
      matched purchases    = rows where reward > 0
      purchase rate %      = 100 * matched purchases / impressions
      attributed revenue   = sum(amount) for reward > 0
      revenue / impression = attributed revenue / impressions
      avg order value      = attributed revenue / matched purchases
    """

    train_files = build_hourly_train_files(start_date, end_date)

    if not train_files:
        return zero_kpi_cards()

    con = duckdb.connect()

    try:
        # inspect available columns
        cols_df = con.execute(
            """
            DESCRIBE SELECT *
            FROM read_parquet(?)
            """,
            [train_files],
        ).df()

        cols = set(cols_df["column_name"].tolist())

        def has(col):
            return col in cols

        # column selection
        impression_expr = "COUNT(DISTINCT impression_id)" if has("impression_id") else "COUNT(*)"

        if has("reward"):
            matched_expr = "COALESCE(CAST(reward AS DOUBLE), 0) > 0"
        elif has("label"):
            matched_expr = "COALESCE(CAST(label AS DOUBLE), 0) > 0"
        elif has("matched"):
            matched_expr = "COALESCE(CAST(matched AS DOUBLE), 0) > 0"
        else:
            matched_expr = "FALSE"


        revenue_col = "base_price"


        revenue_expr = (
            f"""
            COALESCE(
                SUM(
                    CASE WHEN {matched_expr}
                    THEN CAST({revenue_col} AS DOUBLE)
                    ELSE 0 END
                ),
                0
            )
            """
            if revenue_col
            else "0.0"
        )

        unique_stores_expr = "COUNT(DISTINCT store_id)" if has("store_id") else "0"

        if has("pid_shown"):
            unique_products_expr = "COUNT(DISTINCT pid_shown)"
        elif has("product_id"):
            unique_products_expr = "COUNT(DISTINCT product_id)"
        else:
            unique_products_expr = "0"

        avg_propensity_expr = (
            "COALESCE(AVG(CAST(propensity AS DOUBLE)), 0)"
            if has("propensity")
            else "0.0"
        )

        avg_xgb_expr = (
            "COALESCE(AVG(CAST(xgb_prob_mean AS DOUBLE)), 0)"
            if has("xgb_prob_mean")
            else "0.0"
        )

        sql = f"""
        SELECT
            {impression_expr} AS impressions,

            SUM(
                CASE WHEN {matched_expr}
                THEN 1 ELSE 0 END
            ) AS matched_purchases,

            {revenue_expr} AS attributed_revenue,

            {unique_stores_expr} AS unique_stores,

            {unique_products_expr} AS unique_products,

            {avg_propensity_expr} AS avg_propensity,

            {avg_xgb_expr} AS avg_xgb_score

        FROM read_parquet(?)
        """

        row = con.execute(sql, [train_files]).fetchone()

        impressions = int(row[0] or 0)
        matched_purchases = int(row[1] or 0)
        attributed_revenue = float(row[2] or 0.0)
        unique_stores = int(row[3] or 0)
        unique_products = int(row[4] or 0)
        avg_propensity = float(row[5] or 0.0)
        avg_xgb_score = float(row[6] or 0.0)
        total_purchases = 0
        total_revenue = 0

        purchase_rate_pct = (
            100.0 * matched_purchases / impressions
            if impressions
            else 0.0
        )

        revenue_per_impression = (
            attributed_revenue / impressions
            if impressions
            else 0.0
        )

        return (
            fmt_metric("Impressions", f"{impressions:,}"),
            fmt_metric("Matched Purchases", f"{matched_purchases:,}"),
            fmt_metric("Purchase Rate %", f"{purchase_rate_pct:.2f}"),
            fmt_metric("Attributed Revenue", f"€{attributed_revenue:,.2f}"),
            fmt_metric("Total Purchases", f"{total_purchases:,}"),
            fmt_metric("Total Revenue", f"€{total_revenue:,.2f}"),
            fmt_metric("Revenue / Impression", f"{revenue_per_impression:.4f}"),
            fmt_metric("Stores", f"{unique_stores:,}"),
        )

    except Exception as e:
        print("[KPI ERROR]", repr(e))
        return zero_kpi_cards()

    finally:
        con.close()
        
# ------------------------------------------------------------
# Plotly
# ------------------------------------------------------------
def build_eval_plot(cum_df):
    if cum_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Cumulative CVR and Lift")
        return fig

    required_cols = [
        "date",
        "VW_ADF cvr",
        "Random cvr",
        "lift_pct",
    ]

    missing = [col for col in required_cols if col not in cum_df.columns]
    if missing:
        fig = go.Figure()
        fig.update_layout(title=f"Missing columns: {missing}")
        return fig

    plot_df = cum_df[required_cols].copy()

    plot_df["date"] = plot_df["date"].astype(str)

    for col in required_cols[1:]:
        plot_df[col] = pd.to_numeric(
            plot_df[col],
            errors="coerce",
        )

    plot_df = plot_df.dropna(
        subset=required_cols[1:],
        how="all",
    )

    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )

    # VW cumulative CVR
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["VW_ADF cvr"],
            mode="lines+markers",
            name="VW cumulative CVR",
            line=dict(
                color="blue",
                width=3,
            ),
            marker=dict(size=7),
        ),
        secondary_y=False,
    )

    # Random cumulative CVR
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["Random cvr"],
            mode="lines+markers",
            name="Random cumulative CVR",
            line=dict(
                color="orange",
                width=3,
            ),
            marker=dict(size=7),
        ),
        secondary_y=False,
    )

    # Cumulative lift on secondary axis
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["lift_pct"],
            mode="lines+markers",
            name="Lift %",
            line=dict(
                color="green",
                width=3,
                dash="dash",
            ),
            marker=dict(size=7),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Cumulative CVR and Lift: VW vs Random",
        height=450,
        margin=dict(l=60, r=70, t=60, b=60),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        title_text="Date",
    )

    fig.update_yaxes(
        title_text="Cumulative CVR",
        tickformat=".2%",
        secondary_y=False,
    )

    fig.update_yaxes(
        title_text="Lift %",
        ticksuffix="%",
        secondary_y=True,
    )

    return fig

def build_ope_plot(ope_metrics):
    if ope_metrics.empty:
        return px.line(title="OPE Metrics Over Time")

    ope_plot = ope_metrics.melt(
        id_vars=["eval_date"],
        value_vars=["ips", "snips", "dr"],
        var_name="metric",
        value_name="value",
    )

    ope_plot["eval_date"] = ope_plot["eval_date"].astype(str)
    ope_plot["value"] = pd.to_numeric(
        ope_plot["value"],
        errors="coerce",
    )

    ope_plot = ope_plot.dropna(subset=["value"])

    fig = px.line(
        ope_plot,
        x="eval_date",
        y="value",
        color="metric",
        markers=True,
        title="OPE Metrics Over Time",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Estimated Reward",
        height=420,
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )

    return fig

def build_action_plot(action_df: pd.DataFrame):

    fig = px.scatter(
        action_df,
        x="share_pct",
        y="cvr",
        size="impressions",
        hover_name="pid_shown",
        hover_data={
            "impressions": ":,.0f",
            "share_pct": ":.2f",
            "cvr": ":.2%",
        },
        color_discrete_sequence=["#2563EB"],  # blue
        template="plotly_white",
        size_max=40,
    )

    fig.update_traces(
        marker=dict(
            opacity=0.75,
            line=dict(
                color="#1E3A8A",
                width=1,
            ),
        )
    )

    fig.update_layout(
        title="Action Share vs Conversion Rate",
        xaxis_title="Impression Share (%)",
        yaxis_title="CVR",
        height=450,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig.update_xaxes(
        ticksuffix="%",
        showgrid=True,
    )

    fig.update_yaxes(
        tickformat=".1%",
        showgrid=True,
    )

    return fig

# ------------------------------------------------------------
# Evaluation run
# ------------------------------------------------------------
def run_bandit_evaluation(eval_start_date, eval_end_date):
    dates = date_range_list(eval_start_date, eval_end_date)

    if len(dates) < 2:
        empty_daily = pd.DataFrame(
            columns=[
                "date",
                "policy_name",
                "impressions",
                "purchases",
                "cvr",
            ]
        )

        empty_ope = pd.DataFrame(
            columns=[
                "eval_date",
                "log_date",
                "ips",
                "snips",
                "dr",
            ]
        )

        return (
            fmt_metric("VW CVR", "0.0000"),
            fmt_metric("Random CVR", "0.0000"),
            fmt_metric("Lift", "0.00%"),
            fmt_metric("IPS", "0.0000"),
            fmt_metric("SNIPS", "0.0000"),
            fmt_metric("DR", "0.0000"),
            empty_daily,
            empty_ope,
            pd.DataFrame(),
            "Select at least two dates because OPE evaluates today's policy on yesterday's logs.",
        )

    daily_rows = []
    ope_rows = []
    log_lines = []

    # --------------------------------------------------------
    # 1. Logged per-day metrics
    # --------------------------------------------------------
    bandit_policy = "VW_ADF"
    
    for dt in dates:
        train_path = Path(str(Paths.TRAIN_DAILY_PATTERN).replace("*", dt))

        if not train_path.exists():
            log_lines.append(f"[WARN] Missing train file: {train_path}")
            continue

        df = pd.read_parquet(train_path)

        required_cols = {"policy_name", "reward"}
        if not required_cols.issubset(df.columns):
            log_lines.append(
                f"[WARN] Skipping {dt}. Missing columns: {required_cols - set(df.columns)}"
            )
            continue

        df = df.copy()

        df["reward"] = pd.to_numeric(df["reward"], errors="coerce").fillna(0.0)

        daily = (
            df.groupby("policy_name", dropna=False)
            .agg(
                impressions=("reward", "size"),
                purchases=("reward", "sum"),
                cvr=("reward", "mean"),
            )
            .reset_index()
        )
        
        daily["date"] = str(dt)
        
        daily_rows.append(daily)

               
    daily_metrics = (
        pd.concat(daily_rows, ignore_index=True)
        if daily_rows
        else pd.DataFrame(
            columns=[
                "date",
                "policy_name",
                "impressions",
                "purchases",
                "cvr",
            ]
        )
    )
       
    action_cols = [
                    "pid_shown",
                    "impressions",
                    "purchases",
                    "cvr",
                    "share_pct",
                  ]
    
    action_metrics = (
        df.groupby("pid_shown", dropna=False)
        .agg(
            impressions=("reward", "size"),
            purchases=("reward", "sum"),
        )
        .reset_index()
    )
    
    action_metrics["cvr"] = (
        action_metrics["purchases"]
        / action_metrics["impressions"]
    )
    
    action_metrics["share_pct"] = (
        100
        * action_metrics["impressions"]
        / action_metrics["impressions"].sum()
    )
    
    action_metrics = action_metrics.reindex(columns=action_cols)      
    
    action_df = (
        action_metrics
        .nlargest(50, "impressions") # top-k K=50
    )
            
    lift_df = (
        daily_metrics.pivot(
            index="date",
            columns="policy_name",
            values=[
                "impressions",
                "purchases",
                "cvr",
            ],
        )
    )
    
    lift_df.columns = [
        f"{policy} {metric}"
        for metric, policy in lift_df.columns
    ]
    
    lift_df = lift_df.reset_index()
            
    # Safe lift calculation
    lift_df["lift_pct"] = np.where(
        lift_df["Random cvr"] > 0,
        (lift_df[f"{bandit_policy} cvr"] / lift_df["Random cvr"] - 1.0) * 100.0,
        np.nan,
    )
    
    # Optional: keep column order clean
    lift_cols = [
                    "date",
                    f"{bandit_policy} impressions",
                    f"{bandit_policy} purchases",
                    f"{bandit_policy} cvr",
                    "Random impressions",
                    "Random purchases",
                    "Random cvr",
                    "lift_pct",
                ]
    
    lift_df = lift_df.reindex(columns=lift_cols)      
           
    cum_df = lift_df.copy()
    for col in lift_cols:
        if any(word in col for word in ["impressions", "purchases"]):
            cum_df[col] = cum_df[col].cumsum()
            
    for policy in [bandit_policy, "Random"]:
        cum_df[f"{policy} cvr"] = np.where(
            cum_df[f"{policy} impressions"] > 0,
            cum_df[f"{policy} purchases"] / cum_df[f"{policy} impressions"],
            np.nan,
        )
                           
    cum_df["lift_pct"] = np.where(
        cum_df["Random cvr"] > 0,
        (cum_df[f"{bandit_policy} cvr"] / cum_df["Random cvr"] - 1.0) * 100.0,
        np.nan,
    )
    
    # --------------------------------------------------------
    # 2. OPE: evaluate policy_date on previous day's logs
    # --------------------------------------------------------
    for i in range(1, len(dates)):
        log_date = dates[i - 1]
        eval_date = dates[i]

        prev_log_path = Path(str(Paths.TRAIN_DAILY_PATTERN).replace("*", log_date))
        cur_policy_path = Paths.VW_POLICY / f"vw_policy_{eval_date}.bin"

        if not prev_log_path.exists():
            log_lines.append(f"[WARN] Missing previous log file: {prev_log_path}")
            continue

        if not cur_policy_path.exists():
            log_lines.append(f"[WARN] Missing policy file: {cur_policy_path}")
            continue

        try:
            yesterday_logs = pd.read_parquet(prev_log_path)
            today_policy = VwAdfXGBPolicy(model_path=cur_policy_path)

            ips = ips_evaluate(yesterday_logs, today_policy)
            snips = snips_evaluate(yesterday_logs, today_policy)
            dr = doubly_robust(yesterday_logs, today_policy)

            ope_rows.append(
                {
                    "eval_date": eval_date,
                    "log_date": log_date,
                    "ips": ips,
                    "snips": snips,
                    "dr": dr,
                    "log_rows": len(yesterday_logs),
                }
            )

        except Exception as e:
            log_lines.append(f"[ERROR] OPE failed for eval_date={eval_date}: {repr(e)}")

    if ope_rows:
        ope_metrics = pd.DataFrame(ope_rows)
    else:
        ope_metrics = pd.DataFrame(
            columns=[
                "eval_date",
                "log_date",
                "ips",
                "snips",
                "dr",
                "log_rows",
            ]
        )

    # --------------------------------------------------------
    # 3. Aggregate summary
    # --------------------------------------------------------
    if not daily_metrics.empty:
        agg_policy = (
            daily_metrics.groupby("policy_name", dropna=False)
            .agg(
                impressions=("impressions", "sum"),
                purchases=("purchases", "sum"),
            )
            .reset_index()
        )

        agg_policy["cvr"] = np.where(
            agg_policy["impressions"] > 0,
            agg_policy["purchases"] / agg_policy["impressions"],
            0.0,
        )
    else:
        agg_policy = pd.DataFrame(
            columns=[
                "policy_name",
                "impressions",
                "purchases",
                "cvr",
            ]
        )

    vw_cvr = 0.0
    random_cvr = 0.0
    lift = 0.0

    if not agg_policy.empty:
        vw_row = agg_policy.loc[agg_policy["policy_name"] == "VW_ADF"]
        rnd_row = agg_policy.loc[agg_policy["policy_name"] == "Random"]

        if not vw_row.empty:
            vw_cvr = float(vw_row["cvr"].iloc[0])

        if not rnd_row.empty:
            random_cvr = float(rnd_row["cvr"].iloc[0])

        if random_cvr > 0:
            lift = vw_cvr / random_cvr - 1.0

    if not ope_metrics.empty:
        agg_ips = float(ope_metrics["ips"].mean())
        agg_snips = float(ope_metrics["snips"].mean())
        agg_dr = float(ope_metrics["dr"].mean())
    else:
        agg_ips = 0.0
        agg_snips = 0.0
        agg_dr = 0.0

    # --------------------------------------------------------
    # 4. Plot data (cumulated lift)
    # --------------------------------------------------------
    plot_rows = []

    if not cum_df.empty and "lift_pct" in cum_df.columns:
        tmp = cum_df[["date", "lift_pct"]].copy()        
        tmp["date"] = tmp["date"].astype(str)        
        tmp["metric"] = "VW lift vs Random"        
        tmp = tmp.rename(columns={
            "lift_pct": "value"
        })        
        plot_rows.append(
            tmp[["date", "metric", "value"]]
        )

    if not ope_metrics.empty:
        ope_plot = ope_metrics.melt(
            id_vars=["eval_date"],
            value_vars=["ips", "snips", "dr"],
            var_name="metric",
            value_name="value",
        )
        
        ope_plot["date"] = ope_plot["eval_date"].astype(str)
        
        plot_rows.append(
            ope_plot[["date", "metric", "value"]]
        )

    if plot_rows:
        plot_df = pd.concat(plot_rows, ignore_index=True)
    else:
        plot_df = pd.DataFrame(columns=["date", "metric", "value"])

    # --------------------------------------------------------
    # 5. Formatting tables
    # --------------------------------------------------------
    daily_metrics = daily_metrics.copy()
    if not daily_metrics.empty:
        daily_metrics["cvr"] = daily_metrics["cvr"].map(lambda x: f"{x:.4f}")
        daily_metrics["purchases"] = daily_metrics["purchases"].astype(int)
        daily_metrics["impressions"] = daily_metrics["impressions"].astype(int)
        
    lift_df = lift_df.copy()
    if not lift_df.empty:
        for col in lift_cols:
            if "cvr" in col or "lift" in col:
                lift_df[col] = lift_df[col].map(lambda x: f"{x:.4f}")
    
    cum_df = cum_df.copy()
    if not cum_df.empty:
        for col in lift_cols:
            if "cvr" in col or "lift" in col:
                cum_df[col] = cum_df[col].map(lambda x: f"{x:.4f}")
                
    action_df = action_df.copy()
    if not action_df.empty:
        for col in action_cols:
            if "cvr" in col or "share_pct" in col:
                action_df[col] = action_df[col].map(lambda x: f"{x:.4f}")

    ope_metrics = ope_metrics.copy()
    if not ope_metrics.empty:
        for col in ["ips", "snips", "dr"]:
            ope_metrics[col] = ope_metrics[col].map(lambda x: f"{x:.4f}")

    agg_policy = agg_policy.copy()
    if not agg_policy.empty:
        agg_policy["cvr"] = agg_policy["cvr"].map(lambda x: f"{x:.4f}")
        agg_policy["purchases"] = agg_policy["purchases"].astype(int)
        agg_policy["impressions"] = agg_policy["impressions"].astype(int)

    log_text = "\n".join(log_lines) if log_lines else "Evaluation completed."
    
    lift_fig = build_eval_plot(cum_df)
    ope_fig = build_ope_plot(ope_metrics)
    action_fig = build_action_plot(action_metrics)

    return (
        fmt_metric("VW CVR", f"{vw_cvr:.4f}"),
        fmt_metric("Random CVR", f"{random_cvr:.4f}"),
        fmt_metric("Lift", f"{lift:.2%}"),
        fmt_metric("IPS", f"{agg_ips:.4f}"),
        fmt_metric("SNIPS", f"{agg_snips:.4f}"),
        fmt_metric("DR", f"{agg_dr:.4f}"),
        lift_df,
        cum_df,
        action_df,
        ope_metrics,
        lift_fig,
        ope_fig,
        action_fig,
        log_text,
    )


def list_recent_outputs():
    folders = [
        "../data/bandit/wal",
        "../data/bandit/training",
        "../data/pos/wal",
        "../data/pos/purchases",
        "../model/xgboost",
        "../model/vw",
    ]

    rows = []

    for folder in folders:
        p = Path(folder)

        if not p.exists():
            continue

        for f in p.rglob("*"):
            if not f.is_file():
                continue

            rows.append(
                {
                    "path": str(f),
                    "size_kb": round(f.stat().st_size / 1024, 2),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["path", "size_kb", "modified"])

    return (
        pd.DataFrame(rows)
        .sort_values("modified", ascending=False)
        .head(100)
    )


# ------------------------------------------------------------
# Logging to Gradio
# ------------------------------------------------------------

class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def setup_queue_logger(log_queue: queue.Queue):
    logger = logging.getLogger("bandit-sim")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = QueueLogHandler(log_queue)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# ------------------------------------------------------------
# Main UI callback
# ------------------------------------------------------------

def run_from_ui(
    pos_file,
    sim_start_date,
    sim_end_date,
    active_start_time,
    active_end_time,
    nightly_train_time,
    train_lookback_days,
    store_ids_text,
    impression_every_sec,
    reward_match_window,
    batch_size,
    skip_rows,
    max_pos_rows,
    cleanup_before_run,
    archive_cleanup,
    dry_run_cleanup,
):
    if not simulation_lock.acquire(blocking=False):
        yield (
            "A simulation is already running. Please wait until it finishes.",
            *refresh_kpis(sim_start_date, sim_end_date),
            list_recent_outputs(),
        )
        return

    log_queue = queue.Queue()
    logger = setup_queue_logger(log_queue)
    log_lines = []
    
    print("pos_file =", pos_file)
    print("type(pos_file) =", type(pos_file))

    def push_log(msg):
        log_lines.append(msg)
        return "\n".join(log_lines[-500:])
    
    try:
        if pos_file is None:
            raise ValueError("Please select a POS parquet file.")

        pos_path = pos_file.name

        if not Path(pos_path).exists():
            raise ValueError(f"Selected POS parquet file does not exist: {pos_path}")

        start_d = normalize_gradio_date(sim_start_date)
        end_d = normalize_gradio_date(sim_end_date)
        
        if end_d <= start_d:
            raise ValueError(
                f"End date must be after start date. start={start_d} end={end_d}"
            )
        
        sim_start_date_str = start_d.isoformat()
        sim_end_date_str = end_d.isoformat()
        
        active_start_hour = hour_to_int(active_start_time)
        active_end_hour = hour_to_int(active_end_time)
        nightly_train_hour = hour_to_int(nightly_train_time)

        if active_end_hour <= active_start_hour:
            raise ValueError(
                f"Active end time must be after active start time. "
                f"Got start={active_start_time}, end={active_end_time}"
            )

        store_ids = parse_store_ids(store_ids_text)

        max_pos_rows_value = None
        if max_pos_rows is not None and int(max_pos_rows) > 0:
            max_pos_rows_value = int(max_pos_rows)

        cfg = SimulationConfig(
            pos_path=Path(pos_path),
            sim_start_date=sim_start_date_str,
            sim_end_date=sim_end_date_str,
            active_start_hour=active_start_hour,
            active_end_hour=active_end_hour,
            nightly_train_hour=nightly_train_hour,
            train_lookback_days=int(train_lookback_days),
            store_ids=store_ids,
            impression_every_sec=int(impression_every_sec),
            reward_match_window=int(reward_match_window),
            batch_size=int(batch_size),
            skip_rows=int(skip_rows),
            max_pos_rows=max_pos_rows_value,
            cleanup_before_run=bool(cleanup_before_run),
            archive_cleanup=bool(archive_cleanup),
            dry_run_cleanup=bool(dry_run_cleanup),
        )

    except Exception as e:
        yield (
            f"Input validation failed:\n{e}",
            *zero_kpi_cards(),
            list_recent_outputs(),
        )
        return

    def worker():
        try:
            logger.info("Starting simulation from Gradio UI")
            logger.info(f"POS file: {cfg.pos_path}")
            logger.info(f"Simulation dates: {cfg.sim_start_date} to {cfg.sim_end_date}")
            logger.info(
                f"Active window: {active_start_time} to {active_end_time}; "
                f"nightly training: {nightly_train_time}"
            )

            if cfg.cleanup_before_run:
                logger.info("Cleaning old WAL/model files")

                cleanup_wal_files(
                    dirs=[
                        "../data/bandit/wal",
                        "../data/bandit/training",
                        "../data/pos/wal",
                        "../data/pos/purchases",
                        "../model/xgboost",
                        "../model/vw",
                    ],
                    archive=cfg.archive_cleanup,
                    dry_run=cfg.dry_run_cleanup,
                    log=logger,
                )

            run_simulation(cfg, logger)

        except Exception:
            logger.error("Simulation failed")
            logger.error(traceback.format_exc())

        finally:
            simulation_lock.release()
            log_queue.put("__DONE__")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        try:
            msg = log_queue.get(timeout=5)

            if msg == "__DONE__":
                yield (
                    push_log("Run finished."),
                    *refresh_kpis(sim_start_date, sim_end_date),
                    list_recent_outputs(),
                )
                break

            log_text = push_log(msg)

        except queue.Empty:
            log_text = "\n".join(log_lines[-500:])

        yield (
            log_text,
            *refresh_kpis(sim_start_date, sim_end_date),
            list_recent_outputs(),
        )


# ------------------------------------------------------------
# Gradio front panel
# ------------------------------------------------------------
def fmt_metric(title, value):
    return f"""
    <div style="
        background:white;
        border:1px solid #ddd;
        border-radius:12px;
        padding:15px;
        text-align:center;
        min-height:90px;
    ">
        <div style="
            font-size:14px;
            font-weight:bold;
            color:#666;
        ">
            {title}
        </div>

        <div style="
            font-size:32px;
            font-weight:800;
            margin-top:8px;
            color:#2563eb;
        ">
            {value}
        </div>
    </div>
    """

HOURS = hour_choices()

with gr.Blocks(title="Bandit Simulation UI") as demo:

    gr.Markdown("""
    # Ad serving framework UI
    """)


    with gr.Tabs():

        # ==================================================
        # TAB 1 : SIMULATION
        # ==================================================
        with gr.Tab("Simulation"):

            # ==========================================
            # INPUTS
            # ==========================================
        
            with gr.Accordion("Simulation Settings", open=True):
        
                with gr.Row():
        
                    pos_file = gr.File(
                        label="POS Parquet",
                        file_types=[".parquet"],
                    )
        
                    selected_pos_path = gr.Textbox(
                        label="Selected File",
                        interactive=False,
                    )
                    
                    pos_file.change(
                        lambda f: f.name if f else "",
                        inputs=pos_file,
                        outputs=selected_pos_path,
                    )
        
                with gr.Row():
        
                    sim_start_date = gr.DateTime(
                        label="Start Date",
                        value=datetime(2026, 1, 10),
                        include_time=False,
                    )
        
                    sim_end_date = gr.DateTime(
                        label="End Date",
                        value=datetime(2026, 1, 11),
                        include_time=False,
                    )
                
                sim_start_date.change(
                    fn=adjust_end_date,
                    inputs=[sim_start_date, sim_end_date],
                    outputs=sim_end_date,
                )
                
                sim_end_date.change(
                    fn=adjust_end_date,
                    inputs=[sim_start_date, sim_end_date],
                    outputs=sim_end_date,
                )
        
                with gr.Row():
        
                    active_start_time = gr.Dropdown(
                        HOURS,
                        value="08:00",
                        label="Active Start",
                    )
        
                    active_end_time = gr.Dropdown(
                        HOURS,
                        value="14:00",
                        label="Active End",
                    )
        
                    nightly_train_time = gr.Dropdown(
                        HOURS,
                        value="01:00",
                        label="Nightly Train",
                    )
        
                with gr.Row():
        
                    train_lookback_days = gr.Number(
                        label="Lookback Days",
                        value=7,
                    )
        
                    impression_every_sec = gr.Number(
                        label="Impression Interval",
                        value=300,
                    )
        
                    reward_match_window = gr.Number(
                        label="Reward Window",
                        value=3600,
                    )
                    
                with gr.Row():
                    batch_size = gr.Number(
                        label="Parquet batch size",
                        value=4096,
                        precision=0,
                    )
        
                    skip_rows = gr.Number(
                        label="Skip POS rows",
                        value=0,
                        precision=0,
                    )
        
                    max_pos_rows = gr.Number(
                        label="Max POS rows. Use 0 for no cap.",
                        value=0,
                        precision=0,
                    )
                    
                store_ids_text = gr.Textbox(
                    label="Store IDs comma separated. Empty means infer from POS.",
                    value="1,2",
                )
        
                with gr.Row():
                    cleanup_before_run = gr.Checkbox(
                        label="Cleanup before run",
                        value=True,
                    )
        
                    archive_cleanup = gr.Checkbox(
                        label="Archive instead of delete",
                        value=False,
                    )
        
                    dry_run_cleanup = gr.Checkbox(
                        label="Dry-run cleanup",
                        value=False,
                    )
        
                run_button = gr.Button(
                    "Run Simulation",
                    variant="primary",
                )
        
            # ==========================================
            # KPI SECTION
            # ==========================================
        
            gr.Markdown("## Live Metrics")
        
            with gr.Row(equal_height=True):
        
                impressions_md = gr.HTML(
                    value=fmt_metric("Impressions", "0")
                )
        
                matched_purchases_md = gr.HTML(
                    value=fmt_metric("Matched Purchases", "0")
                )
        
                purchase_rate_md = gr.HTML(
                    value=fmt_metric("Purchase Rate %", "0.00")
                )
        
                revenue_md = gr.HTML(
                    value=fmt_metric("Attributed Revenue", "0.00")
                )
        
            with gr.Row(equal_height=True):
        
                total_purchases_md = gr.HTML(
                    value=fmt_metric("Total Purchases", "0")
                )
        
                total_revenue_md = gr.HTML(
                    value=fmt_metric("Total Revenue", "0.00")
                )
        
                rpi_md = gr.HTML(
                    value=fmt_metric("Revenue / Impression", "0.0000")
                )
        
                stores_md = gr.HTML(
                    value=fmt_metric("Stores", "0")
                )
                
            refresh_kpis_button = gr.Button(
                "Refresh KPIs"
            )
        
            # ==========================================
            # LOGS + FILES
            # ==========================================
        
            with gr.Row():        
       
                logs = gr.Textbox(
                    label="Live Logs",
                    lines=20,
                    autoscroll=True,
                )
        
            with gr.Row():
    
                output_files = gr.Dataframe(
                    label="Generated Files",
                    value=list_recent_outputs(),
                    interactive=False,
                )
    
            refresh_files_button = gr.Button(
                "Refresh Files"
            )
        
            run_button.click(
                fn=run_from_ui,
                inputs=[
                    pos_file,
                    sim_start_date,
                    sim_end_date,
                    active_start_time,
                    active_end_time,
                    nightly_train_time,
                    train_lookback_days,
                    store_ids_text,
                    impression_every_sec,
                    reward_match_window,
                    batch_size,
                    skip_rows,
                    max_pos_rows,
                    cleanup_before_run,
                    archive_cleanup,
                    dry_run_cleanup,
                ],
                outputs=[
                    logs,
                    impressions_md,
                    matched_purchases_md,
                    purchase_rate_md,
                    revenue_md,
                    total_purchases_md,
                    total_revenue_md,
                    rpi_md,
                    stores_md,
                    output_files,
                ],
                concurrency_limit=1,
            )
        
        
            refresh_kpis_button.click(
                fn=refresh_kpis,
                inputs=[
                    sim_start_date,
                    sim_end_date,
                ],
                outputs=[
                    impressions_md,
                    matched_purchases_md,
                    purchase_rate_md,
                    revenue_md,
                    total_purchases_md,
                    total_revenue_md,
                    rpi_md,
                    stores_md,
                ]
            )
        
            refresh_files_button.click(
                fn=list_recent_outputs,
                inputs=[],
                outputs=output_files,
            )
    
        # ==================================================
        # TAB 2 : BANDIT EVALUATION
        # ==================================================
        with gr.Tab("Bandit Evaluation"):

            gr.Markdown("## Offline Bandit Evaluation")

            with gr.Row():
                eval_start_date = gr.DateTime(
                    label="Evaluation Start Date",
                    value=datetime(2026, 1, 10),
                    include_time=False,
                )

                eval_end_date = gr.DateTime(
                    label="Evaluation End Date",
                    value=datetime(2026, 1, 16),
                    include_time=False,
                )

            evaluate_button = gr.Button(
                "Run Bandit Evaluation",
                variant="primary",
            )

            gr.Markdown("### Aggregate Metrics")

            with gr.Row(equal_height=True):
                eval_vw_cvr_md = gr.HTML(
                    value=fmt_metric("VW CVR", "0.0000")
                )

                eval_random_cvr_md = gr.HTML(
                    value=fmt_metric("Random CVR", "0.0000")
                )

                eval_lift_md = gr.HTML(
                    value=fmt_metric("Lift", "0.00%")
                )

            with gr.Row(equal_height=True):
                eval_ips_md = gr.HTML(
                    value=fmt_metric("IPS", "0.0000")
                )

                eval_snips_md = gr.HTML(
                    value=fmt_metric("SNIPS", "0.0000")
                )

                eval_dr_md = gr.HTML(
                    value=fmt_metric("DR", "0.0000")
                )

            gr.Markdown("### Evaluation Plot")

            # eval_plot = gr.LinePlot(
            #     label="Daily cumulated lift over time in percentage",
            #     x="date",
            #     y="value",
            #     color="metric",
            #     tooltip=["date", "metric", "value"],
            #     height=500,
            # )
            
            # action_plot = gr.ScatterPlot(
            #     label="Action Share vs CVR",
            #     x="share_pct",
            #     y="cvr",
            #     tooltip=[
            #         "pid_shown",
            #         "impressions",
            #         "share_pct",
            #         "cvr",
            #     ],
            #     height=450,
            # )
            
            eval_plot = gr.Plot(label="Daily Cumulated Lift")
            ope_plot = gr.Plot(label="Daily offline evaluation metrics")
            action_plot = gr.Plot(label="Action Distribution")           
                      
            gr.Markdown("### Daily Logged Metrics")

            daily_eval_table = gr.Dataframe(
                label="Daily Logged Bandit Evaluation Metrics by Policy",
                interactive=False,
                wrap=True,
            )
            
            gr.Markdown("### Daily Cumulated Logged Metrics")

            cum_eval_table = gr.Dataframe(
                label="Daily Cumulated Logged Bandit Evaluation Metrics by Policy",
                interactive=False,
                wrap=True,
            )

            gr.Markdown("### Offline Policy Evaluation")

            ope_eval_table = gr.Dataframe(
                label="OPE Metrics: Today's Policy on Yesterday's Logs",
                interactive=False,
                wrap=True,
            )
            
            gr.Markdown("### Daily action distribution")

            act_eval_table = gr.Dataframe(
                label="Product ids shown and performance",
                interactive=False,
                wrap=True,
            )

            eval_logs = gr.Textbox(
                label="Evaluation Logs",
                lines=10,
                autoscroll=True,
            )

            evaluate_button.click(
                fn=run_bandit_evaluation,
                inputs=[
                    eval_start_date,
                    eval_end_date,
                ],
                outputs=[
                    eval_vw_cvr_md,
                    eval_random_cvr_md,
                    eval_lift_md,
                    eval_ips_md,
                    eval_snips_md,
                    eval_dr_md,
                    daily_eval_table,
                    cum_eval_table,
                    act_eval_table,
                    ope_eval_table,
                    eval_plot,
                    ope_plot,
                    action_plot,
                    eval_logs,
                ],
            )


if __name__ == "__main__":
    demo.queue()

    demo.launch(
        server_name="127.0.0.1",
        server_port=None,
        show_error=True,
    )