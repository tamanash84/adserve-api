import re
import numpy as np
import pandas as pd

SIZE_PATTERNS = [
    # common patterns in product names
    (re.compile(r'(\d+(?:\.\d+)?)\s*(oz|ounce|ounces)\b', re.I), 1.0),
    (re.compile(r'(\d+(?:\.\d+)?)\s*(lb|pound|pounds)\b', re.I), 16.0),   # lb -> oz
    (re.compile(r'(\d+(?:\.\d+)?)\s*(g|gram|grams)\b', re.I), 0.035274),  # g -> oz
    (re.compile(r'(\d+(?:\.\d+)?)\s*(kg)\b', re.I), 35.274),             # kg -> oz
    (re.compile(r'(\d+(?:\.\d+)?)\s*(ml)\b', re.I), 0.033814),           # ml -> fl oz proxy
    (re.compile(r'(\d+(?:\.\d+)?)\s*(l|liter|litre)\b', re.I), 33.814),   # L -> fl oz proxy
    (re.compile(r'(\d+)\s*(ct|count)\b', re.I), None),                   # count proxy
    (re.compile(r'(\d+)\s*(pack)\b', re.I), None),                       # pack proxy
]

def infer_size_proxy(product_name: str) -> float:
    """
    Returns a rough 'size proxy' from product name.
    Uses ounces/fl-oz-ish for weights/volumes, and count/pack as a multiplier proxy.
    """
    if not isinstance(product_name, str) or not product_name:
        return 1.0

    name = product_name.lower()
    size = 1.0

    # weight/volume contribution: take the max detected quantity (avoid summing multiple matches)
    w_candidates = []
    for pat, factor in SIZE_PATTERNS:
        m = pat.search(name)
        if not m:
            continue
        qty = float(m.group(1))
        if factor is None:
            # count/pack-like: treat as multiplier-ish but damped
            w_candidates.append(np.log1p(qty))
        else:
            w_candidates.append(np.log1p(qty * factor))

    if w_candidates:
        # convert to a smooth multiplier around 1
        size *= float(np.exp(np.clip(max(w_candidates), 0, 5)) / np.exp(1.0))  # centered-ish

    # keywords hinting premium/organic/etc.
    premium_boost = 1.0
    if any(k in name for k in ["organic", "artisan", "imported", "grass-fed", "single origin"]):
        premium_boost *= 1.15
    if any(k in name for k in ["family size", "jumbo", "extra large"]):
        premium_boost *= 1.10
    if any(k in name for k in ["value", "budget"]):
        premium_boost *= 0.92

    return size * premium_boost


def generate_static_product_prices(
    products: pd.DataFrame,
    product_id_col="product_id",
    product_name_col="product_name",
    dept_col="department_id",
    aisle_col="aisle_id",
    currency_round=0.01,
    min_price=0.29,
    max_price=99.99,
    seed=42,
):
    """
    products: DataFrame with columns: product_id, product_name, department_id, aisle_id
    returns: DataFrame [product_id, base_price]
    """
    rng = np.random.default_rng(seed)

    df = products[[product_id_col, product_name_col, dept_col, aisle_col]].copy()

    # Global log-price center (tune to your market)
    mu = np.log(3.50)  # typical item around ~3.50

    # Department effects (coarse): draw once per dept
    depts = df[dept_col].dropna().unique()
    dept_effect = {d: rng.normal(0.0, 0.35) for d in depts}  # bigger spread across depts

    # Aisle effects (finer): smaller variance
    aisles = df[aisle_col].dropna().unique()
    aisle_effect = {a: rng.normal(0.0, 0.18) for a in aisles}

    # Size proxy per product name
    size_proxy = df[product_name_col].fillna("").map(infer_size_proxy).astype(float)

    # Product idiosyncratic noise
    eps = rng.normal(0.0, 0.55, size=len(df))  # heavy-ish tail

    logp = (
        mu
        + df[dept_col].map(dept_effect).fillna(0.0).values
        + df[aisle_col].map(aisle_effect).fillna(0.0).values
        + 0.30 * np.log(np.clip(size_proxy.values, 0.5, 10.0))
        + eps
    )

    price = np.exp(logp)

    # Clip and round
    price = np.clip(price, min_price, max_price)
    price = np.round(price / currency_round) * currency_round

    out = df[[product_id_col]].copy()
    out["base_price"] = price.astype(float)
    return out


# products.csv in Instacart: product_id, product_name, aisle_id, department_id
products = pd.read_csv("../data/Instacart/products.csv")
base_prices = generate_static_product_prices(products, seed=7)
base_prices.to_parquet("../data/parquets/product_base_prices.parquet", index=False)