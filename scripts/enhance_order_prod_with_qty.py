import pandas as pd
import numpy as np

def add_quantity_to_order_products(
    in_path: str,
    out_path: str,
    lam: float = 0.5,          # tune: 0.3..0.7 typical
    max_q: int = 10,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    op = pd.read_csv(in_path)

    # q = 1 + Poisson(lam), clipped to [1, max_q]
    q = 1 + rng.poisson(lam=lam, size=len(op))
    op["quantity"] = np.clip(q, 1, max_q).astype("int8")

    # quick sanity
    print(op["quantity"].value_counts(normalize=True).sort_index().head(10))
    op.to_parquet(out_path, index=False)
    return op

# Example
add_quantity_to_order_products(
    in_path="../data/Instacart/order_products__prior.csv",
    out_path="../data/parquets/order_products__prior_qty.parquet",
    lam=0.55,
    seed=7
)

