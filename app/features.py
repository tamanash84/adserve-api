def feats_from_context(ctx: dict) -> str:
    toks = []
    for k, v in ctx.items():
        if isinstance(v, (int, float)):
            toks.append(f"{k}:{v:.6f}")
        else:
            toks.append(f"{k}={v}")
    return " ".join(toks)


def feats_from_item(r, include_rank=False) -> str:
    toks = [f"item={r.grocery_item}"]

    if hasattr(r, "category") and r.category:
        toks.append(f"cat={r.category}")

    if include_rank and hasattr(r, "rank_score"):
        toks.append(f"rank:{float(r.rank_score):.6f}")

    for name in ("base_price", "unit_price"):
        if hasattr(r, name):
            val = getattr(r, name)
            if val is not None:
                toks.append(f"{name}:{float(val):.6f}")

    return " ".join(toks)


def build_adf(shared_feats: str, items_df, include_rank=False):
    lines = [f"shared |c {shared_feats}"]
    for r in items_df.itertuples(index=False):
        lines.append(f"|a {feats_from_item(r, include_rank)}")
    return lines