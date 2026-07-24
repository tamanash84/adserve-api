import pandas as pd
import numpy as np
import math
from node2vec import Node2Vec
import networkx as nx
from pathlib import Path
from scipy.sparse import coo_matrix
from matplotlib.colors import ListedColormap, to_hex
import matplotlib.pyplot as plt
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "../data/Instacart"

products_csv = DATA_DIR / "products.csv"
aisles_csv = DATA_DIR / "aisles.csv"
departments_csv = DATA_DIR / "departments.csv"
orders_csv = DATA_DIR / "orders.csv"
order_products_prior_csv = DATA_DIR / "order_products__prior.csv"

# ETL pipeline to read large CSV's fast
orders_prior_csv = DATA_DIR / "orders_prior.csv"
reader = pd.read_csv(
    orders_csv,
    chunksize=500_000
)

with open(orders_prior_csv, "w") as f_out:
    first = True
    for chunk in reader:
        train = chunk.query("eval_set == 'prior'") #vectorized
        if not train.empty:
            train.to_csv(f_out, index=False, header=first)
            first = False

# products = pd.read_csv(products_csv, dtype={"product_id":"int64",
#                                               "product_name":"string[pyarrow]",
#                                               "aisle_id":"int32",
#                                               "department_id":"int32"})
# aisle_lookup = pd.read_csv(aisles_csv, dtype={"aisle_id":"int32",
#                                               "aisle":"string[pyarrow]"}).set_index("aisle_id")["aisle"].to_dict()
# dept_lookup = pd.read_csv(departments_csv, dtype={"department_id":"int32",
#                                                   "department":"string[pyarrow]"}).set_index("department_id")["department"].to_dict()
# products["aisle"] = products["aisle_id"].map(aisle_lookup)
# products["department"] = products["department_id"].map(dept_lookup)

products = pd.read_csv(products_csv, dtype={"product_id":"int64",
                                              "aisle_id":"int32",
                                              "department_id":"int32"})

# orders/baskets containing products
priors = pd.read_csv(order_products_prior_csv, usecols=(["order_id","product_id"])).sort_values("order_id")

# Map to contiguous indices
order_codes, order_index = pd.factorize(priors['order_id'], sort=False)
prod_codes,  prod_index  = pd.factorize(priors['product_id'], sort=False)

# Optional: cap large baskets before matrix build
# Compute basket sizes and filter orders > MAX_BASKET
sizes = pd.Series(order_codes).groupby(order_codes).size()
mask_orders = sizes[sizes.between(2, 100)].index
keep = np.isin(order_codes, mask_orders)
order_codes = order_codes[keep]
prod_codes  = prod_codes[keep]

n_orders = order_codes.max() + 1
n_prods  = prod_codes.max() + 1

# Build sparse incidence matrix X (orders x products), binary
data = np.ones_like(order_codes, dtype=np.uint8)
X = coo_matrix((data, (order_codes, prod_codes)), shape=(n_orders, n_prods)).tocsr()

# Item supports: diagonal of XtX or simply row sums of X
item_support = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)

# Pair co-occurrence: XtX
XtX = X.T @ X  # (products x products) symmetric

# Ensure CSR for fast row slicing
XtX_csr = XtX.tocsr()   # products x products
item_support = item_support.astype(np.int64)  # c(i)
N = int(n_orders)

# Map dense indices -> original product_id
# 'prod_index' came from factorize(product_id); Series(index_code -> original id)
prod_ids = pd.Series(prod_index)

K = 50
SCORE = 'jaccard'  # or 'lift'
MIN_COOC = 3            # co-occurrence threshold
MIN_JACCARD = 0.02       # jaccard threshold

rows = []
# Iterate rows; for each product i, find top-K j != i by chosen score
for i in range(XtX_csr.shape[0]):
    start, end = XtX_csr.indptr[i], XtX_csr.indptr[i+1]
    js = XtX_csr.indices[start:end]
    coocs = XtX_csr.data[start:end].astype(np.int64)

    # Remove self-pair if present (diagonal; cooc == item_support[i])
    mask = (js != i)
    js = js[mask]
    coocs = coocs[mask]
    if js.size == 0:
        continue

    ci = item_support[i]
    cj = item_support[js]

    if SCORE == 'jaccard':
        denom = (ci + cj - coocs)
        # avoid divide-by-zero
        jacc = np.divide(coocs, denom, out=np.zeros_like(coocs, dtype=float), where=(denom > 0))
        score = jacc
    else:  # Lift
        denom = (ci * cj)
        lift = np.divide(coocs * N, denom, out=np.zeros_like(coocs, dtype=float), where=(denom > 0))
        score = lift

    # Keep only meaningful pairs if you wish (e.g., cooc >= 3), to mimic your earlier min support
    valid = (coocs >= MIN_COOC) & (jacc >= MIN_JACCARD)
    if not np.any(valid):
        continue
    js = js[valid]
    coocs_valid = coocs[valid]
    score_valid = score[valid]
    cj_valid = item_support[js]

    # Select top-K via argpartition (no full sort)
    k = min(K, score_valid.size)
    idx = np.argpartition(-score_valid, k-1)[:k]
    # Order those K by (score DESC, cooc DESC, neighbor id ASC) for determinism
    # Build a small structured array for tie-breaking
    sub = np.stack([score_valid[idx], coocs_valid[idx], js[idx]], axis=1)
    order = np.lexsort((sub[:,2], -sub[:,1], -sub[:,0]))  # score desc, cooc desc, j asc
    idx_sorted = idx[order]

    # Build rows
    a_id = prod_ids.iloc[i]
    b_ids = prod_ids.iloc[js[idx_sorted]].to_numpy()
    scores_sorted = score_valid[idx_sorted]
    coocs_sorted = coocs_valid[idx_sorted]

    if SCORE == 'jaccard':
        # compute lift too if you want both in output
        lift_sorted = (coocs_sorted * N) / (ci * item_support[js[idx_sorted]])
        jacc_sorted = scores_sorted
    else:
        jacc_sorted = coocs_sorted / (ci + item_support[js[idx_sorted]] - coocs_sorted)
        lift_sorted = scores_sorted

    rows.append(pd.DataFrame({
        'a': a_id,
        'b': b_ids,
        'cooc': coocs_sorted,
        'jaccard': jacc_sorted,
        'lift': lift_sorted
    }))

topk = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['a','b','cooc','jaccard','lift'])
topk["score"] = topk["jaccard"] + np.log(topk["lift"])/20.

# Create canonical pairs (smallest first)
canon = topk.copy()
swap = canon['a'] > canon['b']
canon.loc[swap, ['a','b']] = canon.loc[swap, ['b','a']].to_numpy()

dups_sym = canon.duplicated(subset=['a','b'], keep=False)

G = nx.Graph()

for row in topk.itertuples(index=False):
    G.add_edge(row.a, row.b, weight=row.score)

def quick_check(G, sample_nodes=5):
    print(f"|V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}")
    if G.number_of_edges() == 0:
        print("[FATAL] Graph has no edges.")
        return
    # basic degree snapshot
    deg = dict(G.degree())
    iso = [n for n, d in deg.items() if d == 0]
    print(f"Isolates: {len(iso)}")
    if len(iso) > 0:
        print(f"Sample isolates: {iso[:min(5,len(iso))]}")
    # weight sanity
    miss = nan = nonpos = 0
    vals = []
    for _, _, d in G.edges(data=True):
        if 'weight' not in d:
            miss += 1
            continue
        w = d['weight']
        if w is None or (isinstance(w, float) and math.isnan(w)):
            nan += 1
            continue
        if w <= 0:
            nonpos += 1
        vals.append(float(w))
    print(f"Edges missing weight: {miss}, NaN weight: {nan}, non-positive weight: {nonpos}")
    if vals:
        a = np.array(vals, dtype=float)
        print(f"Weight stats: min={a.min():.6f}, p50={np.median(a):.6f}, p95={np.percentile(a,95):.6f}, max={a.max():.6f}")

quick_check(G)

def show_neighbors(G, u, k=10, weight_key="weight"):
    if u not in G:
        print(f"{u} not in graph")
        return
    nbrs = [(v, G[u][v].get(weight_key, 0.0)) for v in G.neighbors(u)]
    nbrs.sort(key=lambda t: (-t[1], t[0]))
    print(f"Neighbors of {u} (top {k})")
    for v, w in nbrs[:k]:
        print(f"  {v}: {w:.6f}")

# Example:
#show_neighbors(G, 30035, k=10)

def k12_neighbors_for_node(
    G: nx.Graph, u, K: int = 100,
    weight_key: str = "weight",
    min_edge_weight: float = 0.0,
    debug: bool = False
):
    if u not in G:
        if debug: print(f"[DBG] Node {u} not in G.")
        return None

    Nu = G[u]
    if len(Nu) == 0:
        if debug: print(f"[DBG] Node {u} has degree 0 (isolate).")
        return None

    scores = defaultdict(float)
    kept_1hop = kept_2hop = 0

    for x, dx in Nu.items():
        w_ux = dx.get(weight_key, None)
        if w_ux is None or not (w_ux > min_edge_weight) or (isinstance(w_ux, float) and math.isnan(w_ux)):
            continue
        # 1-hop
        scores[x] += w_ux
        kept_1hop += 1

        # 2-hop via x
        for v, dv in G[x].items():
            if v == u:
                continue
            w_xv = dv.get(weight_key, None)
            if w_xv is None or not (w_xv > min_edge_weight) or (isinstance(w_xv, float) and math.isnan(w_xv)):
                continue
            scores[v] += w_ux * w_xv
            kept_2hop += 1

    scores.pop(u, None)

    if not scores:
        if debug:
            print(f"[DBG] Node {u}: no valid neighbors after filters. "
                  f"1hop_kept={kept_1hop}, 2hop_contrib={kept_2hop}, min_edge_weight={min_edge_weight}")
        return None

    items = sorted(scores.items(), key=lambda t: (-t[1], t[0]))[:K]
    return items

prd_khop_nbr = []
for u in list(G.nodes()):
    res = k12_neighbors_for_node(G, u, K=100, min_edge_weight=0.0, debug=True)
    nbrs, scores = list(zip(*res))
    prd_khop_nbr.append({"product_id":u, "khop_nbr":nbrs, "cum_scores":scores})

prd_khop_nbr = pd.DataFrame(prd_khop_nbr)


#prd_khop_nbr.to_parquet(base / "model/products_khop_neighbors_map.parquet")

# Louvain communities; uses edge 'weight' if present.
# resolution >1 → more/smaller communities; <1 → fewer/larger
communities = nx.algorithms.community.louvain_communities(
    G, weight="weight", resolution=1.0, seed=42
)

def community_stats_louvain(
    G: nx.Graph,
    communities: list[set],
    weight_key: str = "weight",
    top_k: int = 5,
    compute_avg_clustering: bool = True,
    compute_betweenness: bool = False,  # expensive; computed on each subgraph if True
):
    """
    Compute statistics for each Louvain community.

    Returns:
      comm_stats: pd.DataFrame with one row per community
      node_to_comm: dict[node] -> community_id (size-sorted for stability)
    """
    # 0) Normalize communities -> sort by size (largest gets id 0)
    comms_sorted = sorted(communities, key=len, reverse=True)
    node_to_comm = {n: cid for cid, S in enumerate(comms_sorted) for n in S}

    # 1) Global totals
    # total edge weight (undirected): sum over edges once
    W_total = 0.0
    for _, _, d in G.edges(data=True):
        w = float(d.get(weight_key, 1.0))
        if np.isfinite(w) and w > 0:
            W_total += w
    twoW = 2.0 * W_total
    if twoW <= 0:
        raise ValueError("Total edge weight is zero or invalid; check your edge weights.")

    # weighted degree of every node in G (strength)
    wdeg_all = dict(G.degree(weight=weight_key))

    rows = []
    for cid, S in enumerate(comms_sorted):
        n = len(S)
        if n == 0:
            continue

        # Induced subgraph
        H = G.subgraph(S)

        # Internal edges / weight
        e_in = H.number_of_edges()
        w_in = 0.0
        for _, _, d in H.edges(data=True):
            w_in += float(d.get(weight_key, 1.0))

        # Volume of S (sum of weighted degrees of nodes in S)
        volS = float(sum(wdeg_all.get(u, 0.0) for u in S))

        # External cut weight (edges from S to V \ S), counted once
        # volS = 2*w_in + w_out  =>  w_out = volS - 2*w_in
        w_out = max(0.0, volS - 2.0 * w_in)

        # External edge count (optional, quick scan of boundary)
        ext_e = 0
        boundary_nodes = 0
        boundary_share_sum = 0.0  # mean(ext_weight / total_weight) over boundary nodes
        for u in S:
            ext_w_u = 0.0
            deg_u = 0
            for v, d in G[u].items():
                if v in S:
                    continue
                ext_w_u += float(d.get(weight_key, 1.0))
                ext_e += 1
                deg_u += 1
            if ext_w_u > 0.0:
                boundary_nodes += 1
                total_u = wdeg_all.get(u, 0.0)
                if total_u > 0:
                    boundary_share_sum += (ext_w_u / total_u)

        # Density inside the community (unweighted)
        max_edges = n * (n - 1) / 2.0
        density = (e_in / max_edges) if max_edges > 0 else 0.0

        # Average (unweighted) degree inside H
        avg_deg = (2.0 * e_in / n) if n > 0 else 0.0

        # Average weighted degree (strength) restricted to nodes in S
        avg_strength = (volS / n) if n > 0 else 0.0

        # Conductance (weighted)
        denom = min(volS, twoW - volS)
        conductance = (w_out / denom) if denom > 0 else np.nan

        # Modularity contribution of this community (weighted, undirected):
        # Q_c = (w_in / (2W)) - ( (volS / (2W))^2 )
        modularity_contrib = (w_in / twoW) - ( (volS / twoW) ** 2 )

        # Average clustering within H (weighted if available)
        if compute_avg_clustering:
            try:
                avg_clust = nx.average_clustering(H, weight=weight_key)
            except Exception:
                avg_clust = nx.average_clustering(H)
        else:
            avg_clust = np.nan

        # Top hubs by weighted degree (within H)
        wdeg_H = dict(H.degree(weight=weight_key))
        top_hubs = sorted(wdeg_H.items(), key=lambda t: t[1], reverse=True)[:top_k]

        # Top "bridges": nodes with largest external weight share
        bridge_scores = []
        for u in S:
            total_w = float(wdeg_all.get(u, 0.0))
            if total_w <= 0:
                bridge_scores.append((u, 0.0))
                continue
            ext_w_u = 0.0
            for v, d in G[u].items():
                if v not in S:
                    ext_w_u += float(d.get(weight_key, 1.0))
            bridge_scores.append((u, ext_w_u / total_w))
        top_bridges = sorted(bridge_scores, key=lambda t: t[1], reverse=True)[:top_k]

        # Optional: Betweenness centrality inside H (can be slow)
        if compute_betweenness:
            bc = nx.betweenness_centrality(H, weight=lambda u, v, d: 1.0 / max(d.get(weight_key, 1e-12), 1e-12))
            top_bc = sorted(bc.items(), key=lambda t: t[1], reverse=True)[:top_k]
        else:
            top_bc = None

        rows.append({
            "community_id": cid,
            "size": n,
            "internal_edges": e_in,
            "external_edges": int(ext_e),
            "density": density,
            "avg_degree": avg_deg,
            "avg_strength": avg_strength,
            "w_internal": w_in,
            "w_external": w_out,
            "in_out_weight_ratio": (w_in / (w_out + 1e-12)),
            "conductance": conductance,
            "modularity_contrib": modularity_contrib,
            "avg_clustering": avg_clust,
            "boundary_nodes": int(boundary_nodes),
            "boundary_fraction": (boundary_nodes / n) if n > 0 else 0.0,
            "boundary_share_mean": (boundary_share_sum / boundary_nodes) if boundary_nodes > 0 else 0.0,
            "top_hubs": top_hubs,           # list of (node, weighted_degree)
            "top_bridges": top_bridges,     # list of (node, external_weight_share)
            "top_betweenness": top_bc,      # list of (node, bc) or None
        })

    comm_stats = pd.DataFrame(rows).sort_values(["size", "modularity_contrib"], ascending=[False, False]).reset_index(drop=True)
    return comm_stats, node_to_comm

comm_stats, node_to_comm = community_stats_louvain(
    G,
    communities,
    weight_key="weight",
    top_k=5,
    compute_avg_clustering=True,
    compute_betweenness=False  # set True if communities are small; it's slower
)

def add_purity(comm_stats: pd.DataFrame, node_to_comm: dict, products_df: pd.DataFrame,
               id_col="product_id", attrs=("aisle", "department")):
    df = products_df[[id_col, *attrs]].copy()
    df["community_id"] = df[id_col].map(node_to_comm)
    out = comm_stats.copy()

    for attr in attrs:
        # top value and its share inside each community
        grp = (df.dropna(subset=[attr])
                 .groupby("community_id")[attr]
                 .apply(lambda s: s.value_counts(normalize=True).head(1)))
        # explode into (community_id -> (top_attr, purity))
        idx = grp.index.get_level_values(0)
        top_val = grp.index.get_level_values(1)
        purity = grp.values
        tmp = pd.DataFrame({"community_id": idx, f"top_{attr}": top_val, f"{attr}_purity": purity})
        out = out.merge(tmp, on="community_id", how="left")
    return out


comm_stats2 = add_purity(comm_stats, node_to_comm, products, id_col="product_id", attrs=("aisle","department"))
comm_stats2.sort_values("size")

products_comm = pd.DataFrame(node_to_comm.items(), columns=["product_id", "community_id"])

#comm_stats2.to_parquet(BASE_DIR / "../models/louvain_stats.parquet")

def _prep_vis_weight(G, weight_key="weight", out_key="w_vis", lo=0.1, hi=1.0):
    """Create visualization weight 'w_vis' from 'weight' using log1p + min-max."""
    vals = []
    for _,_,d in G.edges(data=True):
        w = d.get(weight_key, None)
        if w is not None and np.isfinite(w) and w > 0:
            vals.append(float(w))
    if not vals:
        for u,v in G.edges():
            G[u][v][out_key] = 1.0
        return
    w = np.log1p(np.array(vals, dtype=float))
    mn, mx = w.min(), w.max()
    s = (w - mn)/(mx - mn) if mx > mn else np.ones_like(w)
    s = lo + (hi - lo)*s
    i = 0
    for u,v,d in G.edges(data=True):
        base = d.get(weight_key, None)
        if base is not None and np.isfinite(base) and base > 0:
            d[out_key] = float(s[i]); i += 1
        else:
            d[out_key] = lo

def plot_topk_communities_one_canvas(
    G: nx.Graph,
    communities,                # list[set(nodes)]
    K: int = 5,
    weight_key: str = "weight",
    vis_weight_key: str = "w_vis",
    seed: int = 42,
    iterations: int = 500,
    figsize=(13, 11),
    title="Top-5 Louvain communities (weight-aware)"
):
    # sort by size and take top-K
    comms_sorted = sorted(communities, key=len, reverse=True)[:K]
    # Induce subgraph of union of top-K nodes
    keep_nodes = set().union(*comms_sorted)
    H = G.subgraph(keep_nodes).copy()

    # Prepare visualization weights
    _prep_vis_weight(H, weight_key=weight_key, out_key=vis_weight_key)

    # Layout (weight-aware spring); smaller k => tighter packing
    n = max(1, H.number_of_nodes())
    k = 1.1 / np.sqrt(n)
    pos = nx.spring_layout(H, weight=vis_weight_key, seed=seed, k=k, iterations=iterations)

    # Assign colors by community id 0..K-1
    node_to_cid = {}
    for cid, S in enumerate(comms_sorted):
        for u in S: node_to_cid[u] = cid

    # Build a stable palette
    base = plt.get_cmap("tab20").colors
    cmap = ListedColormap((base * ((K // len(base)) + 1))[:K])

    # Draw edges (width uses vis weight)
    ewidths = [0.5 + 2.5 * H[u][v].get(vis_weight_key, 0.1) for u,v in H.edges()]
    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(H, pos, width=ewidths, alpha=0.18, edge_color="#666666")

    # Draw nodes by community
    for cid in range(len(comms_sorted)):
        nodes_c = [u for u in H.nodes() if node_to_cid.get(u, -1) == cid]
        color = to_hex(cmap(cid))
        # size by weighted degree inside H
        wdeg = dict(H.degree(weight=vis_weight_key))
        sizes = [min(50 + 12 * wdeg.get(u, 0), 900) for u in nodes_c]
        nx.draw_networkx_nodes(
            H, pos,
            nodelist=nodes_c,
            node_color=color,
            edgecolors="white",
            linewidths=0.5,
            node_size=sizes,
            alpha=0.95,
            label=f"Comm {cid} (n={len(nodes_c)})"
        )

    plt.legend(loc="upper right", frameon=True)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# --- usage ---
#plot_topk_communities_one_canvas(G, communities, K=2, weight_key="weight")

# Use weights if you have them (the library reads edge attr 'weight' by default)
node2vec = Node2Vec(
    G,
    dimensions=128,
    walk_length=50,
    num_walks=20,
    workers=4,
    weight_key='weight',
    p=1.0, q=0.5,       # tune (p>1, q<1) for DFS-ish; (p<1, q>1) for BFS-ish
    seed=42
)
w2v_model = node2vec.fit(window=10, min_count=1, batch_words=256, epochs=5)

keys = list(w2v_model.wv.key_to_index.keys())        # node ids (product_ids)
vectors = w2v_model.wv.vectors                       # numpy array (n_nodes × dim)

emb_dict = {pid: w2v_model.wv[pid]/np.linalg.norm(w2v_model.wv[pid]) for pid in keys}

embed = pd.DataFrame.from_dict(emb_dict, orient="index")
embed.index.name = "product_id"

embed = embed.rename(columns={i: f"emb_{i}" for i in range(embed.shape[1])})
embed = embed.reset_index()

# reduce size (float32) before writing
emb_cols = [c for c in embed.columns if c.startswith("emb_")]
embed[emb_cols] = embed[emb_cols].astype("float32")
embed["product_id"] = pd.to_numeric(embed["product_id"], errors="coerce").astype("int64")

products_comm["community_id"] = products_comm["community_id"].astype("int32")

products = products.merge(embed, on="product_id", how="left", sort=False, copy=False)
products["has_embedding"] = products[emb_cols[0]].notna().astype("int8")
products[emb_cols] = products[emb_cols].fillna(0.0).astype("float32")

products = products.merge(products_comm, on="product_id", how="left", sort=False, copy=False)
products.to_parquet(BASE_DIR / "../data/parquets/product_embeddings.parquet", index=False)

