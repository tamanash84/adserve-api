from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def build_faiss_index(
    parquet_path: str,
    out_dir: str,
    index_type: str = "hnsw32",
    ef_construction: int = 200,
    ef_search: int = 64,
) -> Tuple[str, str, str]:
    """
    Build FAISS ANN artifacts from product_embeddings.parquet.

    Input parquet columns:
      - product_id (int)
      - emb_0..emb_128 (float)

    Output files written to out_dir:
      - index.faiss   : FAISS index
      - ids.npy       : row->product_id mapping (same order as vectors added to index)
      - emb_norm.npy  : normalized embeddings (float32) stored as .npy (can be memmapped at query time)

    Returns: (index_path, ids_path, emb_path)
    """
    import faiss  # pip install faiss-cpu

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load parquet ----
    df = pd.read_parquet(parquet_path)
    if "product_id" not in df.columns:
        raise ValueError("Expected 'product_id' in parquet")

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("Expected embedding columns emb_0..emb_*")

    emb_cols = sorted(emb_cols, key=lambda x: int(x.split("_")[1]))

    ids = df["product_id"].to_numpy()
    X = df[emb_cols].to_numpy(dtype=np.float32, copy=True)
    
    
    row_norms = np.linalg.norm(X, axis=1)
    good = row_norms > 1e-6   # threshold; tune if needed
    
    print("dropping bad vectors:", int((~good).sum()), "out of", len(row_norms))
    
    X = X[good]
    ids = ids[good]
    
    # ---- normalize for cosine similarity (cos = dot of normalized vectors) ----
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X /= norms  # now X is normalized

    # Save normalized vectors for fast query-time vector retrieval (mmap)
    emb_path = out / "emb_norm.npy"
    np.save(emb_path, X)

    # Save ids mapping
    ids_path = out / "ids.npy"
    np.save(ids_path, ids)
    
    # after X is loaded and normalized
    print("X dtype:", X.dtype)
    print("X shape:", X.shape)
    print("X finite:", np.isfinite(X).all())
    print("X min/max:", float(np.nanmin(X)), float(np.nanmax(X)))
    
    row_norms = np.linalg.norm(X, axis=1)
    print("norms min/max:", float(np.nanmin(row_norms)), float(np.nanmax(row_norms)))
    print("num near-zero norms:", int((row_norms < 1e-6).sum()))
    print("num NaN norms:", int(np.isnan(row_norms).sum()))

    # ---- build FAISS index (inner product) ----
    d = X.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(d)
        index.add(X)
    elif index_type.startswith("hnsw"):
        M = int(index_type.replace("hnsw", ""))
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(X)
    else:
        raise ValueError("Supported index_type: 'flat' or 'hnswXX' (e.g., 'hnsw32')")

    index_path = out / "index.faiss"
    faiss.write_index(index, str(index_path))

    return str(index_path), str(ids_path), str(emb_path)

index_path, ids_path, emb_path = build_faiss_index(
    parquet_path="../data/parquets/product_embeddings.parquet",
    out_dir="../data/ann",
    index_type="hnsw32",       # good default
    ef_construction=200,
    ef_search=64,
)

print(index_path, ids_path, emb_path)

