from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ANN:
    index: object
    ids: np.ndarray                 # shape (N,)
    id_to_row: Dict[int, int]       # product_id -> row index
    X_norm: np.memmap               # normalized embeddings (memmap), shape (N, D)


def load_ann(artifacts_dir: str = "ann_artifacts") -> ANN:
    """
    Load ANN artifacts once (e.g., at app startup).
    Expects:
      - artifacts_dir/index.faiss
      - artifacts_dir/ids.npy
      - artifacts_dir/emb_norm.npy
    """
    import faiss  # pip install faiss-cpu
    from pathlib import Path

    p = Path(artifacts_dir)
    index = faiss.read_index(str(p / "index.faiss"))
    ids = np.load(str(p / "ids.npy"), allow_pickle=True)

    # Memmap embeddings so you don't have to hold all vectors in RAM
    X_norm = np.load(str(p / "emb_norm.npy"), mmap_mode="r")

    id_to_row = {int(pid): int(i) for i, pid in enumerate(ids)}
    return ANN(index=index, ids=ids, id_to_row=id_to_row, X_norm=X_norm)


def query_similar(
    ann: ANN,
    product_id: int,
    *,
    topk: int = 10,
) -> List[Tuple[int, float]]:
    """
    Return topk similar items for product_id as [(neighbor_id, cosine_similarity), ...].
    Cosine similarity is inner product because vectors are normalized.
    """
    if product_id not in ann.id_to_row:
        raise KeyError(f"product_id {product_id} not found in ANN ids map")

    q_row = ann.id_to_row[product_id]
    q = np.asarray(ann.X_norm[q_row:q_row + 1], dtype=np.float32)  # (1, D)

    # search topk+1 and drop self if it appears
    D, I = ann.index.search(q, topk + 1)

    out: List[Tuple[int, float]] = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        pid = int(ann.ids[int(idx)])
        if pid == product_id:
            continue
        out.append((pid, float(score)))
        if len(out) >= topk:
            break

    return out

########### Example use ###############

# ann = load_ann("../data/ann_artifacts")
# try:          # load once at startup
#     neighbors = query_similar(ann, 44683, topk=10)
#     print(neighbors)  # [(pid2, sim), (pid3, sim), ...]
# except KeyError:
#     print("product_id not found in ANN ids map")
