import uuid
import numpy as np
import pandas as pd
from vowpalwabbit import Workspace
from datetime import datetime, timedelta

from .ranker import XGBRanker, VWRanker
from .features import feats_from_context, build_adf
from .storage import AttributionStore


class RealTimeEngine:
    """
    Four-policy realtime engine:

      P1: VW Bandit only (full catalog; no rank feature)
      P2: VW Ranker Top-K + VW Bandit (includes rank feature)
      P3: Random from Top-5 by VW Ranker (uniform; no bandit)
      P4: XGBoost Ranker Top-K + VW Bandit (includes rank feature = xgb_score)

    Served policy is configurable via `served_policy` (default "P1").
    """

    def __init__(
        self,
        vw_bandit_args="--cb_explore_adf --cover 10 --quiet",
        served_policy="P1",
        topK=10,
        randomK=5,
        window_sec=6*3600,
        catalog_csv="models/catalog.csv",
    ):
        self.attr = AttributionStore()

        # Bandit workspaces for the policies that learn/predict
        self.vw_p1 = Workspace(vw_bandit_args)
        self.vw_p2 = Workspace(vw_bandit_args)
        self.vw_p4 = Workspace(vw_bandit_args)

        # Rankers
        self.vw_ranker = VWRanker()
        self.xgb_ranker = XGBRanker()

        # Items catalog
        self.catalog = pd.read_csv(catalog_csv)

        # Config
        self.served_policy = served_policy.upper()
        self.topK = int(topK)
        self.randomK = int(randomK)
        
        self.window_sec = window_sec

    # ---------- internal helpers ----------

    @staticmethod
    def _normalize_probs(probs):
        s = float(sum(probs))
        if s > 0:
            return [float(p) / s for p in probs]
        n = len(probs)
        return [1.0 / n] * n if n else []

    def _vw_predict_probs(self, vw: Workspace, adf_lines):
        ex = vw.parse(adf_lines)
        probs = list(vw.predict(ex))
        vw.finish_example(ex)  # important to avoid memory growth
        return self._normalize_probs(probs)

    @staticmethod
    def _sample_index(probs):
        if not probs:
            return 0
        p = np.asarray(probs, dtype=float)
        p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
        return int(np.random.choice(len(p), p=p)) 

    def _add_label_to_adf(self, lines_no_label: list[str], chosen_idx: int, *, cost: float, prob: float) -> list[str]:
        if not lines_no_label or chosen_idx is None:
            return list(lines_no_label or [])
        lines = list(lines_no_label)
        pos = 1 + int(chosen_idx)  # shared at 0, actions from 1
        if pos < 1 or pos >= len(lines):
            return lines
        safe_prob = max(float(prob), 1e-12)
        safe_cost = float(cost)
        lines[pos] = f"0:{safe_cost:.8f}:{safe_prob:.8f} " + lines[pos]
        return lines

    def _expire_and_learn_zero(self) -> None:
        """
        Learn zero-reward for expired & unmatched impressions before the store prunes them.
        Scans all item deques; does not mutate deques (the store sweeper prunes afterwards).
        """
        store = self.attr
        now = datetime.utcnow()

        # Map per-policy VW workspaces available on the engine.
        policy_to_vw = {
            "P1": getattr(self, "vw_p1", None),
            "P2": getattr(self, "vw_p2", None),
            "P4": getattr(self, "vw_p4", None),
            # P3 is random → no VW model
        }

        # Iterate over deques (time-ordered in your store)
        for item, dq in list(store.by_item.items()):
            # We check a snapshot; we do not modify dq here.
            for imp in list(dq):
                exp_at = imp.get("expires_at")
                if not exp_at or imp.get("matched"):
                    continue
                if now <= exp_at:
                    # Since deques are ordered, once we hit a non-expired one we can stop for this item
                    break

                # expired & unmatched → perform zero-reward learning if we have ADF metadata
                adf   = imp.get("adf_lines")
                idx   = imp.get("chosen_idx")
                prob  = imp.get("prob")
                pol   = imp.get("policy")
                vw    = policy_to_vw.get(pol)

                if vw is None or adf is None or idx is None or prob is None:
                    # Either no VW policy (P3) or metadata missing → skip
                    continue

                learn_lines = self._add_label_to_adf(adf, int(idx), cost=0.0, prob=float(prob))
                ex = vw.parse(learn_lines)
                try:
                    vw.learn(ex)
                finally:
                    vw.finish_example(ex)

    # ---------- main API ----------

    def recommend(self, context: dict):
        shared_feats = feats_from_context(context)

        # Working frame (copy catalog)
        items = self.catalog.copy()

        # 1) VW ranker scores (used by P2 & P3)
        items["rank_score"] = self.vw_ranker.score(items, context)

        # -------- P1: VW bandit only (full set; no rank feature) --------
        adf_p1 = build_adf(shared_feats, items, include_rank=False)
        probs1 = self._vw_predict_probs(self.vw_p1, adf_p1)
        idx1 = self._sample_index(probs1)
        item1 = items.iloc[idx1].grocery_item if len(items) else None
        prob1 = float(probs1[idx1]) if len(items) else 0.0

        # -------- P2: VW ranker Top-K + VW bandit (rank feature) --------
        top_vw = items.sort_values("rank_score", ascending=False).head(self.topK).reset_index(drop=True)
        adf_p2 = build_adf(shared_feats, top_vw, include_rank=True)
        probs2 = self._vw_predict_probs(self.vw_p2, adf_p2) if len(top_vw) else []
        idx2 = self._sample_index(probs2) if len(top_vw) else 0
        item2 = top_vw.iloc[idx2].grocery_item if len(top_vw) else None
        prob2 = float(probs2[idx2]) if len(top_vw) else 0.0

        # -------- P3: Random from Top-5 (uniform; no bandit) --------
        top5 = items.sort_values("rank_score", ascending=False).head(self.randomK).reset_index(drop=True)
        if len(top5):
            idx3 = int(np.random.randint(len(top5)))
            item3 = top5.iloc[idx3].grocery_item
            prob3 = 1.0 / float(len(top5))
        else:
            idx3, item3, prob3 = 0, None, 0.0

        # -------- P4: XGBoost ranker Top-K + VW bandit (rank feature) --------
        items["xgb_score"] = self.xgb_ranker.score(items, context)
        top_xgb = items.sort_values("xgb_score", ascending=False).head(self.topK).reset_index(drop=True)

        # Map xgb_score -> rank_score so build_adf(..., include_rank=True) emits "rank:<val>"
        top_xgb_for_adf = top_xgb.copy()
        top_xgb_for_adf["rank_score"] = top_xgb_for_adf["xgb_score"]

        adf_p4 = build_adf(shared_feats, top_xgb_for_adf, include_rank=True)
        probs4 = self._vw_predict_probs(self.vw_p4, adf_p4) if len(top_xgb) else []
        idx4 = self._sample_index(probs4) if len(top_xgb) else 0
        item4 = top_xgb.iloc[idx4].grocery_item if len(top_xgb) else None
        prob4 = float(probs4[idx4]) if len(top_xgb) else 0.0

        # -------- choose which policy to serve --------
        if self.served_policy == "P1":
            served_item, served_prob, served_adf, served_idx = item1, prob1, adf_p1, idx1
        elif self.served_policy == "P2":
            served_item, served_prob, served_adf, served_idx = item2, prob2, adf_p2, idx2
        elif self.served_policy == "P3":
            served_item, served_prob, served_adf, served_idx = item3, prob3, None, idx3
        elif self.served_policy == "P4":
            served_item, served_prob, served_adf, served_idx = item4, prob4, adf_p4, idx4
        else:
            # default to P1
            served_item, served_prob, served_idx = item1, prob1, idx1

        request_id = str(uuid.uuid4())
        self.attr.add({
            "request_id": request_id,
            "policy": self.served_policy,
            "item": served_item,
            "prob": served_prob,
            "context": context,
            "ts": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(seconds=self.window_sec),            
            # NEW: fields needed for online learning on match
            "adf_lines": served_adf,         # list[str] or None for P3
            "chosen_idx": int(served_idx)    # index within the ADF action set

        })

        return {
            "request_id": request_id,
            "policy": self.served_policy,
            "item": served_item,
            "prob": served_prob,
            "debug": {
                "p1": {"item": item1, "prob": prob1, "idx": idx1},
                "p2": {"item": item2, "prob": prob2, "idx": idx2, "topK": self.topK},
                "p3": {"item": item3, "prob": prob3, "idx": idx3, "randomK": self.randomK},
                "p4": {"item": item4, "prob": prob4, "idx": idx4, "topK": self.topK},
            }
        }

    # ---------- reward attribution ----------

    # def reward(self, request_id, item, revenue):
    #     return self.attr.match(request_id, item, revenue)
    
    def reward(self, request_id, item, revenue):
        # 1) let the store match (and mark matched/expired/etc.)
        res = self.attr.match(request_id, item, float(revenue))   # existing API
        # res ~ {"matched": True/False, "policy": "...", "item": "...", "reward": ...}
        if not res or not res.get("matched"):
            return res
    
        # 2) Retrieve the stored impression to get ADF + idx + prob
        imp = self.attr.by_request.get(request_id)  # impression dict exists on match
        if not imp:
            return res
    
        policy = imp.get("policy")
        adf = imp.get("adf_lines")
        idx = imp.get("chosen_idx")
        prob = imp.get("prob")
    
        # Only VW CB-ADF policies learn (P1, P2, P4). P3 is random, no VW model.
        policy_to_vw = {
            "P1": getattr(self, "vw_p1", None),
            "P2": getattr(self, "vw_p2", None),
            "P4": getattr(self, "vw_p4", None),
        }
        vw = policy_to_vw.get(policy)
    
        # Guardrails
        if vw is None or adf is None or idx is None or prob is None:
            # no learning (e.g., P3 or missing metadata)
            return res
    
        # 3) Build labeled ADF with cost = -reward (maximize reward)
        cost = -float(res.get("reward", 0.0))
        learn_lines = self._add_label_to_adf(adf, int(idx), cost=cost, prob=float(prob))
    
        # 4) Parse → learn → finish
        ex = vw.parse(learn_lines)
        try:
            vw.learn(ex)
        finally:
            vw.finish_example(ex)
    
        return res