import uuid
import pandas as pd
from vowpalwabbit import Workspace
from datetime import datetime
from .ranker import XGBRanker, VWRanker
from .features import feats_from_context, build_adf
from .storage import AttributionStore

class RealTimeEngine:
    def __init__(self):
        self.attr = AttributionStore()
        self.vw_p1  = Workspace("--cb_explore_adf --cover 10 --quiet")
        self.vw_p2  = Workspace("--cb_explore_adf --cover 10 --quiet")
        self.vw_p4  = Workspace("--cb_explore_adf --cover 10 --quiet")

        self.vw_ranker = VWRanker()
        self.xgb_ranker = XGBRanker()

        self.catalog = pd.read_csv("models/catalog.csv")

    # ------------------------
    #   MAIN RECOMMENDATION
    # ------------------------
    def recommend(self, context):

        shared_feats = feats_from_context(context)

        # build working df
        items = self.catalog.copy()
        items["rank_score"] = [
            self.vw_ranker.score_one(r.grocery_item, r.category)
            for r in items.itertuples(index=False)
        ]

        # ----------------------------
        # P1 baseline  (no rank, full set)
        # ----------------------------
        adf_p1 = build_adf(shared_feats, items, include_rank=False)
        probs1 = list(self.vw_p1.predict(self.vw_p1.parse(adf_p1)))
        idx1   = int(pd.Series(probs1).sample(weights=probs1).index[0])
        item1  = items.iloc[idx1].grocery_item

        # ----------------------------
        # P2 Top-10 VW ranker
        # ----------------------------
        top10_vw = items.sort_values("rank_score", ascending=False).head(10)
        adf_p2 = build_adf(shared_feats, top10_vw, include_rank=True)
        probs2 = list(self.vw_p2.predict(self.vw_p2.parse(adf_p2)))
        idx2   = int(pd.Series(probs2).sample(weights=probs2).index[0])
        item2  = top10_vw.iloc[idx2].grocery_item

        # ----------------------------
        # P4 Top-10 XGB ranker
        # ----------------------------
        items["xgb_score"] = self.xgb_ranker.score(items)
        top10_xgb = items.sort_values("xgb_score", ascending=False).head(10)
        adf_p4 = build_adf(shared_feats, top10_xgb, include_rank=True)
        probs4 = list(self.vw_p4.predict(self.vw_p4.parse(adf_p4)))
        idx4   = int(pd.Series(probs4).sample(weights=probs4).index[0])
        item4  = top10_xgb.iloc[idx4].grocery_item

        # Choose SERVED policy (P1 for now)
        served_policy = "P1"
        served_item   = item1
        served_prob   = probs1[idx1]

        request_id = str(uuid.uuid4())

        self.attr.add({
            "request_id": request_id,
            "policy": served_policy,
            "item": served_item,
            "prob": served_prob,
            "context": context,
            "ts": datetime.utcnow()
        })

        return {
            "request_id": request_id,
            "item": served_item,
            "prob": served_prob,
            "policy": served_policy,
            "debug": {
                "p2_item": item2,
                "p4_item": item4
            }
        }

    # ------------------------
    #   REWARD ATTRIBUTION
    # ------------------------
    def reward(self, request_id, item, revenue):
        return self.attr.match(request_id, item, revenue)