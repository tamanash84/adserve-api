import joblib
import json
from vowpalwabbit import Workspace

class XGBRanker:
    def __init__(self, base_path="models"):
        self.model = joblib.load(f"{base_path}/xgb_ranker.pkl")
        self.pre = joblib.load(f"{base_path}/preprocess.pkl")
        self.meta = json.load(open(f"{base_path}/feature_meta.json"))

    def score(self, df):
        X = self.pre.transform(df[self.meta["feature_columns"]])
        return self.model.predict(X)


class VWRanker:
    def __init__(self, args="--loss_function=squared --quiet"):
        self.vw = Workspace(args)

    def score_one(self, item, cat):
        return float(self.vw.predict(f"|i item={item} cat={cat}"))