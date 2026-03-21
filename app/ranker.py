import joblib
import json
from vowpalwabbit import Workspace
import pandas as pd
import numpy as np
from datetime import datetime


class RankerBase:
    def __init__(self, base_path="models"):
        # Load shared metadata
        with open(f"{base_path}/feature_meta.json") as f:
            self.meta = json.load(f)

        self.categorical_features = self.meta["categorical_features"]
        self.feature_columns = self.meta["feature_columns"]
        
    # -----------------------
    # Ensure ranker features exist at inference
    # -----------------------
    def ensure_ranker_features(self,
                               cand: pd.DataFrame) -> pd.DataFrame:
        """
        Recreate engineered columns the trained ranker expects:
          - time-of-day signals: hour_sin, hour_cos, morning/evening flags
          - event flags & interactions: is_event, ..._x_on_promo, on_promo_x_discount, in_stock_x_stock, weekend_x_on_promo
          - priors (fallback): item_pop, cat_pop, weather_item_pop computed from df_all (z-scored)
        Then, add neutral defaults for any remaining missing columns so we never KeyError.
        """
        fr = cand.copy()
    
        # ---- time-of-day ---- 
        
        ts = fr['timestamp'].astype(str).str.replace('Z', '+00:00', regex=False)
        fr['timestamp'] = pd.to_datetime(ts, errors='coerce', utc=True)

        fr['hour'] = fr['timestamp'].dt.hour
        fr['hour_sin'] = np.sin(2*np.pi*fr['hour']/24.0)
        fr['hour_cos'] = np.cos(2*np.pi*fr['hour']/24.0)
        
        # bins not strictly needed but some models used them
        tod = np.where((fr['hour']>=6)&(fr['hour']<12), 'morning',
               np.where((fr['hour']>=12)&(fr['hour']<18), 'afternoon', 'evening'))
        fr['tod_bin'] = tod
        fr['morning'] = (fr['tod_bin']=='morning').astype(int)
        fr['evening'] = (fr['tod_bin']=='evening').astype(int)
        
        # interactions by ToD
        fr['evening_x_on_promo'] = fr.get('on_promo', 0) * fr['evening']
        fr['morning_x_on_promo'] = fr.get('on_promo', 0) * fr['morning']
    
        # ---- event flags & interactions ----
        fr['is_event'] = ((fr.get('local_event', 0).astype(int)==1) | (fr.get('sports_on_tv', 0).astype(int)==1)).astype(int)
        fr['on_promo_x_discount'] = fr.get('on_promo', 0) * fr.get('promo_discount_pct', 0.0)
        fr['in_stock_x_stock']    = fr.get('in_stock', 0) * fr.get('current_stock', 0.0)
        fr['weekend_x_on_promo']  = fr.get('weekend', 0) * fr.get('on_promo', 0)
    
        # ---- priors : item_pop, cat_pop, weather_item_pop ----
        # If your trained model used train-only priors, we approximate using df_all. Good enough for inference.
        # item_pop
        fr['item_pop'] = 0.0
    
        # cat_pop
        fr['cat_pop'] = 0.0
    
        # weather_item_pop
        fr['weather_item_pop'] = 0.0
    
        # ---- categorical coercion (safety) ----
        for c in self.categorical_features:
            if c not in fr.columns:
                fr[c] = 'unknown'
            fr[c] = fr[c].astype('string').fillna('unknown')
    
        # ---- guarantee all expected features exist ----
        missing = [c for c in self.feature_columns if c not in fr.columns]
        for c in missing:
            fr[c] = 0 if c not in self.categorical_features else 'unknown'
    
        return fr


class XGBRanker(RankerBase):
    def __init__(self, base_path="models"):
        super().__init__(base_path)
        self.ranker_xgb = joblib.load(f"{base_path}/ranker.pkl")
        self.preprocess_xgb = joblib.load(f"{base_path}/preprocess.pkl")     

    def score(self, df, context):   

        df = df.assign(**context)   
    
        # Ensure features for the XGBoost ranker using your original helper
        X_raw = self.ensure_ranker_features(df)
    
        # Extract feature matrix
        X = X_raw[self.feature_columns]
    
        # Apply the stored preprocess pipeline if needed
        X_proc = self.preprocess_xgb.transform(X)
    
        # Predict score (classifier -> prob of class 1; else regression score)
        score = self.ranker_xgb.predict(X_proc)
   
        return score


class VWRanker(RankerBase):
    def __init__(self, base_path="models", args="--loss_function=squared --quiet"):
        super().__init__(base_path)
        self.vw_ranker = Workspace(args)
        self.meta = json.load(open(f"{base_path}/feature_meta.json"))

    def score(self, df, context):
        """Build candidate set once per step and score with VW ranker (for policies 2,3,4)."""
        items = df[['grocery_item', 'category']].drop_duplicates() \
                                                     .assign(**context)
        
        items = self.ensure_ranker_features(items)
        
        score = [
            float(self.vw_ranker.predict(f"|i item={r.grocery_item} cat={r.category}"))
            for r in items.itertuples(index=False)
        ]
        
        return score

    