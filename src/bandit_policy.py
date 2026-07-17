from abc import ABC, abstractmethod
from typing import Iterable, Dict, List, Any, Optional
import numpy as np
import random
import json

from vowpalwabbit import Workspace

class BanditPolicy(ABC):
    
    @staticmethod
    def _normalize_probs(probs):
        s = float(sum(probs))
        if s > 0:
            return [float(p) / s for p in probs]
        n = len(probs)
        return [1.0 / n] * n if n else []

    @staticmethod
    def _sample_index(probs):
        if not probs:
            return 0
        p = np.asarray(probs, dtype=float)
        p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
        return int(np.random.choice(len(p), p=p))    
       
    @abstractmethod
    def select_action(
        self,
        shared_context: Dict[str, Any],
        action_context: List[Dict[str, Any]],
        candidates: List[int],
        ) -> Dict[str, Any]:
        pass

    def update_models(self, new_ensemble: Any) -> None:
        """Optional: replace the underlying model(s)."""
        pass
    
   
class VwAdfXGBPolicy(BanditPolicy):

    DEFAULT_ARGS = (
        "--cb_explore_adf "
        "--cover 5 "
        "--quiet "
        "--epsilon 0.1 "
        "-q ca "
        "--learning_rate 0.1"
    )

    def __init__(
        self,
        vw_bandit_args: str = DEFAULT_ARGS,
        model_path: str | None = None
    ):
        args = vw_bandit_args

        if model_path:
            args = f"{args} -i {model_path}"

        self.vw = Workspace(args)
      
    
    @staticmethod
    def _vw_predict_probs(vw: Workspace, adf_lines):
        ex = vw.parse(adf_lines)
        probs = list(vw.predict(ex))
        vw.finish_example(ex)  # important to avoid memory growth
        return BanditPolicy._normalize_probs(probs)

    @staticmethod
    def _add_label_to_adf(lines_no_label: list[str], chosen_idx: int, *, cost: float, prob: float) -> list[str]:
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
    
    @staticmethod
    def _feats_from_dict(features: Dict[str, Any]) -> str:    
        
        """Action is a flat dict with at least product_id and optional numeric fields.   
       
        |a item=123 dept=4 aisle=17 comm=901 brand=55 rank:0.8421 price:2.4900 promo=1
        """        
        with open("xgb_feature_names.json") as f:
            xgb_features = json.load(f)
        cat_keys = xgb_features["categorical"]
        
        toks: List[str] = []
        for k in cat_keys:
            v = features.get(k)
            if v is not None:
                toks.append(f"{k}={int(v)}")  

        for k, v in features.items():
            if k in cat_keys:
                continue
            if v is None:
                continue
            if isinstance(v, (int, float)):
                toks.append(f"{k}:{float(v):.6f}")
            else:
                toks.append(f"{k}={str(v).replace(' ', '_')}")

        return " ".join(toks)
    
    @staticmethod
    def _build_adf(
        shared_feats: Optional[Dict[str, Any]],
        actions: Iterable[Dict[str, Any]]
    ) -> List[str]:
        shared_adf = "shared |s"
        if shared_feats:
            sf = VwAdfXGBPolicy._feats_from_dict(shared_feats)
            if sf:
                shared_adf += " " + sf

        lines: List[str] = [shared_adf]

        for a in actions:
            af = VwAdfXGBPolicy._feats_from_dict(a)
            lines.append("|a" + ((" " + af) if af else ""))

        return lines   

    def select_action(self, shared_context, action_context, candidates):
        served_adf = VwAdfXGBPolicy._build_adf(shared_context, action_context)        
        probs = VwAdfXGBPolicy._vw_predict_probs(self.vw, served_adf) # propensity
        served_idx = VwAdfXGBPolicy._sample_index(probs)
        served_pid = candidates[served_idx]
        out =  {    
                    "policy_name": "VW_ADF",
                    "chosen_index": served_idx,
                    "propensity": probs[served_idx],
                    "pid_shown": served_pid,
                    "comment": served_adf,
                } 
        
        return out   
    
    def probs_given_adf(self, adf: list[str]) -> list[float]:
        probs = VwAdfXGBPolicy._vw_predict_probs(self.vw, adf) # propensity
        return probs 
      
    
# class SoftmaxXGBPolicy(BanditPolicy):
#     def __init__(self, score_fn, temperature=1.0):
#         self.score_fn = score_fn
#         self.temp = temperature

#     def select_action(self, context, candidates):
#         probs = np.array([self.score_fn(c)[0] for c in candidates])
#         # Apply softmax: exp(prob/temp) / sum
#         exp_probs = np.exp(probs / self.temp)
#         action_probs = exp_probs / exp_probs.sum()
#         chosen_idx = np.random.choice(len(candidates), p=action_probs)
#         action = candidates[chosen_idx]
#         return action, action_probs[chosen_idx], {"xgb_mean": probs[chosen_idx]}


# class EpsilonGreedyXGBPolicy(BanditPolicy):
#     def __init__(self, score_fn, epsilon=0.02):
#         self.score_fn = score_fn          # callable: candidate -> (mean_prob, var)
#         self.epsilon = epsilon

#     def select_action(self, context, candidates):
#         # Score all candidates
#         scores = {c: self.score_fn(c)[0] for c in candidates}
#         best = max(scores, key=scores.get)

#         if random.random() < 1 - self.epsilon:
#             action = best
#             prop = (1 - self.epsilon) + self.epsilon / len(candidates)
#         else:
#             action = random.choice(candidates)
#             prop = self.epsilon / len(candidates)

#         return action, prop, {"xgb_mean": scores[action]}
 
    
class RandomPolicy(BanditPolicy):
    def select_action(self, shared_context, action_context, candidates):
        action = random.choice(candidates)
        prop = 1.0 / len(candidates)

        out =  {    "policy_name": "Random",
                    "chosen_index": candidates.index(action),
                    "propensity": prop,
                    "pid_shown": action,
                    "comment": [],
                } 
        
        return out
