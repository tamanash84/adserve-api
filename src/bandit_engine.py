from datetime import datetime
from typing import Any
import os, json
import threading
import numpy as np
from pathlib import Path
from bandit_policy import BanditPolicy, RandomPolicy


class BanditEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # first check (no locking, fast path)
        if cls._instance is None:
            with cls._lock:
                # second check (with lock)
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._model_lock = threading.Lock()
        return cls._instance

    def __init__(self, policy: BanditPolicy, p_random: float = 0.1, seed:int = 42):
        if self._initialized: return
        self._initialized = True
        self.policy = policy
        self.random_policy = RandomPolicy()
        self.p_random = p_random
        self.rng = np.random.default_rng(seed)
    
    @staticmethod   
    def log_wal(out_dir: Path, record: dict[str, Any], prefix: str, fsync:bool = False):   
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.fromisoformat(record["timestamp"])        
        date_str = ts.date().isoformat()
        out_path = out_dir / f"{prefix}_{date_str}.jsonl"

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            if fsync:
                os.fsync(f.fileno())   

    def recommend(self, shared_context: dict[str, Any], 
                  action_context: list[dict[str, Any]],
                  candidates: list[int]) -> dict[str, Any]:
        
        with self._model_lock:
            policy = self.random_policy if self.rng.random() < self.p_random else self.policy
            
        return policy.select_action(shared_context, action_context, candidates)
    
    
    def reload_policy(self, new_policy: BanditPolicy):
        """Atomically swap the policy (thread-safe)."""
        with self._model_lock:   # protect write
            self.policy = new_policy
    


    

            

    
        





    
    


