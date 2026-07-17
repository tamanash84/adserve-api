from pathlib import Path

class Paths:
    ROOT = Path(__file__).resolve().parent

    DATA     = ROOT / "data"
    PARQUETS = DATA / "parquets"

    STORES  = PARQUETS / "stores_fixed_context.parquet"
    WEATHER = PARQUETS / "weather_hourly.parquet"
    EVENTS  = PARQUETS / "events_hourly.parquet"
    PRICES  = PARQUETS / "product_base_prices.parquet"
    EMBEDS  = PARQUETS / "product_embeddings.parquet"
    POS     = PARQUETS / "orders_prod_multistore_pos.parquet"
    
    BANDIT  = DATA / "bandit"
    BANDIT_WAL  = BANDIT / "wal"
    BANDIT_TRAIN  = BANDIT / "training"    
    TRAIN_DAILY_PATTERN = BANDIT_TRAIN / "date=*" / "train.parquet"
    TRAIN_HOURLY_PATTERN = BANDIT_TRAIN / "date=*" / "hour=*" / "train.parquet"
    
    POS  = DATA / "pos"
    POS_WAL  = POS / "wal"
    POS_TRAIN  = POS / "purchases"
    
    MODEL = ROOT / "model"
    XGB_MODEL = MODEL / "xgboost"
    VW_POLICY = MODEL / "vw"
    

