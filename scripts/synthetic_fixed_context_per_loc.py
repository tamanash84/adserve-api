import sys
from pathlib import Path

# repo_root = parent of "scripts/"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.govt_demography_api import get_age_group, get_country_origin
from api.overpass_poi_api import get_poi_features
import pandas as pd

year = 2025

stores = pd.read_csv("stores.csv")
out = []
for row in stores.itertuples(index=False):
    store_id = int(row.store_id)
    lat      = row.latitude
    lon      = row.longitude
    pc = int(row.postcode)

    res1 = get_age_group(pc, year)
    res2 = get_country_origin(pc, year)
    
    res3 = get_poi_features(
            lat=lat,
            lon=lon,
            bands=((0, 50), (50, 200), (200, 800)),
            missing_value=-1.0,
            add_any=True,
            add_density=False,
            add_open_now=True,
            tz="Europe/Amsterdam",
            self_exclude_keywords=("albert heijn", "ah"),
        )
    
    res = {"store_id":store_id, **res1, **res2, **res3}
    out.append(res)

df = pd.DataFrame(out)
df.to_parquet("stores_fixed_context.parquet", index=False)