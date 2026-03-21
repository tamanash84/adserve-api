import requests
import json

# Change BASE for your deployment (local or remote)
#BASE = "https://adserve-api.onrender.com"
BASE = " http://127.0.0.1:8000"

def print_response(resp, label=""):
    print(f"\n--- {label} ---")
    print("Status :", resp.status_code)
    print("CT     :", resp.headers.get("content-type"))
    txt = resp.text
    print("Text   :", txt[:500], "..." if len(txt) > 500 else "")
    try:
        print("JSON   :", json.dumps(resp.json(), indent=2)[:1000])
    except Exception as e:
        print("JSON   : <parse failed>", repr(e))

# 1) Health
h = requests.get(f"{BASE}/health", timeout=10)
print_response(h, "HEALTH")

# 2) Config (optional)
cfg = requests.get(f"{BASE}/config", timeout=10)
print_response(cfg, "CONFIG (GET)")

# 3) Recommend with P2 (VW score + VW bandit)
payload_p2 = {
    "context": {
        "timestamp": "2026-03-19T10:22:00+01:00",
        "weather_7timer": "pcloudyday",
        "weekend": 0, "holiday": 0, "local_event": 0, "sports_on_tv": 0,
        "temperature_c": 17.0,
        # You can also pass demographics if you want to override:
        "pct_households_with_kids": 30.0,
        "pct_single_households": 30.0,
        "pct_under_25": 30.0,
        "pct_65_plus": 30.0,
        "social_support_pct": 30.0,
        # Promo knobs:
        "on_promo": 0, "promo_type": "none", "promo_discount_pct": 0.0
    },
    "served_policy": "P2",
    "topK": 10
}
r2 = requests.post(f"{BASE}/recommend", json=payload_p2, timeout=20)
print_response(r2, "RECOMMEND P2")
served_req_id_p2, served_item_p2 = None, None
if r2.ok and r2.headers.get("content-type","").startswith("application/json"):
    body = r2.json()
    served_req_id_p2 = body.get("request_id")
    served_item_p2 = body.get("item")
    debug = body.get("debug", {})
    # print("\n[DEBUG P2]")
    # print(json.dumps(debug, indent=2))

# 4) Recommend with P4 (XGB score + VW bandit)
payload_p4 = {
    "context": {
        "timestamp": "2026-03-19T10:22:00+01:00",
        "weather_7timer": "pcloudyday",
        "weekend": 0, "holiday": 0, "local_event": 0, "sports_on_tv": 0,
        "temperature_c": 17.0,
        # You can also pass demographics if you want to override:
        "pct_households_with_kids": 30.0,
        "pct_single_households": 30.0,
        "pct_under_25": 30.0,
        "pct_65_plus": 30.0,
        "social_support_pct": 30.0,
        # Promo knobs:
        "on_promo": 0, "promo_type": "none", "promo_discount_pct": 0.0
    },
    "served_policy": "P4",
    "topK": 10
}
r4 = requests.post(f"{BASE}/recommend", json=payload_p4, timeout=20)
print_response(r4, "RECOMMEND P4")
served_req_id_p4, served_item_p4 = None, None
if r4.ok and r4.headers.get("content-type","").startswith("application/json"):
    body = r4.json()
    served_req_id_p4 = body.get("request_id")
    served_item_p4 = body.get("item")
    debug = body.get("debug", {})
    # print("\n[DEBUG P4]")
    # print(json.dumps(debug, indent=2))

# 5) Reward (for the last served call, i.e., P4)
if served_req_id_p4 and served_item_p4:
    rw = requests.post(
        f"{BASE}/reward",
        json={"request_id": served_req_id_p4, "purchased_item": served_item_p4, "revenue": 1.99},
        timeout=10
    )
    print_response(rw, "REWARD (P4)")
else:
    print("\nNo valid P4 response to reward; check server logs or payload schema.")