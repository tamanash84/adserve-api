import requests, json

BASE = "http://localhost:8080"

r = requests.post(f"{BASE}/recommend", json={"context":{"category":"pantry","temperature_c":17}})
print("Recommend:", r.status_code, r.json())

req_id = r.json()["request_id"]
item   = r.json()["item"]

rw = requests.post(f"{BASE}/reward", json={"request_id": req_id, "purchased_item": item, "revenue": 1.99})
print("Reward   :", rw.status_code, rw.json())