# AdServe API (Production-Ready)

This service provides real-time recommendations using:

* VW CB-ADF contextual bandits
* XGBoost ranking model
* Delayed attribution tracking

## 

### End‑to‑end event flow (Salesforce POS → API → Bandit)

##### 1\) Impression (engine)

&#x20;  - Engine selects "served\_item" under policy Pk with prob p.

&#x20;  - Engine calls store.add({

&#x20;      request\_id,

&#x20;      policy: "P4",

&#x20;      item: served\_item,

&#x20;      prob: p,

&#x20;      context: {..., session\_id/device\_id, ...},

&#x20;      ts: now\_utc,

&#x20;      expires\_at: now\_utc + window,

&#x20;      # (optionally ADF lines \& idx if you want online VW learning on match)

&#x20;    })

&#x20;  - Optionally render "?rid=<request\_id>" as QR / shortlink on digital signage.



##### 2\) Checkout at POS (Salesforce)

&#x20;  - When the customer pays, POS sends webhook:

&#x20;    POST /event/purchase

&#x20;    {

&#x20;      event\_id: "...",             # unique per transaction (idempotency)

&#x20;      request\_id: "...",           # preferred (if you used QR with rid)

&#x20;      session\_id: "AMS\_SCREEN\_07", # screen/device id, optional fallback

&#x20;      purchased\_item: "SKU123",    # if POS can nominate the target SKU

&#x20;      occurred\_at: "2026-03-19T12:44:00Z",

&#x20;      lines: \[

&#x20;        {"sku":"SKU123","qty":1,"unit\_price":1.29,"discount":0.10},

&#x20;        {"sku":"SKU999","qty":2,"unit\_price":0.99,"discount":0.00}

&#x20;      ]

&#x20;    }



##### 3\) API handling

&#x20;  - If request\_id present -> deterministic match()

&#x20;  - Else if (session\_id + purchased\_item) -> match\_by\_session()

&#x20;  - Else -> match\_by\_item() for each line

&#x20;  - Compute revenue = (unit\_price - discount) \* qty (or your margin rule)



##### 4\) Attribution \& learning

&#x20;  - Store sets impression.matched=True, reward=revenue.

&#x20;  - Optional: on\_match(impression) hook triggers VW/XGB learning with IPS,

&#x20;              using prob saved at impression time.



##### 5\) Maintenance (background sweeper)

&#x20;  - Every 60s, sweep\_expired() removes expired/matched impressions from deques

&#x20;    to keep memory bounded.



##### 6\) Analytics / audit

&#x20;  - Log impressions \& rewards (CSV/DB) for KPI dashboards and offline eval.



##### Error/edge cases:

&#x20;  - Late purchase -> reason: "expired"

&#x20;  - Unknown request\_id -> 404

&#x20;  - Duplicate event\_id -> reason: "duplicate\_event"

&#x20;  - Item mismatch (strict policy) -> reason: "item\_mismatch"

