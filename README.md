# AdServe API (Production-Ready)

This service provides real-time recommendations using:

- VW CB-ADF contextual bandits
- XGBoost ranking model
- Delayed attribution tracking

## Endpoints

### POST /recommend

Input:
{
  "context": {...}
}

Output:
{
  "request_id": "...",
  "item": "Milk 1L",
  "policy": "P1",
  "prob": 0.23,
  "debug": {...}
}

### POST /reward

Input:
{
  "request_id": "...",
  "purchased_item": "Milk 1L",
  "revenue": 2.99
}

Output:
{
  "matched": true,
  "policy": "P1",
  "reward": 2.99
}

## Run locally