import requests

API_URL = "http://localhost:8000/predict"

samples = [
    {
        "customer_id": "C1001",
        "features": {
            "tenure": 3,
            "monthly_charges": 95.2,
            "contract_type": "Month-to-month",
        },
    },
    {
        "customer_id": "C1002",
        "features": {
            "tenure": 24,
            "monthly_charges": 65.0,
            "contract_type": "One year",
        },
    },
    {
        "customer_id": "C1003",
        "features": {
            "tenure": 60,
            "monthly_charges": 45.3,
            "contract_type": "Two year",
        },
    },
]

for sample in samples:
    response = requests.post(API_URL, json=sample)
    print(f"{sample['customer_id']} → {response.json()}")
