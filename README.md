# Fraud Detection API

A production-ready REST API built with FastAPI to predict customer churn based on input features.

## 📁 Project Structure



fraud-detection-project/ ├── models/ │   └── model.pkl ├── src/ │   └── api/ │       └── app.py ├── .venv/ ├── requirements.txt └── README.md


## 🚀 Service Management

Start or restart the API service:

```bash
sudo systemctl restart fraud-api.service


Check service status:

sudo systemctl status fraud-api.service


🧪 Example Request (via curl)

curl -X POST http://<your-public-ip>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "C12345", "features": {"tenure": 12, "monthly_charges": 75.5, "contract_type": "One year"}}'


📊 Expected Response

{
  "customer_id": "C12345",
  "churn": false,
  "confidence": 0.78
}


🛠 Dependencies

Install required packages:

pip install -r requirements.txt


🔐 Notes

Ensure port `8000` is open in both AWS Security Group and UFW firewall to allow external access.
