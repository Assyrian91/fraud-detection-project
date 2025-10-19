# 📥 Data Setup Instructions

## Required Dataset

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

**⚠️ Dataset is NOT included in this repository due to its large size (150 MB)**

---

## 🔽 Download Options

### Option 1: Kaggle Website (Recommended)

1. Visit: https://www.kaggle.com/mlg-ulb/creditcardfraud
2. Click **Download** button
3. Extract and place `creditcard.csv` in: `data/raw/creditcard.csv`

### Option 2: Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Configure API credentials (https://www.kaggle.com/docs/api)
# Download dataset
kaggle datasets download -d mlg-ulb/creditcardfraud

# Extract
unzip creditcardfraud.zip -d data/raw/

# Verify
python -c "import pandas as pd; df = pd.read_csv('data/raw/creditcard.csv'); print(f'✅ Data loaded: {df.shape}')"

📊 Dataset Info
	•	Rows: 284,807 transactions
	•	Columns: 31 features
	•	Time, V1-V28 (PCA transformed), Amount, Class
	•	Fraud Cases: 492 (0.172%)
	•	File Size: ~150 MB
✅ Verify Setup

python -c "from pathlib import Path; print('✅ Data ready!' if Path('data/raw/creditcard.csv').exists() else '❌ Download data first')"

🚀 Next Steps
After downloading data:
# Train model
python scripts/train_model.py --balance-method smote --verbose

# Evaluate model
python scripts/evaluate_model.py --save-plots

# Start API
uvicorn src.api.app:app --reload