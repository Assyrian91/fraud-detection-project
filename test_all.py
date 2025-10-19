"""Quick test to verify everything works"""
import sys
from pathlib import Path

print("=" * 60)
print("🧪 Testing Fraud Detection Project")
print("=" * 60)

# Test 1: Imports
print("\n[1/8] Testing imports...")
try:
    from src.config import Config
    from data.preprocessing import *  # adjusted to match data/preprocessing.py
    from models.train import ModelTrainer
    from models.predict import FraudPredictor
    from src.features_engineering import FeatureEngineer
    from src.api.app import app
    from src.utils.helpers import load_json
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Config
print("\n[2/8] Testing Config...")
try:
    config = Config()
    print(f"✓ Project directory: {config.BASE_DIR}")
    print(f"✓ Data directory: {config.DATA_DIR}")
    print(f"✓ Models directory: {config.MODELS_DIR}")
except Exception as e:
    print(f"✗ Config failed: {e}")

# Test 3: Check directories
print("\n[3/8] Checking directories...")
directories = [
    'data', 'data/raw', 'data/processed', 'models',
    'logs', 'reports', 'notebooks', 'tests', 'src'
]
for dir_name in directories:
    if Path(dir_name).exists():
        print(f"✓ {dir_name}/")
    else:
        print(f"⚠ {dir_name}/ not found")

# Test 4: Check data file
print("\n[4/8] Checking data files...")
if Path("data/raw/creditcard.csv").exists():
    import pandas as pd
    df = pd.read_csv("data/raw/creditcard.csv")
    print(f"✓ Raw data found: {df.shape}")
    print(f"  - Rows: {len(df):,}")
    print(f"  - Columns: {len(df.columns)}")
    if "Class" in df.columns:
        print(f"  - Fraud cases: {df['Class'].sum():,} ({df['Class'].mean() * 100:.2f}%)")
else:
    print("⚠ creditcard.csv not found in data/raw/")

# Test 5: Check model files
print("\n[5/8] Checking trained models...")
model_file = Path("models/logistic_model.pkl")
scaler_file = Path("models/scaler.pkl")

if model_file.exists():
    print(f"✓ Model found: {model_file.name}")
else:
    print("⚠ No trained model (run training script)")

if scaler_file.exists():
    print(f"✓ Scaler found: {scaler_file.name}")
else:
    print("⚠ No scaler file (run training script)")

# Test 6: Test libraries
print("\n[6/8] Testing key libraries...")
try:
    import numpy as np
    import pandas as pd
    import sklearn
    import xgboost
    import lightgbm
    import fastapi
    import dash
    print(f"✓ NumPy: {np.__version__}")
    print(f"✓ Pandas: {pd.__version__}")
    print(f"✓ Scikit-learn: {sklearn.__version__}")
    print(f"✓ XGBoost: {xgboost.__version__}")
    print(f"✓ LightGBM: {lightgbm.__version__}")
    print(f"✓ FastAPI: {fastapi.__version__}")
    print(f"✓ Dash: {dash.__version__}")
except Exception as e:
    print(f"✗ Library test failed: {e}")

# Test 7: Initialize components
print("\n[7/8] Testing component initialization...")
try:
    preprocessor = None
    try:
        from data.preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        print("✓ DataPreprocessor initialized")
    except Exception:
        print("⚠ DataPreprocessor not found or failed to initialize")

    trainer = ModelTrainer()
    print("✓ ModelTrainer initialized")

    engineer = FeatureEngineer()
    print("✓ FeatureEngineer initialized")
except Exception as e:
    print(f"✗ Component initialization failed: {e}")

# Test 8: Check test files
print("\n[8/8] Checking test files...")
test_files = [
    'tests/__init__.py',
    'tests/preprocessing.py',
    'tests/test_model.py',
    'tests/test_api.py'
]
for test_file in test_files:
    if Path(test_file).exists():
        print(f"✓ {test_file}")
    else:
        print(f"⚠ {test_file} missing")

print("\n" + "=" * 60)
print("✓ All tests completed!")
print("=" * 60)
print("\n📋 Next steps:")
print("1. Run tests: pytest tests\\ -v")
print("2. Train model: python scripts\\train_model.py --verbose")
print("3. Start API: uvicorn src.api.app:app --reload")
print("4. Open notebook: jupyter notebook")
print("\n" + "=" * 60)
