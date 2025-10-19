"""
Unit tests for data preprocessing module
"""
import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import DataPreprocessor
from src.config import Config

@pytest.fixture
def sample_data():
“”“Create sample transaction data for testing”””
np.random.seed(42)

```
n_samples = 1000
data = {
    'Time': np.random.randint(0, 172800, n_samples),
    'Amount': np.random.exponential(50, n_samples),
    'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
}

# Add V features
for i in range(1, 29):
    data[f'V{i}'] = np.random.randn(n_samples)

return pd.DataFrame(data)
```

@pytest.fixture
def preprocessor():
“”“Create preprocessor instance”””
return DataPreprocessor()

class TestDataPreprocessor:
“”“Test cases for DataPreprocessor class”””

```
def test_initialization(self, preprocessor):
    """Test preprocessor initialization"""
    assert preprocessor is not None
    assert preprocessor.scaler is not None
    assert preprocessor.config is not None

def test_clean_data_removes_duplicates(self, preprocessor, sample_data):
    """Test that clean_data removes duplicate rows"""
    # Add duplicate rows
    df_with_dupes = pd.concat([sample_data, sample_data.iloc[:10]], ignore_index=True)
    
    df_cleaned = preprocessor.clean_data(df_with_dupes)
    
    assert len(df_cleaned) < len(df_with_dupes)
    assert df_cleaned.duplicated().sum() == 0

def test_clean_data_handles_missing_values(self, preprocessor, sample_data):
    """Test that clean_data handles missing values"""
    # Introduce missing values
    sample_data.loc[0:5, 'V1'] = np.nan
    
    df_cleaned = preprocessor.clean_data(sample_data)
    
    assert df_cleaned.isnull().sum().sum() == 0

def test_split_data_maintains_class_ratio(self, preprocessor, sample_data):
    """Test that split maintains fraud ratio"""
    X_train, X_test, y_train, y_test = preprocessor.split_data(sample_data)
    
    # Check shapes
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) + len(y_test) == len(sample_data)
    
    # Check fraud ratio is similar
    train_ratio = y_train.sum() / len(y_train)
    test_ratio = y_test.sum() / len(y_test)
    original_ratio = sample_data['Class'].sum() / len(sample_data)
    
    assert abs(train_ratio - original_ratio) < 0.01
    assert abs(test_ratio - original_ratio) < 0.01

def test_scale_features_normalizes_data(self, preprocessor, sample_data):
    """Test that scaling normalizes features"""
    X_train, X_test, y_train, y_test = preprocessor.split_data(sample_data)
    
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Check scaling (mean should be close to 0, std close to 1)
    for col in [c for c in X_train_scaled.columns if c != 'Time']:
        assert abs(X_train_scaled[col].mean()) < 0.1
        assert abs(X_train_scaled[col].std() - 1.0) < 0.1

def test_handle_imbalance_increases_fraud_samples(self, preprocessor, sample_data):
    """Test that SMOTE increases minority class"""
    X_train, X_test, y_train, y_test = preprocessor.split_data(sample_data)
    
    initial_fraud_count = y_train.sum()
    
    X_balanced, y_balanced = preprocessor.handle_imbalance(X_train, y_train, method='smote')
    
    final_fraud_count = y_balanced.sum()
    
    assert final_fraud_count > initial_fraud_count

def test_preprocess_pipeline_returns_correct_shapes(self, preprocessor, sample_data):
    """Test complete preprocessing pipeline"""
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        sample_data, 
        balance_method='smote',
        save_artifacts=False
    )
    
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

def test_save_and_load_scaler(self, preprocessor, sample_data, tmp_path):
    """Test scaler saving and loading"""
    X_train, X_test, y_train, y_test = preprocessor.split_data(sample_data)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Save scaler
    scaler_path = tmp_path / "test_scaler.pkl"
    preprocessor.save_scaler(scaler_path)
    
    assert scaler_path.exists()
    
    # Create new preprocessor and load scaler
    new_preprocessor = DataPreprocessor()
    new_preprocessor.load_scaler(scaler_path)
    
    # Test that loaded scaler works
    X_test_scaled_new = X_test.copy()
    cols_to_scale = [col for col in X_test.columns if col != 'Time']
    X_test_scaled_new[cols_to_scale] = new_preprocessor.scaler.transform(X_test[cols_to_scale])
    
    pd.testing.assert_frame_equal(X_test_scaled, X_test_scaled_new)
```

if **name** == “**main**”:
pytest.main([**file**, “-v”])
