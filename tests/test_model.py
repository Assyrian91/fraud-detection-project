"""
Unit tests for model training and prediction
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.models.train import ModelTrainer
from src.models.predict import FraudPredictor
from src.config import Config

@pytest.fixture
def sample_train_data():
“”“Create sample training data”””
np.random.seed(42)
n_samples = 500

```
X = pd.DataFrame({
    f'V{i}': np.random.randn(n_samples) for i in range(1, 29)
})
X['Amount'] = np.random.exponential(50, n_samples)

y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.9, 0.1]))

return X, y
```

@pytest.fixture
def trainer():
“”“Create ModelTrainer instance”””
return ModelTrainer()

class TestModelTrainer:
“”“Test cases for ModelTrainer class”””

```
def test_initialization(self, trainer):
    """Test trainer initialization"""
    assert trainer is not None
    assert trainer.config is not None
    assert trainer.models == {}

def test_get_models_returns_dict(self, trainer):
    """Test that get_models returns model dictionary"""
    models = trainer.get_models()
    
    assert isinstance(models, dict)
    assert len(models) > 0
    assert 'Logistic Regression' in models
    assert 'Random Forest' in models

def test_train_single_model(self, trainer, sample_train_data):
    """Test training a single model"""
    X, y = sample_train_data
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    trained_model = trainer.train_single_model(model, X, y, "Test Model")
    
    assert trained_model is not None
    assert hasattr(trained_model, 'predict')

def test_cross_validate_model(self, trainer, sample_train_data):
    """Test cross-validation"""
    X, y = sample_train_data
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = trainer.cross_validate_model(model, X, y, "Test Model")
    
    assert isinstance(cv_scores, dict)
    assert 'accuracy' in cv_scores
    assert 'f1' in cv_scores
    assert 'mean' in cv_scores['accuracy']
    assert 'std' in cv_scores['accuracy']

def test_evaluate_model(self, trainer, sample_train_data):
    """Test model evaluation"""
    X, y = sample_train_data
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    metrics, cm, report, y_pred, y_pred_proba = trainer.evaluate_model(
        model, X_test, y_test, "Test Model"
    )
    
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert cm.shape == (2, 2)
    assert len(y_pred) == len(y_test)

def test_save_and_load_model(self, trainer, sample_train_data, tmp_path):
    """Test model saving and loading"""
    X, y = sample_train_data
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    # Save model
    model_path = tmp_path / "test_model.pkl"
    trainer.best_model = model
    trainer.best_model_name = "Test Model"
    trainer.save_model(filepath=model_path)
    
    assert model_path.exists()
    
    # Load model
    loaded_model = trainer.load_model(model_path)
    
    # Test predictions are the same
    original_pred = model.predict(X)
    loaded_pred = loaded_model.predict(X)
    
    np.testing.assert_array_equal(original_pred, loaded_pred)
```

class TestFraudPredictor:
“”“Test cases for FraudPredictor class”””

```
@pytest.fixture
def predictor(self, tmp_path, sample_train_data):
    """Create predictor with temporary model"""
    X, y = sample_train_data
    
    # Train and save model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    model_path = tmp_path / "model.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    
    import joblib
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    return FraudPredictor(model_path, scaler_path)

def test_predictor_initialization(self, predictor):
    """Test predictor initialization"""
    assert predictor is not None
    assert predictor.model is not None
    assert predictor.scaler is not None

def test_predict_single_transaction(self, predictor):
    """Test single transaction prediction"""
    transaction = {f'V{i}': np.random.randn() for i in range(1, 29)}
    transaction['Amount'] = 100.0
    
    prediction = predictor.predict(transaction)
    
    assert prediction is not None
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]

def test_predict_proba(self, predictor):
    """Test probability prediction"""
    transaction = {f'V{i}': np.random.randn() for i in range(1, 29)}
    transaction['Amount'] = 100.0
    
    probabilities = predictor.predict_proba(transaction)
    
    assert probabilities is not None
    assert len(probabilities) == 1
    assert 0 <= probabilities[0] <= 1

def test_predict_with_confidence(self, predictor):
    """Test prediction with confidence"""
    transaction = {f'V{i}': np.random.randn() for i in range(1, 29)}
    transaction['Amount'] = 100.0
    
    results = predictor.predict_with_confidence(transaction)
    
    assert len(results) == 1
    result = results[0]
    
    assert 'is_fraud' in result
    assert 'fraud_probability' in result
    assert 'confidence' in result
    assert 'risk_level' in result
    assert isinstance(result['is_fraud'], bool)
    assert 0 <= result['fraud_probability'] <= 1
    assert 0 <= result['confidence'] <= 1

def test_predict_batch(self, predictor):
    """Test batch prediction"""
    transactions = pd.DataFrame({
        f'V{i}': np.random.randn(10) for i in range(1, 29)
    })
    transactions['Amount'] = np.random.exponential(50, 10)
    
    predictions = predictor.predict_batch(transactions, batch_size=5)
    
    assert len(predictions) == 10
    assert all(p in [0, 1] for p in predictions)

def test_set_threshold(self, predictor):
    """Test threshold setting"""
    new_threshold = 0.7
    predictor.set_threshold(new_threshold)
    
    assert predictor.threshold == new_threshold

def test_invalid_threshold(self, predictor):
    """Test invalid threshold raises error"""
    with pytest.raises(ValueError):
        predictor.set_threshold(1.5)
    
    with pytest.raises(ValueError):
        predictor.set_threshold(-0.1)
```

if **name** == “**main**”:
pytest.main([**file**, “-v”])
