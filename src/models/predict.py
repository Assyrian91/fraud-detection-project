"""
Model prediction module for fraud detection
"""
import sys
from pathlib import Path
from typing import Union, List, Dict

import pandas as pd
import numpy as np
import joblib
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.config import Config


class FraudPredictor:
    """Class for making fraud predictions"""

    def __init__(self, model_path=None, scaler_path=None):
        self.config = Config()

        # Load model
        if model_path is None:
            model_path = self.config.BEST_MODEL_FILE
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load scaler
        if scaler_path is None:
            scaler_path = self.config.SCALER_FILE
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")

        self.threshold = self.config.FRAUD_THRESHOLD

    def preprocess_input(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input must be dict, list of dicts, or DataFrame")

        required_features = self.config.FEATURE_COLUMNS

        # إضافة أي عمود مفقود بالقيمة صفر
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0
                logger.warning(f"Feature '{feature}' missing. Set to 0.")

        df = df[required_features]

        cols_to_scale = [col for col in df.columns if col != "Time"]
        if cols_to_scale:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

        return df

    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """Predict fraud (0 or 1)"""
        df = self.preprocess_input(data)
        predictions = self.model.predict(df)
        logger.info(f"Predictions completed. Shape: {predictions.shape}")
        return predictions

    def predict_proba(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """Get fraud probability scores"""
        df = self.preprocess_input(data)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(df)[:, 1]
        else:
            probabilities = self.model.decision_function(df)
            logger.warning("Model has no predict_proba. Using decision_function.")
        logger.info(f"Probabilities calculated. Shape: {probabilities.shape}")
        return probabilities

    def predict_with_confidence(self, data: Union[pd.DataFrame, Dict, List[Dict]], threshold: float = None) -> List[Dict]:
        if threshold is None:
            threshold = self.threshold

        probabilities = self.predict_proba(data)
        predictions = (probabilities >= threshold).astype(int)

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'transaction_id': i,
                'is_fraud': bool(pred),
                'fraud_probability': float(prob),
                'confidence': float(abs(prob - 0.5) * 2),
                'risk_level': self._get_risk_level(prob)
            })
        return results

    def _get_risk_level(self, probability: float) -> str:
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Medium"
        elif probability < 0.8:
            return "High"
        else:
            return "Critical"

    def predict_batch(self, data: pd.DataFrame, batch_size: int = 1000) -> np.ndarray:
        n_samples = len(data)
        predictions = np.zeros(n_samples)
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = data.iloc[i:end_idx]
            predictions[i:end_idx] = self.predict(batch)
            if (i // batch_size) % 10 == 0:
                logger.info(f"Processed {end_idx}/{n_samples} samples")
        logger.info("Batch predictions completed")
        return predictions

    def explain_prediction(self, data: Dict, feature_names: List[str] = None) -> Dict:
        result = self.predict_with_confidence(data)[0]
        df = self.preprocess_input(data)
        explanation = {'prediction': result, 'input_features': data}
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(df.columns, self.model.feature_importances_))
            explanation['top_features'] = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        return explanation

    def set_threshold(self, new_threshold: float):
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        logger.info(f"Threshold updated from {self.threshold} to {new_threshold}")
        self.threshold = new_threshold


if __name__ == '__main__':
    predictor = FraudPredictor()

    transaction = {feature: 0 for feature in predictor.config.FEATURE_COLUMNS}

    transaction.update({
        'V1': -1.359807,
        'V2': -0.072781,
        'V3': 2.536347,
        'V4': 1.378155,
        'Amount': 149.62,
    })

    result = predictor.predict_with_confidence(transaction)
    print(result)
