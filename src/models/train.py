"""
Model training module for fraud detection
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import joblib
from loguru import logger
import json
from datetime import datetime
from pathlib import Path

from src.config import Config

class ModelTrainer:
    """Class for training and evaluating ML models"""

    def __init__(self):
        self.config = Config()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.cv_results = {}

    def get_models(self):
        """Initialize different models for comparison"""
        logger.info("Initializing models...")

        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=self.config.MAX_ITER,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS,
                scale_pos_weight=1
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS,
                class_weight='balanced',
                verbose=-1
            )
        }

        self.models = models
        return models

    def train_single_model(self, model, X_train, y_train, model_name="Model"):
        """Train a single model"""
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        logger.info(f"{model_name} training completed")
        return model

    def cross_validate_model(self, model, X_train, y_train, model_name="Model"):
        """Perform cross-validation on a model"""
        logger.info(f"Cross-validating {model_name}...")
        cv = StratifiedKFold(n_splits=self.config.CV_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE)

        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_scores = {}

        for score in scoring:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=score, n_jobs=self.config.N_JOBS)
            cv_scores[score] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            logger.info(f"{model_name} - {score}: {scores.mean():.4f} (+/- {scores.std():.4f})")

        self.cv_results[model_name] = cv_scores
        return cv_scores

    def hyperparameter_tuning(self, model_name, X_train, y_train):
        """Perform hyperparameter tuning using GridSearchCV"""
        logger.info(f"Performing hyperparameter tuning for {model_name}...")

        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, -1],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [31, 50, 100]
            }
        }

        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return self.models[model_name]

        base_model = self.models[model_name]
        param_grid = param_grids[model_name]

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.RANDOM_STATE)

        grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='f1', n_jobs=self.config.N_JOBS, verbose=1)
        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best F1 score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate model on test set"""
        logger.info(f"Evaluating {model_name}...")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_test, y_pred)
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.info(f"\n{model_name} Evaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}")
        logger.info(f"\nConfusion Matrix:\n{cm}")

        return metrics, cm, report, y_pred, y_pred_proba

    def train_and_compare_models(self, X_train, y_train, X_test, y_test):
        """Train and compare multiple models"""
        logger.info("Starting model comparison...")
        self.get_models()
        results = {}

        for model_name, model in self.models.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing: {model_name}")
            logger.info(f"{'='*50}")

            cv_scores = self.cross_validate_model(model, X_train, y_train, model_name)
            trained_model = self.train_single_model(model, X_train, y_train, model_name)
            metrics, cm, report, y_pred, y_pred_proba = self.evaluate_model(trained_model, X_test, y_test, model_name)

            results[model_name] = {
                'model': trained_model,
                'cv_scores': cv_scores,
                'test_metrics': metrics,
                'confusion_matrix': cm.tolist(),
                'classification_report': report
            }

        best_model_name = max(results.keys(), key=lambda x: results[x]['test_metrics']['f1'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name

        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"F1 Score: {results[best_model_name]['test_metrics']['f1']:.4f}")

        return results

    def save_model(self, model=None, filepath=None, model_name=None):
        """Save trained model"""
        if model is None:
            model = self.best_model
            model_name = self.best_model_name

        if filepath is None:
            filepath = Path(self.config.BEST_MODEL_FILE)
        else:
            filepath = Path(filepath)

        joblib.dump(model, filepath)
        logger.info(f"Model '{model_name}' saved to {filepath}")

        metadata = {
            'model_name': model_name,
            'saved_at': datetime.now().isoformat(),
            'config': {
                'random_state': self.config.RANDOM_STATE,
                'test_size': self.config.TEST_SIZE
            }
        }

        metadata_file = filepath.parent / f"{filepath.stem}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Metadata saved to {metadata_file}")

    def load_model(self, filepath=None):
        """Load a trained model"""
        if filepath is None:
            filepath = Path(self.config.BEST_MODEL_FILE)
        else:
            filepath = Path(filepath)

        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model

    def save_results(self, results):
        """Save training results to JSON"""
        results_file = Path(self.config.REPORTS_DIR) / "metrics" / "training_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        json_results = {}
        for model_name, model_data in results.items():
            json_results[model_name] = {
                'cv_scores': model_data['cv_scores'],
                'test_metrics': model_data['test_metrics'],
                'confusion_matrix': model_data['confusion_matrix'],
                'classification_report': model_data['classification_report']
            }

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=4)

        logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    from data.preprocessing import DataPreprocessor

    # Load and preprocess data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(df)

    # Train models
    trainer = ModelTrainer()
    results = trainer.train_and_compare_models(X_train, y_train, X_test, y_test)

    # Save best model
    trainer.save_model()
    trainer.save_results(results)
