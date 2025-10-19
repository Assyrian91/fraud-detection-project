"""
Utility functions for fraud detection system
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger
import shutil


def load_json(filepath: Path) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict, filepath: Path):
    """Save data to JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Data saved to {filepath}")


def calculate_fraud_statistics(y_true: pd.Series) -> Dict[str, Any]:
    """
    Calculate fraud statistics from labels

    Args:
        y_true: True labels

    Returns:
        Dictionary with fraud statistics
    """
    total = len(y_true)
    fraud_count = y_true.sum()
    normal_count = total - fraud_count
    fraud_ratio = fraud_count / total

    return {
        'total_transactions': int(total),
        'fraud_transactions': int(fraud_count),
        'normal_transactions': int(normal_count),
        'fraud_ratio': float(fraud_ratio),
        'imbalance_ratio': float(normal_count / fraud_count) if fraud_count > 0 else float('inf')
    }


def format_confusion_matrix(cm: np.ndarray) -> Dict[str, int]:
    """
    Format confusion matrix into dictionary

    Args:
        cm: Confusion matrix array

    Returns:
        Dictionary with confusion matrix values
    """
    tn, fp, fn, tp = cm.ravel()
    return {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'total': int(cm.sum())
    }


def calculate_cost_matrix(cm: np.ndarray,
                          fp_cost: float = 10.0,
                          fn_cost: float = 100.0) -> Dict[str, float]:
    """
    Calculate cost metrics based on confusion matrix

    Args:
        cm: Confusion matrix
        fp_cost: Cost of false positive
        fn_cost: Cost of false negative

    Returns:
        Dictionary with cost metrics
    """
    tn, fp, fn, tp = cm.ravel()
    total_cost = (fp * fp_cost) + (fn * fn_cost)
    avg_cost_per_transaction = total_cost / cm.sum()

    return {
        'false_positive_cost': float(fp * fp_cost),
        'false_negative_cost': float(fn * fn_cost),
        'total_cost': float(total_cost),
        'average_cost_per_transaction': float(avg_cost_per_transaction),
        'fp_unit_cost': fp_cost,
        'fn_unit_cost': fn_cost
    }


def generate_report_timestamp() -> str:
    """Generate timestamp for reports"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_id() -> str:
    """Create unique experiment ID"""
    return f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def log_experiment(experiment_id: str,
                   config: Dict,
                   metrics: Dict,
                   output_dir: Path):
    """
    Log experiment details

    Args:
        experiment_id: Unique experiment identifier
        config: Experiment configuration
        metrics: Experiment metrics
        output_dir: Directory to save logs
    """
    experiment_log = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'metrics': metrics
    }

    log_file = output_dir / 'experiments' / f'{experiment_id}.json'
    save_json(experiment_log, log_file)


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple model results

    Args:
        results: Dictionary of model results

    Returns:
        DataFrame with comparison
    """
    comparison_data = []

    for model_name, model_data in results.items():
        metrics = model_data.get('test_metrics', {})
        row = {
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1 Score': metrics.get('f1', 0),
            'ROC AUC': metrics.get('roc_auc', 0)
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    df = df.sort_values('F1 Score', ascending=False)

    return df


def find_optimal_threshold(y_true: np.ndarray,
                           y_scores: np.ndarray,
                           metric: str = 'f1') -> float:
    """
    Find optimal classification threshold

    Args:
        y_true: True labels
        y_scores: Prediction scores
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        Optimal threshold
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    thresholds = np.linspace(0, 1, 100)
    scores = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]

    logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.4f}")
    return float(optimal_threshold)


def validate_dataframe(df: pd.DataFrame,
                       required_columns: List[str]) -> bool:
    """
    Validate DataFrame has required columns

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from model

    Args:
        model: Trained model
        feature_names: List of feature names

    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        logger.warning("Model doesn't have feature importance attribute")
        return pd.DataFrame()

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    df = df.sort_values('importance', ascending=False)
    return df


def memory_usage_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate memory usage report for DataFrame

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with memory usage information
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    column_memory = df.memory_usage(deep=True).to_dict()
    column_memory_mb = {k: v / 1024**2 for k, v in column_memory.items()}

    return {
        'total_memory_mb': float(memory_mb),
        'shape': df.shape,
        'columns': len(df.columns),
        'rows': len(df),
        'column_memory_mb': column_memory_mb
    }


def sample_stratified(df: pd.DataFrame,
                      target_column: str,
                      sample_size: int,
                      random_state: int = 42) -> pd.DataFrame:
    """
    Sample DataFrame while maintaining class distribution

    Args:
        df: Input DataFrame
        target_column: Name of target column
        sample_size: Number of samples to draw
        random_state: Random seed

    Returns:
        Sampled DataFrame
    """
    return df.groupby(target_column, group_keys=False).apply(
        lambda x: x.sample(
            min(len(x), int(sample_size * len(x) / len(df))),
            random_state=random_state
        )
    )


def create_backup(filepath: Path, backup_dir: Optional[Path] = None):
    """
    Create backup of file

    Args:
        filepath: File to backup
        backup_dir: Directory to store backup
    """
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return

    if backup_dir is None:
        backup_dir = filepath.parent / 'backups'

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{filepath.stem}_{timestamp}{filepath.suffix}"
    backup_path = backup_dir / backup_filename

    shutil.copy2(filepath, backup_path)
    logger.info(f"Backup created: {backup_path}")


if __name__ == "__main__":
    # Example usage
    print("Utility functions loaded successfully")
