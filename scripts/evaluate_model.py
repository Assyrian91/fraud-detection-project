"""
Script to evaluate trained fraud detection model
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    auc, precision_recall_curve, average_precision_score
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.config import Config
from src.models.predict import FraudPredictor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate fraud detection model')

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to trained model file'
    )

    parser.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='Path to test data file'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold'
    )

    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save evaluation plots'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save evaluation results'
    )

    return parser.parse_args()


def plot_confusion_matrix(cm, output_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {output_path}")
    plt.close()


def plot_roc_curve(y_true, y_scores, output_path=None):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {output_path}")
    plt.close()

    return roc_auc


def plot_precision_recall_curve(y_true, y_scores, output_path=None):
    """Plot precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to {output_path}")
    plt.close()

    return avg_precision


def plot_threshold_metrics(y_true, y_scores, output_path=None):
    """Plot metrics vs threshold"""
    from sklearn.metrics import precision_score, recall_score, f1_score

    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Classification Threshold')
    plt.legend()
    plt.grid(alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Threshold metrics plot saved to {output_path}")
    plt.close()


def main():
    """Main evaluation pipeline"""
    args = parse_args()

    logger.info("="*50)
    logger.info("Starting Model Evaluation")
    logger.info("="*50)

    # Initialize config
    config = Config()

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else config.REPORTS_DIR / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictor
    logger.info("Loading model...")
    model_path = args.model_path if args.model_path else config.BEST_MODEL_FILE
    predictor = FraudPredictor(model_path=model_path)
    predictor.set_threshold(args.threshold)

    # Load test data
    logger.info("Loading test data...")
    test_data_path = args.test_data if args.test_data else config.TEST_DATA_FILE
    test_df = pd.read_csv(test_data_path)

    X_test = test_df.drop(config.TARGET_COLUMN, axis=1)
    y_test = test_df[config.TARGET_COLUMN]
